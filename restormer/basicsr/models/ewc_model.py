import importlib
from collections import OrderedDict

import torch

from basicsr.models.image_restoration_model import ImageCleanModel
from basicsr.utils import get_root_logger

loss_module = importlib.import_module('basicsr.models.losses')


class EWCModel(ImageCleanModel):
    """Elastic Weight Consolidation (Kirkpatrick et al., PNAS 2017) baseline.

    Regularization-based continual learning. Forgetting is resisted 
    by a quadratic penalty that anchors each parameter near its
    previous-task optimum, weighted by its diagonal Fisher information:
        L = L_task + (lambda / 2) * sum_i  F_i * (theta_i - theta*_i)^2

    Configs:
          lambda_ewc: !!float 1e3     # penalty strength (REQUIRES a sweep;
                                      # empirical-Fisher scale with L1 is small,
                                      # typical useful range ~1e2 - 1e5)
          prev_ewc_state: ~           # path to previous ewc_state_*.pth
    """

    def init_training_settings(self):
        # Parent sets up net_g (holding previous-stage weights), pixel loss,
        # optimizer, schedulers and optional EMA.
        super(EWCModel, self).init_training_settings()

        logger = get_root_logger()
        ewc_opt = self.opt['train'].get('ewc', {}) or {}

        self.ewc_lambda = float(ewc_opt.get('lambda_ewc', 0.0))
        prev_state_path = ewc_opt.get('prev_ewc_state', None)

        self.fisher = None
        self.theta_star = None

        if prev_state_path is None or self.ewc_lambda <= 0:
            logger.info(
                'EWCModel: penalty INACTIVE for this stage '
                '(base task or lambda_ewc<=0). Behaves as naive sequential '
                'full fine-tuning.')
            return

        state = torch.load(prev_state_path, map_location='cpu')
        fisher = state['fisher']
        theta_star = state['theta_star']

        # Match against the *bare* model's parameter names (strip DDP prefix)
        self.fisher = {k: v.to(self.device) for k, v in fisher.items()}
        self.theta_star = {k: v.to(self.device) for k, v in theta_star.items()}
        for t in self.fisher.values():
            t.requires_grad_(False)
        for t in self.theta_star.values():
            t.requires_grad_(False)

        # Coverage check: the Fisher/anchor are keyed by the parameter names of
        # the network they were consolidated on. A near-zero overlap almost
        # always means a wrong-backbone state (e.g. a Restormer state loaded
        # into a NAFNet net), which would otherwise silently evaluate the
        # penalty to 0 and degrade EWC to naive fine-tuning.
        bare = self.get_bare_model(self.net_g)
        net_names = {n for n, p in bare.named_parameters() if p.requires_grad}
        covered = net_names & set(self.fisher.keys())
        frac = len(covered) / max(len(net_names), 1)
        if frac < 0.5:
            raise ValueError(
                f'EWC Fisher covers only {frac:.1%} of net_g trainable '
                f'parameters ({len(covered)}/{len(net_names)}). This is almost '
                f'certainly a backbone/arch mismatch (wrong prev_ewc_state for '
                f'this network_g). Check that {prev_state_path} was consolidated '
                f'on the same backbone.')
        if frac < 0.98:
            logger.warning(
                f'EWC Fisher covers {frac:.1%} of net_g trainable parameters '
                f'({len(covered)}/{len(net_names)}). The uncovered params are '
                f'left unregularized (e.g. LN biases when D1 was consolidated in '
                f'BiasFree space). Confirm this is intended.')

        logger.info(
            f'EWCModel: penalty ACTIVE | lambda_ewc={self.ewc_lambda} | '
            f'anchored {len(covered)}/{len(net_names)} params ({frac:.1%}) '
            f'loaded Fisher+anchor for {len(self.fisher)} tensors from '
            f'{prev_state_path}')

    def _ewc_penalty(self):
        bare = self.get_bare_model(self.net_g)
        penalty = None
        for name, p in bare.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.fisher:
                continue
            term = (self.fisher[name] * (p - self.theta_star[name]) ** 2).sum()
            penalty = term if penalty is None else penalty + term
        if penalty is None:
            return torch.zeros((), device=self.device)
        return 0.5 * self.ewc_lambda * penalty

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]
        self.output = preds[-1]

        loss_dict = OrderedDict()

        l_pix = 0.
        for pred in preds:
            l_pix = l_pix + self.cri_pix(pred, self.gt)
        loss_dict['l_pix'] = l_pix
        l_total = l_pix

        if self.fisher is not None:
            l_ewc = self._ewc_penalty()
            loss_dict['l_ewc'] = l_ewc
            l_total = l_total + l_ewc

        l_total.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)