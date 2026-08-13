import importlib
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn.functional as F

from basicsr.models.image_restoration_model import ImageCleanModel
from basicsr.utils import get_root_logger

loss_module = importlib.import_module('basicsr.models.losses')


class LwFModel(ImageCleanModel):
    """Learning without Forgetting (Li & Hoiem, TPAMI 2017) baseline.

    Configs:
        enabled: true  # auto-disabled when pretrain_network_g is None
        lambda_kd: 1.0  # distillation weight
        distill_type: L1  # L1 (default) or L2
    """

    def init_training_settings(self):
        # Parent sets up net_g (already holding the previous-stage weights,
        # loaded in __init__ before this call), pixel loss, optimizer,
        # schedulers and optional EMA.
        super(LwFModel, self).init_training_settings()

        logger = get_root_logger()
        train_opt = self.opt['train']
        lwf_opt = train_opt.get('lwf', {}) or {}

        self.lambda_kd = float(lwf_opt.get('lambda_kd', 1.0))
        distill_type = lwf_opt.get('distill_type', 'L1').upper()
        if distill_type == 'L1':
            self._distill = F.l1_loss
        elif distill_type in ('L2', 'MSE'):
            self._distill = F.mse_loss
        else:
            raise ValueError(f'Unsupported LwF distill_type: {distill_type}')

        has_prev = self.opt['path'].get('pretrain_network_g', None) is not None
        self.lwf_enabled = bool(lwf_opt.get('enabled', True)) and has_prev

        if not self.lwf_enabled:
            self.net_g_old = None
            logger.info(
                'LwFModel: distillation DISABLED for this stage '
                '(base task or lwf.enabled=false). Behaves as naive '
                'sequential full fine-tuning.')
            return

        # Frozen teacher = deep copy of net_g *now*, i.e. before any training,
        # so it holds exactly the previous-stage (pretrain_network_g) weights.
        self.net_g_old = deepcopy(self.get_bare_model(self.net_g))
        self.net_g_old = self.net_g_old.to(self.device)
        self.net_g_old.eval()
        for p in self.net_g_old.parameters():
            p.requires_grad_(False)

        logger.info(
            f'LwFModel: distillation ENABLED | lambda_kd={self.lambda_kd} | '
            f'distill_type={distill_type}. Frozen teacher snapshot taken '
            f'from pretrain_network_g.')

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]
        self.output = preds[-1]

        loss_dict = OrderedDict()

        # Task pixel loss
        l_pix = 0.
        for pred in preds:
            l_pix = l_pix + self.cri_pix(pred, self.gt)
        loss_dict['l_pix'] = l_pix
        l_total = l_pix

        # LwF knowledge-distillation term on the new-task inputs.
        if self.lwf_enabled:
            with torch.no_grad():
                old_preds = self.net_g_old(self.lq)
                if isinstance(old_preds, list):
                    old_preds = old_preds[-1]
            l_kd = self.lambda_kd * self._distill(self.output, old_preds)
            loss_dict['l_kd'] = l_kd
            l_total = l_total + l_kd

        l_total.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
