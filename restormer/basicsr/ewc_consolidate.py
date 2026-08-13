"""Consolidate the diagonal (empirical) Fisher for EWC after each training stage.

Examples:
Consolidate EWC-Restormer on D1:
    PYTHONPATH=. python ./basicsr/ewc_consolidate.py \
        -opt ./Motion_Deblurring/Options/Degradations_D1_Consolidate_EWC_Restormer.yml \
        --ckpt ./experiments/archived_checkpoints/Degradations_D1_only/gaussian_color_denoising_blind.pth \
        --out ./experiments/archived_checkpoints/ewc_states/ewc_state_D1_restormer.pth
        --strict_load false

Consolidate EWC-NAFNet on D1:
    PYTHONPATH=. python ./basicsr/ewc_consolidate.py \
        -opt ./Motion_Deblurring/Options/Degradations_D1_Consolidate_EWC_NAFNet.yml \
        --ckpt ./experiments/archived_checkpoints/Degradations_D1_only/NAFNet-SIDD-width32.pth \
        --out ./experiments/archived_checkpoints/ewc_states/ewc_state_D1_nafnet.pth

Consolidate EWC-Restormer on D2:
    PYTHONPATH=. python ./basicsr/ewc_consolidate.py \
        -opt ./Motion_Deblurring/Options/Degradations_D1_ft_D2_EWC_Restormer.yml \
        --ckpt ./experiments/archived_checkpoints/Degradations_D1_ft_D2_EWC_Restormer/net_g_latest.pth \
        --prev_ewc ./experiments/archived_checkpoints/ewc_states/ewc_state_D1_restormer.pth \
        --out ./experiments/archived_checkpoints/ewc_states/ewc_state_D1_ft_D2_restormer.pth

Consolidate EWC-NAFNet on D2:
    PYTHONPATH=. python ./basicsr/ewc_consolidate.py \
        -opt ./Motion_Deblurring/Options/Degradations_D1_ft_D2_EWC_NAFNet.yml \
        --ckpt ./experiments/archived_checkpoints/Degradations_D1_ft_D2_EWC_NAFNet/net_g_latest.pth \
        --prev_ewc ./experiments/archived_checkpoints/ewc_states/ewc_state_D1_nafnet.pth \
        --out ./experiments/archived_checkpoints/ewc_states/ewc_state_D1_ft_D2_nafnet.pth
"""

import argparse
import os
from copy import deepcopy

import torch
import yaml

from basicsr.data import create_dataloader, create_dataset
from basicsr.models.archs import define_network


def _load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _build_train_loader(opt):
    """Reconstruct the stage's *train* dataloader from its YAML."""
    dataset_opt = deepcopy(opt['datasets']['train'])
    dataset_opt['phase'] = 'train'
    dataset_opt.setdefault('scale', opt.get('scale', 1))
    # Force batch size 1 so each backward gives a per-sample gradient; this
    # keeps the Fisher a clean per-example expectation and is memory-light.
    dataset_opt['batch_size_per_gpu'] = 1
    dataset_opt['use_shuffle'] = True
    dataset_opt['num_worker_per_gpu'] = dataset_opt.get('num_worker_per_gpu', 4)
    dataset_opt['dataset_enlarge_ratio'] = 1

    train_set = create_dataset(dataset_opt)
    loader = create_dataloader(
        train_set, dataset_opt, num_gpu=1, dist=False, sampler=None,
        seed=opt.get('manual_seed', 0))
    return loader


@torch.enable_grad()
def compute_fisher(net, loader, device, num_samples):
    """Diagonal empirical Fisher = E[(d L1 / d theta)^2]."""
    net.eval()  # Restormer/NAFNet have no BN/dropout: eval==train for grads
    for p in net.parameters():
        p.requires_grad_(True)

    fisher = {n: torch.zeros_like(p, device=device)
              for n, p in net.named_parameters() if p.requires_grad}

    seen = 0
    for data in loader:
        if seen >= num_samples:
            break
        lq = data['lq'].to(device)
        gt = data['gt'].to(device)

        net.zero_grad(set_to_none=True)
        out = net(lq)
        if isinstance(out, list):
            out = out[-1]
        loss = torch.nn.functional.l1_loss(out, gt)
        loss.backward()

        for n, p in net.named_parameters():
            if p.grad is not None and n in fisher:
                fisher[n] += p.grad.detach() ** 2
        seen += 1

    if seen == 0:
        raise RuntimeError('No samples were drawn for Fisher estimation.')
    for n in fisher:
        fisher[n] /= float(seen)

    print(f'Estimated Fisher over {seen} samples.')
    return fisher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', required=True,
                        help='YAML of the stage just trained (for its dataset).')
    parser.add_argument('--ckpt', required=True,
                        help='Trained net_g checkpoint of this stage.')
    parser.add_argument('--prev_ewc', default=None,
                        help='Previous ewc_state_*.pth, or ~ / omit for base.')
    parser.add_argument('--out', required=True, help='Output ewc_state path.')
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Online-EWC decay on the previous Fisher.')
    parser.add_argument('--param_key', default='params')
    parser.add_argument('--strict_load', type=lambda x: str(x).lower() == 'true', default=True)
    args = parser.parse_args()

    prev_path = None if args.prev_ewc in (None, '~', 'None', '') else args.prev_ewc

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = _load_yaml(args.opt)

    # Build the network from the YAML and load this stage's trained weights.
    net = define_network(deepcopy(opt['network_g']))
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt.get(args.param_key, ckpt)
    state_dict = {k[7:] if k.startswith('module.') else k: v
                  for k, v in state_dict.items()}
    net = net.to(device)

    # Anchor theta* = this stage's trained weights (detached constants).
    theta_star = {n: p.detach().clone().cpu()
                  for n, p in net.named_parameters() if p.requires_grad}

    loader = _build_train_loader(opt)
    fisher_task = compute_fisher(net, loader, device, args.num_samples)
    fisher_task = {n: f.cpu() for n, f in fisher_task.items()}

    # Online consolidation: F <- gamma * F_prev + F_task ; anchor <- latest.
    if prev_path is not None:
        prev = torch.load(prev_path, map_location='cpu')
        prev_fisher = prev['fisher']
        fisher = {}
        for n, f in fisher_task.items():
            if n in prev_fisher:
                fisher[n] = args.gamma * prev_fisher[n] + f
            else:
                fisher[n] = f
        print(f'Merged with previous Fisher (gamma={args.gamma}).')
    else:
        fisher = fisher_task
        print('No previous Fisher; this is the base-task consolidation.')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({'fisher': fisher, 'theta_star': theta_star,
                'gamma': args.gamma, 'source_ckpt': args.ckpt}, args.out)
    print(f'Saved consolidated EWC state to {args.out}')


if __name__ == '__main__':
    main()
