### Example code for testing cross-degradation domains using Restormer
### Standard backbone testing (evaluate D1 -> D2 Restormer on D2 data):
###   PYTHONPATH=.. python test_degradations.py \
###      --weights ../experiments/archived_checkpoints/Degradations_D1_ft_D2/net_g_latest.pth \
###      --yaml_file Options/Degradations_D1_ft_D2_Restormer.yml \
###      --domain D2_deblur --save_images
###
### Adapter testing (evaluate D2 adapter on D2 data):
###   PYTHONPATH=.. python test_degradations.py \
###      --weights ../experiments/archived_checkpoints/Degradations_D1_ft_D2_Adapters/net_g_latest.pth \
###      --yaml_file Options/Degradations_D1_ft_D2_Restormer_Adapters.yml \
###      --domain D2_deblur --use_adapters --adapter_id 0
###
### Adapter testing (evaluate on D1 data with backbone only, no adapters):
###   PYTHONPATH=.. python test_degradations.py \
###      --weights ../experiments/archived_checkpoints/Degradations_D1_ft_D2_Adapters/net_g_latest.pth \
###      --yaml_file Options/Degradations_D1_ft_D2_Restormer_Adapters.yml \
###      --domain D1_denoise --use_adapters --adapter_id -1

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from basicsr.models.archs.restormer_adapters_arch import RestormerAdapters
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2


VALID_DOMAINS = [
    "D1_denoise",
    "D2_deblur",
    "D3_derain",
    "D4_dehaze",
    "D5_lowlight",
]


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_config(yaml_path):
    import yaml
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    with open(yaml_path, mode='r') as f:
        return yaml.load(f, Loader=Loader)


def build_model(args):
    yaml_file = args.yaml_file
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"YAML config not found: {yaml_file}")

    config = load_config(yaml_file)
    network_config = config['network_g'].copy()
    network_config.pop('type', None)

    use_adapters = args.use_adapters

    if use_adapters:
        adapter_config = network_config.pop('adapter_config', None)
        model = RestormerAdapters(**network_config, adapter_config=adapter_config)
        checkpoint = torch.load(args.weights, map_location='cpu')
        state_dict = checkpoint.get('params', checkpoint)

        # pre-allocating adapter_list before loading state_dict to avoid key mismatches
        model.prepare_adapter_list_for_loading(state_dict)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = Restormer(**network_config)
        checkpoint = torch.load(args.weights, map_location='cpu')
        state_dict = checkpoint.get('params', checkpoint)
        model.load_state_dict(state_dict)

    print(f"Loaded weights from: {args.weights}")
    return model, use_adapters


def main():
    parser = argparse.ArgumentParser(description='Cross-Degradation Testing using Restormer')
    parser.add_argument('--input_dir', default='./Datasets/', type=str,
                        help='Directory of validation/test images')
    parser.add_argument('--result_dir', default='./results/', type=str,
                        help='Directory for results')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights (.pth checkpoint)')
    parser.add_argument('--yaml_file', type=str, required=True,
                        help='Path to model YAML config file.')
    parser.add_argument('--domain', type=str, required=True,
                        choices=VALID_DOMAINS,
                        help='Degradation domain to evaluate on')
    parser.add_argument('--save_images', action='store_true',
                        help='Save restored images to result_dir')

    parser.add_argument('--use_adapters', action='store_true',
                        help='Use adapter mode. '
                        'When enabled, RestormerAdapters is used instead of plain Restormer.')
    parser.add_argument('--adapter_id', type=int, default=-1,
                        help='Adapter to activate during inference (default: -1).'
                             ' -1 : backbone only, no adapter applied.'
                             ' k : adapter_list[k] for a previously committed domain.'
                             ' Ignored when --use_adapters is not set.')

    args = parser.parse_args()
    model, use_adapters = build_model(args)

    if use_adapters and args.adapter_id == -1:
        print("WARNING: use_adapters is set but adapter_id=-1 (backbone only).")
        print("Pass adapter_id to use the trained adapter.")

    model.cuda()
    model = nn.DataParallel(model)
    model.eval()

    if use_adapters:
        if args.adapter_id == -1:
            print(f"Mode: RestormerAdapters, backbone only (adapter_id={args.adapter_id})")
        else:
            print(f"Mode: RestormerAdapters, with adapter {args.adapter_id}")
    else:
        print(f"Mode: plain Restormer backbone")
    print(f"Evaluating on {args.domain}")

    # D1 uses real CBSD68 denoising dataset; D2-D5 use synthesized domains
    if args.domain == "D1_denoise":
        dataset_name = 'CBSD68_sigma25_Full_Images'
    else:
        dataset_name = f'{args.domain}_Full_Images'

    inp_dir = os.path.join(args.input_dir, 'val', dataset_name, 'input_crops')
    gt_dir = os.path.join(args.input_dir, 'val', dataset_name, 'target_crops')
    result_dir = os.path.join(args.result_dir, 
                              f'{dataset_name}_Adapters' if use_adapters else f'{dataset_name}')

    if args.save_images:
        os.makedirs(result_dir, exist_ok=True)

    has_gt = os.path.exists(gt_dir)

    files = natsorted(glob(os.path.join(inp_dir, '*.png')))
    if len(files) == 0:
        print(f"No images found in {inp_dir}")
        return

    print(f"Testing on {len(files)} images from {dataset_name} (validation split)")

    factor = 8  # padding factor for Restormer window size

    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(load_img(file_)) / 255.
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img_tensor.unsqueeze(0).cuda()

            # pad to multiple of factor
            h, w = input_.shape[2], input_.shape[3]
            H = ((h + factor) // factor) * factor
            W = ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if use_adapters:
                restored = model(input_, adapter_id=args.adapter_id)
            else:
                restored = model(input_)

            restored = restored[:, :, :h, :w]

            restored = torch.clamp(restored, 0, 1).cpu().detach()
            restored = restored.permute(0, 2, 3, 1).squeeze(0).numpy()

            if has_gt:
                filename = os.path.basename(file_)
                gt_path = os.path.join(gt_dir, filename)
                if os.path.exists(gt_path):
                    gt = np.float32(load_img(gt_path)) / 255.
                    psnr_val = psnr(gt, restored, data_range=1.0)
                    ssim_val = ssim(gt, restored, data_range=1.0, channel_axis=2)
                    psnr_values.append(psnr_val)
                    ssim_values.append(ssim_val)

            if args.save_images:
                save_path = os.path.join(
                    result_dir,
                    os.path.splitext(os.path.basename(file_))[0] + '.png'
                )
                save_img(save_path, img_as_ubyte(restored))

    print(f"\nResults for {dataset_name} (validation split)")
    print("=" * 50)

    if psnr_values:
        print(f"PSNR: {np.mean(psnr_values):.4f} dB")
        print(f"SSIM: {np.mean(ssim_values):.4f}")
    else:
        print("No ground truth available for metric computation")

    if args.save_images:
        print(f"\nRestored images saved to: {result_dir}")


if __name__ == '__main__':
    main()
