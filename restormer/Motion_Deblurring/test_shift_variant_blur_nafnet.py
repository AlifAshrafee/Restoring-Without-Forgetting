### Testing Shift-Variant Blur Deblurring using NAFNet with Adapters
###
### Evaluate V2 adapter on V2 data:
###   PYTHONPATH=.. python test_shift_variant_blur_nafnet.py \
###      --weights ../experiments/archived_checkpoints/NAFNet_V1_ft_V2_Adapters/net_g_latest.pth \
###      --yaml_file Options/Shift_Variant_V1_ft_V2_Deblurring_NAFNet_Adapters.yml \
###      --variant 2 --adapter_id 0 --save_images
###
### Evaluate backbone only (V1) using adapter checkpoint:
###   PYTHONPATH=.. python test_shift_variant_blur_nafnet.py \
###      --weights ../experiments/archived_checkpoints/NAFNet_V1_ft_V2_Adapters/net_g_latest.pth \
###      --yaml_file Options/Shift_Variant_V1_ft_V2_Deblurring_NAFNet_Adapters.yml \
###      --variant 1 --adapter_id -1

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from natsort import natsorted
from glob import glob
from basicsr.models.archs.nafnet_adapters_arch import NAFNetAdapters
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2


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
    config = load_config(args.yaml_file)
    network_config = config['network_g'].copy()
    network_config.pop('type', None)
    adapter_config = network_config.pop('adapter_config', None)

    model = NAFNetAdapters(**network_config, adapter_config=adapter_config)

    checkpoint = torch.load(args.weights, map_location='cpu')
    state_dict = checkpoint.get('params', checkpoint)

    model.prepare_adapter_list_for_loading(state_dict)
    model.load_state_dict(state_dict, strict=False)

    print(f"Loaded weights from: {args.weights}")
    print(f"Committed adapters: {len(model.adapter_list)}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Shift-Variant Blur Deblurring using NAFNet with Adapters')
    parser.add_argument('--input_dir', default='./Datasets/', type=str,
                        help='Directory of validation/test images')
    parser.add_argument('--result_dir', default='./results/', type=str,
                        help='Directory for results')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to NAFNetAdapters checkpoint (.pth)')
    parser.add_argument('--yaml_file', type=str, required=True,
                        help='Path to NAFNetAdapters YAML config file')
    parser.add_argument('--variant', type=int, default=1,
                        help='Variant number to evaluate on (1-6)')
    parser.add_argument('--adapter_id', type=int, default=-1,
                        help='Adapter to activate. '
                             '-1: backbone only, '
                             'k: adapter_list[k] for committed domain k')
    parser.add_argument('--save_images', action='store_true',
                        help='Save restored images to result_dir')

    args = parser.parse_args()
    model = build_model(args)

    if args.adapter_id == -1:
        print(f"Mode: backbone only (adapter_id=-1)")
    else:
        print(f"Mode: adapter_id={args.adapter_id}")
    print(f"Evaluating on variant V{args.variant}")

    model.cuda()
    model = nn.DataParallel(model)
    model.eval()

    dataset_name = f'ShiftVariant_V{args.variant}_Full_Images'
    inp_dir = os.path.join(args.input_dir, 'val', dataset_name, 'input_crops')
    gt_dir = os.path.join(args.input_dir, 'val', dataset_name, 'target_crops')
    result_dir = os.path.join(args.result_dir, dataset_name)

    if args.save_images:
        os.makedirs(result_dir, exist_ok=True)

    has_gt = os.path.exists(gt_dir)

    files = natsorted(glob(os.path.join(inp_dir, '*.png')))
    if len(files) == 0:
        print(f"No images found in {inp_dir}")
        return

    print(f"Testing on {len(files)} images from {dataset_name}")

    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(load_img(file_)) / 255.
            input_ = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()

            h, w = input_.shape[2], input_.shape[3]

            restored = model(input_, adapter_id=args.adapter_id)
            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach()
            restored = restored.permute(0, 2, 3, 1).squeeze(0).numpy()

            if has_gt:
                gt_path = os.path.join(gt_dir, os.path.basename(file_))
                if os.path.exists(gt_path):
                    gt = np.float32(load_img(gt_path)) / 255.
                    psnr_values.append(psnr(gt, restored, data_range=1.0))
                    ssim_values.append(ssim(gt, restored, data_range=1.0,
                                            channel_axis=2))

            if args.save_images:
                save_path = os.path.join(
                    result_dir,
                    os.path.splitext(os.path.basename(file_))[0] + '.png')
                save_img(save_path, img_as_ubyte(restored))

    print(f"\nResults for {dataset_name}")
    print("=" * 50)

    if psnr_values:
        print(f"PSNR: {np.mean(psnr_values):.4f} dB")
        print(f"SSIM: {np.mean(ssim_values):.4f}")
    else:
        print("No ground truth available for metric computation")

    if args.save_images:
        print(f"Restored images saved to: {result_dir}")


if __name__ == '__main__':
    main()