### Sim-to-real evaluation for RwF (Restormer + adapters) across the
### five canonical degradation domains (D1 denoise / D2 deblur / D3 derain /
### D4 dehaze / D5 lowlight).
###
### Example:
###   PYTHONPATH=.. python test_real_degradations_with_routing.py \
###       --weights    ../experiments/.../Adapters/net_g_latest.pth \
###       --yaml_file  Options/..._Restormer_Adapters.yml \
###       --prototypes ../experiments/.../prototypes/proto_enc1.pth \
###       --input_dir  ./Datasets/test/

import os
import argparse
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
import cv2
import yaml

from basicsr.models.archs.restormer_adapters_arch import RestormerAdapters
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


DIR_TO_PROTO = {
    "D1_gaussian_denoising": "D1_denoise",
    "D2_motion_deblurring": "D2_deblur",
    "D3_image_deraining": "D3_derain",
    "D4_image_dehazing": "D4_dehaze",
    "D5_lowlight_enhancement": "D5_lowlight",
}

DOMAIN_TO_ADAPTER = {
    "D1_denoise": -1,
    "D2_deblur": 0,
    "D3_derain": 1,
    "D4_dehaze": 2,
    "D5_lowlight": 3,
}

SHORT = {"D1_denoise": "D1", "D2_deblur": "D2", "D3_derain": "D3",
         "D4_dehaze": "D4", "D5_lowlight": "D5"}

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg")


def load_config(yaml_path):
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    with open(yaml_path, "r") as f:
        return yaml.load(f, Loader=Loader)


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def list_images(d):
    files = []
    for e in IMG_EXTS:
        files.extend(glob(os.path.join(d, e)))
    return natsorted(files)


def find_gt(gt_dir, src_path):
    stem = os.path.splitext(os.path.basename(src_path))[0]
    direct = os.path.join(gt_dir, os.path.basename(src_path))
    if os.path.exists(direct):
        return direct
    for ext in (".png", ".jpg", ".jpeg"):
        cand = os.path.join(gt_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    return None


def build_model(args):
    config = load_config(args.yaml_file)
    network_config = config["network_g"].copy()
    network_config.pop("type", None)
    adapter_config = network_config.pop("adapter_config", None)
    model = RestormerAdapters(**network_config, adapter_config=adapter_config)
    checkpoint = torch.load(args.weights, map_location="cpu")
    state_dict = checkpoint.get("params", checkpoint)
    model.prepare_adapter_list_for_loading(state_dict)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {args.weights}")
    print(f"Committed adapters: {len(model.adapter_list)}")
    return model


class FeatureMatcher:
    def __init__(self, model, prototypes_gpu):
        self.model = model
        self.prototypes = prototypes_gpu
        self._features = {}
        self._handle = model.encoder_level1[-1].register_forward_hook(self._hook)

    def _hook(self, m, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        self._features["enc1"] = out

    def embedding(self, img_tensor):
        with torch.no_grad():
            _ = self.model(img_tensor, adapter_id=-1)
        feat = self._features["enc1"]
        emb_mean   = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # [B, C]
        emb_mean_n = F.normalize(emb_mean, p=2, dim=1)
        emb_std    = feat.flatten(2).std(dim=2)  # [B, C]
        emb_std_n  = F.normalize(emb_std,  p=2, dim=1)
        emb = F.normalize(torch.cat([emb_mean_n, emb_std_n], dim=1), p=2, dim=1)
        return emb.squeeze(0)

    def restore(self, img_tensor, adapter_id):
        with torch.no_grad():
            return self.model(img_tensor, adapter_id=adapter_id)

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def match_domain(emb, prototypes_gpu):
    sims = {d: torch.dot(emb, p).item() for d, p in prototypes_gpu.items()}
    return max(sims, key=sims.get), sims


def discover_runs(test_root, domain_filter=None, testset_filter=None):
    runs = []
    if not os.path.isdir(test_root):
        print(f"[ERROR] test root does not exist: {test_root}")
        return runs
    for dd in sorted(os.listdir(test_root)):
        dpath = os.path.join(test_root, dd)
        if not os.path.isdir(dpath):
            continue
        if domain_filter and dd not in domain_filter:
            continue
        if dd not in DIR_TO_PROTO:
            print(f"[WARN] unknown domain dir '{dd}', skipping")
            continue
        for ts in sorted(os.listdir(dpath)):
            tspath = os.path.join(dpath, ts)
            if not os.path.isdir(tspath):
                continue
            if testset_filter and ts not in testset_filter:
                continue
            inp_dir = os.path.join(tspath, "degraded")
            gt_dir  = os.path.join(tspath, "gt")
            if not os.path.isdir(inp_dir):
                print(f"[WARN] no degraded/ under {tspath}, skipping")
                continue
            runs.append((dd, ts, inp_dir, gt_dir))
    return runs


def evaluate(args):
    model = build_model(args).cuda().eval()

    proto_data = torch.load(args.prototypes, map_location="cpu")
    prototypes = proto_data["prototypes"]
    proto_meta = proto_data.get("metadata", {})
    print(f"\nLoaded prototypes: {args.prototypes}")
    for d, m in sorted(proto_meta.items()):
        print(f"  {d}: dim={m.get('embedding_dim', '?')}, "
              f"samples={m.get('num_samples', '?')}, "
              f"adapter_id={m.get('adapter_id', '?')}")

    prototypes_gpu = {d: p.cuda() for d, p in prototypes.items()}
    matcher = FeatureMatcher(model, prototypes_gpu)
    domain_list = sorted(prototypes_gpu.keys())

    runs = discover_runs(
        args.input_dir,
        domain_filter=set(args.domains) if args.domains else None,
        testset_filter=set(args.test_sets) if args.test_sets else None,
    )
    if not runs:
        print("No (domain, test_set) pairs found. Check --input_dir / --domains / --test_sets.")
        return

    factor = 8
    results = {}  # (dd, ts) -> per-image accumulators

    for dd, ts, inp_dir, gt_dir in runs:
        proto_domain  = DIR_TO_PROTO[dd]
        gt_adapter_id = DOMAIN_TO_ADAPTER[proto_domain]
        has_gt = os.path.isdir(gt_dir)
        files = list_images(inp_dir)
        if not files:
            print(f"[WARN] no images in {inp_dir}, skipping")
            continue
        print(f"\n[{dd} / {ts}] {len(files)} images "
              f"(GT adapter={gt_adapter_id}, has_gt={has_gt})")

        r = {
            "psnr_pred":   [], "ssim_pred":   [],
            "psnr_oracle": [], "ssim_oracle": [],
            "correct": 0, "total": 0, "predictions": [],
        }

        for f in tqdm(files, desc=f"{dd}/{ts}", unit="img"):
            torch.cuda.empty_cache()

            img_np = np.float32(load_img(f)) / 255.0
            x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).cuda()
            _, _, h, w = x.shape
            pad_h = (factor - h % factor) % factor
            pad_w = (factor - w % factor) % factor
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), "reflect") if (pad_h or pad_w) else x

            # --- Pass 1: domain identification (backbone-only forward via hook) ---
            emb = matcher.embedding(x_pad)
            pred_domain, _ = match_domain(emb, prototypes_gpu)
            pred_adapter_id = DOMAIN_TO_ADAPTER[pred_domain]
            r["predictions"].append(pred_domain)
            r["total"]   += 1
            r["correct"] += int(pred_domain == proto_domain)

            # --- Pass 2: predicted-adapter restoration ---
            y_pred = matcher.restore(x_pad, adapter_id=pred_adapter_id)
            y_pred = torch.clamp(y_pred[:, :, :h, :w], 0, 1).cpu()

            # --- Pass 3: oracle-adapter restoration (transfer upper bound) ---
            y_oracle = matcher.restore(x_pad, adapter_id=gt_adapter_id)
            y_oracle = torch.clamp(y_oracle[:, :, :h, :w], 0, 1).cpu()

            if has_gt:
                gt_path = find_gt(gt_dir, f)
                if gt_path is not None:
                    gt_img = np.float32(load_img(gt_path)) / 255.0
                    p_img = y_pred.squeeze(0).permute(1, 2, 0).numpy()
                    o_img = y_oracle.squeeze(0).permute(1, 2, 0).numpy()
                    # Some real-world pairs (e.g. RealBlur) come with sub-pixel misalignments;
                    # we treat them as-is here. Sizes are expected to match for canonical sets.
                    if gt_img.shape == p_img.shape:
                        r["psnr_pred"].append(psnr(gt_img, p_img, data_range=1.0))
                        r["ssim_pred"].append(ssim(gt_img, p_img, data_range=1.0, channel_axis=2))
                        r["psnr_oracle"].append(psnr(gt_img, o_img, data_range=1.0))
                        r["ssim_oracle"].append(ssim(gt_img, o_img, data_range=1.0, channel_axis=2))

            if args.save_images:
                out_dir = os.path.join(args.result_dir, dd, ts)
                os.makedirs(out_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(f))[0]
                save_img(os.path.join(out_dir, stem + "_pred.png"),
                         img_as_ubyte(y_pred.squeeze(0).permute(1, 2, 0).numpy()))
                if args.save_oracle:
                    save_img(os.path.join(out_dir, stem + "_oracle.png"),
                             img_as_ubyte(y_oracle.squeeze(0).permute(1, 2, 0).numpy()))

        results[(dd, ts)] = r

    matcher.close()

    print("\n" + "=" * 100)
    print("SIM-TO-REAL CROSS-DEGRADATION EVALUATION")
    print("=" * 100)
    header = (f"{'Domain':<26} {'Test set':<18} {'N':>5}  {'Acc':>6}  "
              f"{'PSNR(pred)':>10}  {'PSNR(oracle)':>12}  "
              f"{'SSIM(pred)':>10}  {'SSIM(oracle)':>12}")
    print(header)
    print("-" * len(header))

    by_domain = defaultdict(list)
    for (dd, ts), r in results.items():
        by_domain[dd].append((ts, r))

    all_pp, all_po, all_sp, all_so = [], [], [], []
    total_correct, total_imgs = 0, 0

    for dd in sorted(by_domain.keys()):
        for ts, r in by_domain[dd]:
            acc = 100 * r["correct"] / r["total"] if r["total"] else 0.0
            pp = np.mean(r["psnr_pred"])   if r["psnr_pred"]   else float("nan")
            po = np.mean(r["psnr_oracle"]) if r["psnr_oracle"] else float("nan")
            sp = np.mean(r["ssim_pred"])   if r["ssim_pred"]   else float("nan")
            so = np.mean(r["ssim_oracle"]) if r["ssim_oracle"] else float("nan")
            print(f"{dd:<26} {ts:<18} {r['total']:>5}  {acc:>5.1f}%  "
                  f"{pp:>8.2f}dB  {po:>10.2f}dB  "
                  f"{sp:>10.4f}  {so:>12.4f}")
            all_pp.extend(r["psnr_pred"]);   all_po.extend(r["psnr_oracle"])
            all_sp.extend(r["ssim_pred"]);   all_so.extend(r["ssim_oracle"])
            total_correct += r["correct"];   total_imgs   += r["total"]

    print("-" * len(header))
    overall_acc = 100 * total_correct / total_imgs if total_imgs else 0.0
    if all_pp:
        print(f"{'Overall':<26} {'':<18} {total_imgs:>5}  {overall_acc:>5.1f}%  "
              f"{np.mean(all_pp):>8.2f}dB  {np.mean(all_po):>10.2f}dB  "
              f"{np.mean(all_sp):>10.4f}  {np.mean(all_so):>12.4f}")
        print(f"\nMean PSNR gap (oracle - predicted): "
              f"{np.mean(all_po) - np.mean(all_pp):+.2f} dB")
    else:
        print(f"{'Overall':<26} {'':<18} {total_imgs:>5}  {overall_acc:>5.1f}%   "
              f"(no GT available for any test set)")

    # confusion matrix over prototype-bank domains
    n = len(domain_list)
    idx = {d: i for i, d in enumerate(domain_list)}
    C = np.zeros((n, n), dtype=int)
    for (dd, ts), r in results.items():
        gt_d = DIR_TO_PROTO[dd]
        for pred_d in r["predictions"]:
            if pred_d in idx:
                C[idx[gt_d], idx[pred_d]] += 1

    print("\nConfusion Matrix (rows = ground-truth domain, cols = predicted):")
    print("        " + "".join(f"  {SHORT.get(d, d):<5}" for d in domain_list))
    for i, gt_d in enumerate(domain_list):
        row = f"  {SHORT.get(gt_d, gt_d):<5} "
        for j in range(n):
            v = C[i, j]
            row += f" [{v:>4}]" if i == j else f"  {v:>4} "
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Sim-to-real cross-degradation evaluation with prototype routing")
    parser.add_argument("--input_dir", type=str, default="./Datasets/test/",
                        help="Root containing <Dk_domain>/<test_set>/{degraded, gt}/")
    parser.add_argument("--result_dir", type=str, default="./results_sim2real/")
    parser.add_argument("--weights", type=str,
                        default="../experiments/archived_checkpoints/Degradations_D1_D2_D3_D4_ft_D5_Adapters/net_g_latest.pth")
    parser.add_argument("--yaml_file", type=str,
                        default="Options/Degradations_D1_D2_D3_D4_ft_D5_Restormer_Adapters.yml")
    parser.add_argument("--prototypes", type=str,
                        default="../experiments/archived_checkpoints/prototypes/proto_enc1_v2.pth",
                        help="Path to prototypes .pth (encoder_level1, GAP+std L2-norm separate)")
    parser.add_argument("--domains", nargs="+", default=None,
                        help="Subset of domain directory names "
                             "(e.g. D1_gaussian_denoising D5_lowlight_enhancement)")
    parser.add_argument("--test_sets",  nargs="+", default=None,
                        help="Subset of test-set names (e.g. CBSD68_sigma25 LOL_v1)")
    parser.add_argument("--save_images", action="store_true",
                        help="Save predicted-adapter restorations to result_dir/<domain>/<test_set>/")
    parser.add_argument("--save_oracle", action="store_true",
                        help="Also save oracle-adapter restorations (requires --save_images)")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
