# ------------------------------------------------------------------------
# Restormer Adapters - Degradation Prototype Extraction
# Usage (earlier-hook bottleneck):
#   PYTHONPATH=.. python extract_degradation_prototypes.py \
#       --weights ../experiments/archived_checkpoints/Degradations_D1_D2_D3_D4_ft_D5_Adapters/net_g_latest.pth \
#       --yaml_file Options/Degradations_D1_D2_D3_D4_ft_D5_Restormer_Adapters.yml \
#       --data_root Datasets/train --batch_size 16 \
#       --output ../experiments/archived_checkpoints/prototypes/proto_enc1.pth \
#       --hook_layer encoder_level1[-1] --max_samples 13800 (50% of patches per domain)
# ------------------------------------------------------------------------

import os
import argparse
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import cv2
import yaml


DOMAINS = ["D1_denoise", "D2_deblur", "D3_derain", "D4_dehaze", "D5_lowlight"]

DOMAIN_TO_DIR = {d: d for d in DOMAINS}

DOMAIN_TO_ADAPTER = {
    "D1_denoise": -1,
    "D2_deblur": 0,
    "D3_derain": 1,
    "D4_dehaze": 2,
    "D5_lowlight": 3,
}


def load_config(yaml_path):
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    with open(yaml_path, "r") as f:
        return yaml.load(f, Loader=Loader)


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def collect_files(data_root, domain, max_samples):
    data_dir = os.path.join(data_root, DOMAIN_TO_DIR[domain], "input_crops")
    if not os.path.exists(data_dir):
        return [], data_dir
    files = natsorted(glob(os.path.join(data_dir, "*.png")))
    if max_samples is not None and len(files) > max_samples:
        rng = np.random.RandomState(seed=42)
        idx = rng.choice(len(files), max_samples, replace=False)
        idx.sort()
        files = [files[i] for i in idx]
    return files, data_dir


def get_hook_target(model, hook_layer):
    """Resolve 'name' or 'name[idx]' (e.g., 'encoder_level2[-1]', 'latent[-1]')."""
    if "[" in hook_layer:
        name, rest = hook_layer.split("[", 1)
        idx = int(rest.rstrip("]"))
        return getattr(model, name)[idx]
    return getattr(model, hook_layer)


def build_model(yaml_file, weights_path):
    from basicsr.models.archs.restormer_adapters_arch import RestormerAdapters
    config = load_config(yaml_file)
    network_config = config["network_g"].copy()
    network_config.pop("type", None)
    adapter_config = network_config.pop("adapter_config", None)
    model = RestormerAdapters(**network_config, adapter_config=adapter_config)
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("params", checkpoint)
    model.prepare_adapter_list_for_loading(state_dict)
    model.load_state_dict(state_dict, strict=False)

    num_committed = len(model.adapter_list)
    print(f"Loaded checkpoint from: {weights_path}")
    print(f"Committed adapters: {num_committed}")
    print(f"Total adapter sets: {num_committed + 1}")

    return model


class CropsDataset(Dataset):
    def __init__(self, files, factor=8):
        self.files = files
        self.factor = factor

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.files[idx]), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(np.float32(img) / 255.0).permute(2, 0, 1)
        _, h, w = t.shape
        pad_h = (self.factor - h % self.factor) % self.factor
        pad_w = (self.factor - w % self.factor) % self.factor
        if pad_h or pad_w:
            t = F.pad(t.unsqueeze(0), (0, pad_w, 0, pad_h), "reflect").squeeze(0)
        return t


class HookedExtractor:
    def __init__(self, model, hook_layer):
        self.model = model
        self._features = {}
        self._handle = get_hook_target(model, hook_layer).register_forward_hook(self._hook)

    def _hook(self, m, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        self._features["bn"] = out

    def extract_batch(self, x):
        with torch.no_grad():
            _ = self.model(x, adapter_id=-1)
        bn = self._features["bn"]                              # [B,C,h',w']

        ## Prototype method 1: GAP only (dim=C)
        # emb = F.adaptive_avg_pool2d(bn, 1).flatten(1)           # [B,C]
        # emb = F.normalize(emb, p=2, dim=1)

        ## Prototype method 2: GAP + std (dim=2C)
        # emb_mean = F.adaptive_avg_pool2d(bn, 1).flatten(1)     # [B,C]
        # emb_std  = bn.flatten(2).std(dim=2)                    # [B,C]
        # emb = F.normalize(torch.cat([emb_mean, emb_std], dim=1), p=2, dim=1)

        ## Prototype method 3: GAP + std with separate L2 norms (dim=2C)
        emb_mean = F.adaptive_avg_pool2d(bn, 1).flatten(1)     # [B,C]
        emb_mean_n = F.normalize(emb_mean, p=2, dim=1)
        emb_std  = bn.flatten(2).std(dim=2)                    # [B,C]
        emb_std_n  = F.normalize(emb_std,  p=2, dim=1)
        emb = F.normalize(torch.cat([emb_mean_n, emb_std_n], dim=1), p=2, dim=1)
        return emb

    def close(self):
        self._handle.remove()


def worker_bottleneck(rank, weights, yaml_file, data_root, domains, hook_layer,
                      max_samples, batch_size, num_data_workers, factor, ret_q):
    try:
        domain = domains[rank]
        gpu_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(gpu_id)

        files, data_dir = collect_files(data_root, domain, max_samples)
        if not files:
            ret_q.put(("bn", domain, None, 0, f"no files in {data_dir}"))
            return

        model = build_model(yaml_file, weights).cuda()
        model.eval()
        extractor = HookedExtractor(model, hook_layer)

        loader = DataLoader(CropsDataset(files, factor=factor),
                            batch_size=batch_size, num_workers=num_data_workers,
                            pin_memory=True, shuffle=False, drop_last=False)

        emb_sum, count = None, 0
        pbar = tqdm(loader, desc=f"[GPU{gpu_id}] {domain} ({hook_layer})",
                    position=rank, leave=True)
        for batch in pbar:
            batch = batch.cuda(non_blocking=True)
            emb = extractor.extract_batch(batch)
            s = emb.sum(0).cpu()
            emb_sum = s if emb_sum is None else emb_sum + s
            count += emb.shape[0]

        prototype = F.normalize(emb_sum / count, p=2, dim=0)
        extractor.close()
        ret_q.put(("bn", domain, prototype, count, None))
    except Exception as e:
        import traceback
        ret_q.put(("bn", domains[rank], None, 0,
                   f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def main():
    parser = argparse.ArgumentParser(
        description="Parallelized prototype extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--hook_layer", type=str, default="encoder_level1[-1]",
                        help="Which bottleneck layer to hook. Examples: "
                             "'encoder_level1[-1]', 'encoder_level2[-1]', 'encoder_level3[-1]', "
                             "'latent[-1]', 'down1_2', 'down2_3', 'down3_4', 'up4_3'.")
    parser.add_argument("--weights", type=str,
                        default="../experiments/archived_checkpoints/Degradations_D1_D2_D3_D4_ft_D5_Adapters/net_g_latest.pth",
                        help="RestormerAdapters checkpoint")
    parser.add_argument("--yaml_file", type=str,
                        default="Options/Degradations_D1_D2_D3_D4_ft_D5_Restormer_Adapters.yml",
                        help="YAML config")
    parser.add_argument("--data_root", type=str, default="Datasets/train",
                        help="Root containing per-domain training subdirectories")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .pth path for prototypes")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per domain (None = use all)")
    parser.add_argument("--domains", type=str, nargs="*", default=None, choices=DOMAINS,
                        help="Subset of domains (default: all five)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_data_workers", type=int, default=2,
                        help="DataLoader workers per process")
    parser.add_argument("--factor", type=int, default=8,
                        help="Spatial padding factor")
    args = parser.parse_args()

    if not (args.weights and args.yaml_file):
        parser.error("--weights and --yaml_file are required.")
    if torch.cuda.device_count() == 0:
        raise RuntimeError("No CUDA GPU detected; parallelization requires a GPU.")

    domains = args.domains or DOMAINS
    print(f"Hook layer: {args.hook_layer}")
    print(f"GPUs available: {torch.cuda.device_count()}")
    print(f"Spawning {len(domains)} workers for: {domains}")

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []

    for rank in range(len(domains)):
        p = ctx.Process(target=worker_bottleneck,
                        args=(rank, args.weights, args.yaml_file, args.data_root,
                                domains, args.hook_layer, args.max_samples,
                                args.batch_size, args.num_data_workers, args.factor, q))
        p.start(); procs.append(p)

    prototypes, metadata = {}, {}
    for _ in range(len(domains)):
        _, domain, proto, count, err = q.get()
        if err:
            print(f"\n[ERROR] {domain}:\n{err}"); continue
        prototypes[domain] = proto
        metadata[domain] = {"domain": domain, "adapter_id": -1,
                            "num_samples": count, "embedding_dim": int(proto.shape[0])}
        print(f"[DONE] {domain}: dim={proto.shape[0]}, samples={count}")
    for p in procs: p.join()

    save_dict = {
        "prototypes": prototypes,
        "metadata": metadata,
        "config": {
            "hook_layer": args.hook_layer,
            "yaml_file": args.yaml_file,
            "weights": args.weights,
            "max_samples": args.max_samples,
            "normalization": "L2",
            "adapter_id": -1,
        },
    }

    if not prototypes:
        raise RuntimeError("No prototypes extracted.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(save_dict, args.output)
    print(f"\nPrototypes saved to: {args.output}")
    print(f"Domains: {sorted(prototypes.keys())}, embedding dim: "
          f"{next(iter(prototypes.values())).shape[0]}")


if __name__ == "__main__":
    main()
