import importlib
import torch
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
from basicsr.models.image_restoration_model import ImageCleanModel
from basicsr.utils import get_root_logger

loss_module = importlib.import_module('basicsr.models.losses')


class AdapterModel(ImageCleanModel):
    def __init__(self, opt):
        self.adapter_training = opt.get("adapter_training", True)
        super().__init__(opt)
        logger = get_root_logger()
        logger.info(f"AdapterModel initialized")

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]
        logger = get_root_logger()

        if self.adapter_training:
            bare_model = self.get_bare_model(self.net_g)  # accessing underlying model
            if self.opt.get('commit_loaded_adapter', False):  # commits loaded adapter to adapter_list
                bare_model.commit_and_reinit()  # reinitializes cur_adapter fresh for new domain
                logger.info(f"Committed {len(bare_model.adapter_list)} loaded adapter(s) to adapter_list")
                logger.info(f"Fresh cur_adapter initialized for new domain {self.opt.get('current_domain', -1)}")

            bare_model.freeze_backbone()  # freezing backbone + all adapter_list entries

            total = sum(p.numel() for p in self.net_g.parameters())
            trainable = sum(p.numel() for p in self.net_g.parameters() if p.requires_grad)
            frozen = total - trainable

            logger.info(f"Parameter statistics:")
            logger.info(f"Total Params: {total}")
            logger.info(f"Frozen Params: {frozen} (backbone)")
            logger.info(f"Trainable Params: {trainable} ({100*trainable/total:.2f}%)")

        self.cri_pix = None

        if train_opt.get("pixel_opt"):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            raise ValueError('Pixel loss not specified in options.')

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        logger = get_root_logger()

        optim_params = []
        for _, param in self.net_g.named_parameters():  # collects only trainable adapter parameters
            if param.requires_grad:
                optim_params.append(param)

        logger.info(f"Optimizer will update {len(optim_params)} parameter groups")

        optim_type = train_opt["optim_g"].pop("type")

        if optim_type == "Adam":
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == "AdamW":
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(f'Optimizer {optim_type} is not supported')

        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        adapter_id = self.opt.get('current_domain', -1)
        self.output = self.net_g(self.lq, adapter_id=adapter_id)

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:  # pixel loss (L1)
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix

        l_total.backward()

        if self.opt["train"].get("use_grad_clip", False):
            torch.nn.utils.clip_grad_norm_(self.get_bare_model(self.net_g).cur_adapter.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def model_to_device(self, net):
        net = net.to(self.device)
        if self.opt.get('dist', False):
            net = DistributedDataParallel(net, device_ids=[torch.cuda.current_device()],
                                          find_unused_parameters=True)
        return net

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        adapter_id = self.opt.get("current_domain", -1)
        self.net_g.eval()
        with torch.no_grad():
            pred = self.net_g(img, adapter_id=adapter_id)
        if isinstance(pred, list):
            pred = pred[-1]
        self.output = pred
        self.net_g.train()

    def load_network(self, net, load_path, strict=True, param_key="params"):
        """
        Loads network weights with adapter-aware handling.

        When loading previous checkpoint into AdapterModel:
        - backbone weights: loaded from checkpoint
        - adapter weights: randomly initialized by the model, we use strict=False to allow this.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)

        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)

        if param_key is not None:
            if param_key not in load_net and "params" in load_net:
                param_key = "params"
            load_net = load_net.get(param_key, load_net)

        for k in list(load_net.keys()):
            if k.startswith("module."):  # removes 'module.' prefix from DataParallel
                load_net[k[7:]] = load_net.pop(k)

        model_keys = set(net.state_dict().keys())
        checkpoint_keys = set(load_net.keys())

        missing_in_ckpt = model_keys - checkpoint_keys  # adapter keys (missing in checkpoint)
        adapter_keys = [k for k in missing_in_ckpt if "adapter" in k]
        other_missing = [k for k in missing_in_ckpt if "adapter" not in k]

        unexpected = checkpoint_keys - model_keys  # keys in checkpoint but not model (shouldn't happen)

        logger.info(f"Loading network from {load_path}")
        logger.info(f"Checkpoint keys: {len(checkpoint_keys)}")
        logger.info(f"Model keys: {len(model_keys)}")
        logger.info(f"Adapter keys (initialized fresh): {len(adapter_keys)}")

        if other_missing:
            logger.warning(f"Non-adapter missing keys: {other_missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {list(unexpected)[:5]}...")

        net.prepare_adapter_list_for_loading(load_net)  # prepares adapter_list to receive loaded adapter if present
        net.load_state_dict(load_net, strict=False)  # strict=False allows missing adapter keys

        logger.info(f"Successfully loaded backbone weights")

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, "net_g", current_iter)  # saves backbone + adapters
        self.save_training_state(epoch, current_iter)
