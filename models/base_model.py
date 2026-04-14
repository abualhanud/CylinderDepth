# Copyright (c) 2023 42dot. All rights reserved.
import os
import torch

_OPTIMIZER_NAME ='adam'


class BaseModel:
    def __init__(self, cfg):
        self._dataloaders = {}
        self.mode = None
        self.models = None
        self.optimizer = None
        self.lr_scheduler = None
        self.ddp_enable = False

    def read_config(self, cfg):
        raise NotImplementedError('Not implemented for BaseModel')

    def prepare_dataset(self):
        raise NotImplementedError('Not implemented for BaseModel')

    def set_optimizer(self):
        raise NotImplementedError('Not implemented for BaseModel')        
  
    def train_dataloader(self):
        return self._dataloaders['train']

    def val_dataloader(self):
        return self._dataloaders['val']

    def eval_dataloader(self):
        return self._dataloaders['eval']
    
    def set_train(self):
        self.mode = 'train'
        for m in self.models.values():
            m.train()

    def set_val(self):
        self.mode = 'val'
        for m in self.models.values():
            m.eval()

    def save_model(self, epoch):
        curr_model_weights_dir = os.path.join(self.save_weights_root, f'weights_{epoch}')
        os.makedirs(curr_model_weights_dir, exist_ok=True)

        for model_name, model in self.models.items():
            model_file_path = os.path.join(curr_model_weights_dir, f'{model_name}.pth')
            to_save = model.state_dict()
            torch.save(to_save, model_file_path)
        
        # save optimizer
        optim_file_path = os.path.join(curr_model_weights_dir, f'{_OPTIMIZER_NAME}.pth')
        torch.save(self.optimizer.state_dict(), optim_file_path)

    def load_weights(self):
        assert os.path.isdir(self.load_weights_dir), f'\tCannot find {self.load_weights_dir}'
        print(f'Loading a model from {self.load_weights_dir}')

        map_location = None
        if self.pretrain and self.ddp_enable:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % (self.world_size - 1)}

        for n in self.models_to_load:
            print(f'Loading {n} weights...')
            path = os.path.join(self.load_weights_dir, f'{n}.pth')

            # load checkpoint dict (supports plain dict or {"state_dict": ...})
            ckpt = torch.load(path, map_location=map_location) if map_location else torch.load(path)
            pre_trained_dict = ckpt.get("state_dict", ckpt)

            # harmonize DDP/DataParallel "module." prefix between ckpt and model
            model_dict = self.models[n].state_dict()
            model_keys = model_dict.keys()
            ckpt_keys = pre_trained_dict.keys()
            model_has_module = any(k.startswith("module.") for k in model_keys)
            ckpt_has_module = any(k.startswith("module.") for k in ckpt_keys)

            if ckpt_has_module and not model_has_module:
                pre_trained_dict = {k[len("module."):]: v for k, v in pre_trained_dict.items()}
            elif model_has_module and not ckpt_has_module:
                pre_trained_dict = {f"module.{k}": v for k, v in pre_trained_dict.items()}

            # keep only keys present in model with matching tensor shapes
            pre_trained_dict = {
                k: v for k, v in pre_trained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            # diagnostics AFTER harmonization/filtering
            consumed = set(pre_trained_dict.keys())
            leftover_in_model = sorted(set(model_dict.keys()) - consumed)
            missing_in_model = sorted(set(consumed) - set(model_dict.keys()))
            print(f"[{n}] provided={len(ckpt_keys)} | will_load={len(consumed)} | "
                f"not_in_ckpt={len(leftover_in_model)} | not_in_model={len(missing_in_model)}")
            if leftover_in_model:
                print(f"[{n}] not_in_ckpt examples: {leftover_in_model[:20]}")
            if missing_in_model:
                print(f"[{n}] not_in_model examples: {missing_in_model[:20]}")

            # load (strict=False tolerates benign leftovers)
            self.models[n].load_state_dict({**model_dict, **pre_trained_dict}, strict=False)

        if self.mode == 'train':
            optim_file_path = os.path.join(self.load_weights_dir, f'{_OPTIMIZER_NAME}.pth')
            if os.path.isfile(optim_file_path):
                try:
                    print(f'Loading {_OPTIMIZER_NAME} weights')
                    optimizer_dict = torch.load(
                        optim_file_path, map_location=map_location
                    ) if map_location else torch.load(optim_file_path)
                    self.optimizer.load_state_dict(optimizer_dict)
                except ValueError:
                    print(f'\tCannnot load {_OPTIMIZER_NAME} - the optimizer will be randomly initialized')
            else:
                print(f'\tCannot find {_OPTIMIZER_NAME} weights, so the optimizer will be randomly initialized')