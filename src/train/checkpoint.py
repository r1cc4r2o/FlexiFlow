from abc import ABC
from pathlib import Path
from time import time
import importlib
import os


from natsort import natsorted
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


class Criterion(ABC):
    def reset(self):
        pass

    def save(self, *args, **kwargs):
        pass


class TimeCriterion(Criterion):
    def __init__(self, save_every=1800):
        self.time = 0
        self.save_every = save_every
    def reset(self):
        self.time = time()
    def save(self, *args, **kwargs):
        return (time() - self.time) > self.save_every

def unwrap_module(m):
    """Return the underlying nn.Module, whether wrapped in DDP or not."""
    return m.module if isinstance(m, DDP) else m


class CheckpointManager:
    def __init__(self, vocabulary, model, optimizer, save_path, seed, batch=0, epoch=0, swa=None, scheduler=None, protein_vocabulary=None, device="cpu"):
        self.model = model
        self.swa = swa
        if not hasattr(unwrap_module(model), "get_state_dict") or not hasattr(unwrap_module(model), "load_from_state_dict"):
            raise ValueError("model needs to have implemented `get_state_dict` and `load_from_state_dict`")
        self.vocabulary = vocabulary
        self.optimizer = optimizer
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.seed = seed
        self.epoch = epoch
        self.batch = batch
        self.scheduler = scheduler
        self.protein_vocabulary = protein_vocabulary


    def get_state_dict(self):
        v = self.vocabulary.get_state_dict()
        v["class_name"] = self.vocabulary.__class__.__name__
        v["class_module"] = self.vocabulary.__class__.__module__
        
        if self.protein_vocabulary is not None:
            pv = self.protein_vocabulary.get_state_dict()
            pv["class_name"] = self.protein_vocabulary.__class__.__name__
            pv["class_module"] = self.protein_vocabulary.__class__.__module__
        else:
            pv = None

        params = {
            "seed": self.seed,
            "epoch": self.epoch,
            "batch": self.batch,
            "device": self.device,
            "vocabulary": v,
            "protein_vocabulary": pv,
        }
        return params
    
    
    def save(self, model, optimizer, scheduler, epoch, batch, swa=None):
        model_state = unwrap_module(model).get_state_dict()
        opt_state = optimizer.state_dict()
        opt_state["class_name"] = optimizer.__class__.__name__
        opt_state["class_module"] = optimizer.__class__.__module__
        # if scheduler is not None:
        scheduler_state = scheduler.state_dict()
        scheduler_state["class_name"] = scheduler.__class__.__name__
        scheduler_state["class_module"] = scheduler.__class__.__module__
        # else:
        #     scheduler_state = None

        if swa is not None:
            swa_state = swa.state_dict()
            swa_state["class_name"] = swa.__class__.__name__
            swa_state["class_module"] = swa.__class__.__module__
        else:
            swa_state = None
            
        t = int(time())        
        torch.save(model_state, self.save_path / f"model_{t:d}_{epoch:d}_{batch:d}.pt")
        torch.save(opt_state, self.save_path / f"opt_{t:d}_{epoch:d}_{batch:d}.pt")
        torch.save(scheduler_state, self.save_path / f"scheduler_{t:d}_{epoch:d}_{batch:d}.pt")
        torch.save(swa_state, self.save_path / f"swa_{t:d}_{epoch:d}_{batch:d}.pt")
    
        self.epoch = epoch
        self.batch = batch
        checkpoint_state = self.get_state_dict()
        
        torch.save(checkpoint_state, self.save_path / f"chpt_{t:d}_{epoch:d}_{batch:d}.pt")

    def save_tmp(self, model, optimizer, scheduler, epoch, batch, swa=None):
        model_state = unwrap_module(model).get_state_dict()
        opt_state = optimizer.state_dict()
        opt_state["class_name"] = optimizer.__class__.__name__
        opt_state["class_module"] = optimizer.__class__.__module__
        # if self.scheduler is not None:
        scheduler_state = scheduler.state_dict()
        scheduler_state["class_name"] = scheduler.__class__.__name__
        scheduler_state["class_module"] = scheduler.__class__.__module__
        # else:
        #     scheduler_state = None

        if swa is not None:
            swa_state = swa.state_dict()
            swa_state["class_name"] = swa.__class__.__name__
            swa_state["class_module"] = swa.__class__.__module__
        else:
            swa_state = None
            
        t = int(time())        
        torch.save(model_state, self.save_path / f"model_{epoch:d}_tmp.pt")
        torch.save(opt_state, self.save_path / f"opt_{epoch:d}_tmp.pt")
        torch.save(scheduler_state, self.save_path / f"scheduler_{epoch:d}_tmp.pt")
        torch.save(swa_state, self.save_path / f"swa_{epoch:d}_tmp.pt")
    
        self.epoch = epoch
        self.batch = batch
        checkpoint_state = self.get_state_dict()
        
        torch.save(checkpoint_state, self.save_path / f"chpt_{epoch:d}_tmp.pt")

    @staticmethod
    def load_object(state):
        class_name = state["class_name"]
        class_module = state["class_module"]
        try:
            if class_module == "__main__":
                model_class = globals()[class_name]
            else:
                model_class = getattr(
                    importlib.import_module(class_module), class_name
                )
        except BaseException:
            raise ValueError(f"Class `{class_name}` not found")
        del state["class_name"]
        del state["class_module"]
        return model_class, state
        
    
    @staticmethod
    def load_model(state, device):
        model_class, state = CheckpointManager.load_object(state)
        model = model_class.load_from_state_dict(state, device)
        return model


    @staticmethod
    def load_swa_model(state, model):
        if state is None:
            return None
        swa_model_class, state = CheckpointManager.load_object(state)
        swa_model = swa_model_class(unwrap_module(model))
        swa_model.load_state_dict(state)
        return swa_model
        
    
    @staticmethod
    def load_optimizer(state, model):
        optimizer_class, state = CheckpointManager.load_object(state)
        opt = optimizer_class(model.parameters())
        # FIXME: keys are missing now because of the conditional part
        try:
            opt.load_state_dict(state)
        except ValueError:
            pass
        return opt

    
    @staticmethod
    def load_scheduler(state, optimizer):
        if state is None:
            return None
        scheduler_class, state = CheckpointManager.load_object(state)
        scheduler = scheduler_class(optimizer)
        # FIXME: keys are missing now because of the conditional part
        try:
            scheduler.load_state_dict(state)
        except ValueError:
            pass
        return scheduler


    @staticmethod
    def load_vocabulary(state):
        if state is None:
            return None
        vocabulary_class, state = CheckpointManager.load_object(state)
        try:
            vocabulary = vocabulary_class.load_from_state_dict(state)
        except ValueError:
            pass
        return vocabulary


    @classmethod
    def restore_last_checkpoint(cls, save_path, world_size=1, device=None, verbose=False):
        save_path = Path(save_path)
        checkpoints = natsorted([ f for f in save_path.glob("chpt_*.pt")])[::-1]
        checkpoint_loaded = False
        for checkpoint in checkpoints:
            try:
                chpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
                
                if device is None:
                    device = chpt["device"]

                vocabulary = cls.load_vocabulary(chpt["vocabulary"])
                del chpt["vocabulary"]
                chpt["protein_vocabulary"] = cls.load_vocabulary(chpt["protein_vocabulary"])
  
                model_chpt = os.path.basename(str(checkpoint)).replace("chpt_", "model_")
                opt_chpt = os.path.basename(str(checkpoint)).replace("chpt_", "opt_")
                scheduler_chpt = os.path.basename(str(checkpoint)).replace("chpt_", "scheduler_")
                swa_chpt = os.path.basename(str(checkpoint)).replace("chpt_", "swa_")
                
                model_state = torch.load(save_path / model_chpt, map_location="cpu", weights_only=False)
                opt_state = torch.load(save_path / opt_chpt, map_location="cpu", weights_only=False)
                scheduler_state = torch.load(save_path / scheduler_chpt, map_location="cpu", weights_only=False)
                swa_state = torch.load(save_path / swa_chpt, map_location="cpu", weights_only=False)

                model = cls.load_model(model_state, device)
                if world_size > 1:
                    model = DDP(model, device_ids=[device])
                opt = cls.load_optimizer(opt_state, model)
                scheduler = cls.load_scheduler(scheduler_state, opt)
                swa = cls.load_swa_model(swa_state, model)

                checkpoint_manager = cls(vocabulary, model, opt, save_path, swa=swa, scheduler=scheduler, **chpt)
             
                checkpoint_loaded = True
                break
            except BaseException:
                pass
        if not checkpoint_loaded:
            return None, None, None, None, None
        if verbose:
            print(f"Loaded checkpoint: {checkpoint}")
        return model, opt, scheduler, swa, checkpoint_manager