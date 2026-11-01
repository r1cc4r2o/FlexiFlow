
import sys
import os
import yaml
import pickle
import socket

import random
import argparse
import builtins
import importlib

import numpy as np

from tqdm import tqdm
from typing import List
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.molecule import MoleculeBatch

import time
import json
import psutil
import subprocess

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def del_list_inplace(l, id_to_del):
    for i in sorted(id_to_del, reverse=True):
        del(l[i])

# global skipping semaphore
def should_skip_update(local_condition, rank):
    # Convert to tensor (1 if skip, 0 if not)
    skip_tensor = torch.tensor([int(local_condition)], device=rank)
    dist.all_reduce(skip_tensor, op=dist.ReduceOp.SUM)
    return skip_tensor.item() > 0

        
def compute_grad_norm(model):
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        return 0.0
    total_norm = sum([torch.sum(p.grad.detach() ** 2).item() for p in parameters]) ** 0.5
    return total_norm


def get_free_port(start_port=10000, max_port=65535):
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free ports available in range")


def get_distributed_port():
    """Get a port that works with SLURM distributed setup"""
    rank = int(os.environ.get("SLURM_PROCID", 0))

    if rank == 0:
        port = get_free_port()
        
        port_file = f"/tmp/distributed_port_{os.environ.get('SLURM_JOB_ID', 'local')}.txt"
        with open(port_file, 'w') as f:
            f.write(str(port))
        return port
    else:
        port_file = f"/tmp/distributed_port_{os.environ.get('SLURM_JOB_ID', 'local')}.txt"
        max_wait = 30  # Wait up to 30 seconds
        for _ in range(max_wait * 10):
            try:
                if os.path.exists(port_file):
                    with open(port_file, 'r') as f:
                        port = int(f.read().strip())
                    return port
            except (FileNotFoundError, ValueError):
                pass
            time.sleep(0.1)
        
        raise RuntimeError(f"Worker process {rank} couldn't read port from master")


def ddp_setup(rank: int, world_size: int, port: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{port:d}"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def prepare_input(batch, t, vocab, rank):
    h = torch.nn.functional.one_hot(batch.c[...,0], num_classes=len(vocab.atom_types)+1).to(rank)
    h2 = torch.nn.functional.one_hot(batch.c[...,1], num_classes=len(vocab.charge_types)+1).to(rank)
    e = torch.nn.functional.one_hot(batch.a, num_classes=len(vocab.bond_types)).to(rank)
    t = t.view(-1, 1, 1).expand(-1, h.size(1), -1).to(rank)
    features = torch.cat((t, h.float()), dim=2)
    return batch.x, features, h2, e, batch.mask, batch.y


class Logger:
    def __init__(self, name, log_dir):
        self.log_dir = log_dir
        self.name = name
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        with open(self.log_dir / f"{self.name}.log", "w") as f:
            f.write("")

    def info(self, message):
        with open(self.log_dir / f"{self.name}.log", "a") as f:
            f.write(f"{message}\n")

class ChekpointMonitor:
    def __init__(self, save_path, model_name):
        self.save_path = save_path
        self.model_name = model_name

    def save(self, model, optimizer, scheduler, step, epoch, loss):
        checkpoint = {
            "model": model.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "loss": loss,
        }
        torch.save(checkpoint,f"{self.save_path}/checkpoint.seml.{epoch}.pt")
        optimizer = {
            "optimizer": optimizer.state_dict(),
        }
        torch.save(optimizer, f"{self.save_path}/optimizer.seml.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("config_file", type=str)
    parser.add_argument("--gpu", default=None, type=int)
    # DDP configs:
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--type_model", default="model", type=str, help="type of model")
    parser.add_argument("--base_path", default="/scratch/project_465002040/riccardo/project/checkpoints/metamol", type=str, help="base path")
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="local rank for distributed training"
    )
    args = parser.parse_args()
    return args
    


def loss_fn(model, batch,noise_sampler, interporlator, 
            vocab, nll_loss,
            time_sampler, device, batch_id, world_size):

    local_skip_update = False

    ###########################
    ###### prepare data #######
    ###########################

    num_atoms = batch.num_atoms
    enc_noise = noise_sampler.sample_batch([num_atoms] * len(batch.x))

    try:

        batch = batch.random_rotate()
        enc_batch = MoleculeBatch(**vocab.encode(batch))
        # enc_batch = batch.encode(batch)

        t = time_sampler.sample_time(len(enc_batch.x))

        enc_noise_align, enc_t = interporlator.interpolate(enc_noise, enc_batch, t)

        x_t, h_t, _, e_t, mask_t, y_t = prepare_input(enc_t, t, vocab, device)
        x1, h1, c1, e1, mask1, y1 = prepare_input(enc_batch, t, vocab, device)

        
    except:

        local_skip_update = True

        print(f"Error in batch {batch_id}, skipping...")
        return None, None, None, None, None, None, None, None, None, local_skip_update
    

    ###########################
    ###### forward pass #######
    ###########################

    x1, y1, h1, e1, c1, mask1 = \
        x1.to(device), y1.to(device), h1.to(device), e1.to(device), c1.to(device), mask1.to(device)
    
    x_t, y_t, h_t, e_t, mask_t = \
        x_t.to(device), y_t.to(device), h_t.to(device), e_t.to(device), mask_t.to(device)


    x_p, y_p, h_p, hy_p, c_p, cy_p, e_p, ey_p = model(x=x_t, y=y_t, h=h_t, e=e_t, mask=mask_t)


    ##########################
    ###### compute loss ######
    ##########################

    pair_mask = torch.logical_or(mask1[...,None], mask1[:,None])

    coord_loss_x = (x_p - x1)**2 * (~mask1).unsqueeze(-1) 
    coord_loss_y = (y_p - y1)**2 * (~mask1).unsqueeze(-1)
    coord_loss_x = coord_loss_x.sum(dim=(1, 2)) / (~mask1).float().sum(-1, keepdims=True).clamp(min=1.0)
    coord_loss_y = coord_loss_y.sum(dim=(1, 2)) / (~mask1).float().sum(-1, keepdims=True).clamp(min=1.0)

    regularization_loss_bonds = ((e_p - ey_p)**2 * (~pair_mask)[...,None]).sum(dim=(1, 2, 3)) / (torch.sqrt((~pair_mask).sum(dim=(1, 2)).float()) + 1e-16)
    regularization_loss_types = ((h_p - hy_p)**2 * (~mask1)[...,None]).sum(dim=(1, 2)) / (torch.sqrt((~mask1).sum(dim=(1)).float()) + 1e-16)
    regularization_loss_charges = ((c_p - cy_p)**2 * (~mask1)[...,None]).sum(dim=(1, 2)) / (torch.sqrt((~mask1).sum(dim=(1)).float()) + 1e-16)
    
    type_loss = nll_loss(h_p, torch.argmax(h1[...,1:], dim=-1), mask=mask1, ignore_index=vocab.padding)
    type_loss_reg = nll_loss(hy_p, torch.argmax(h1[...,1:], dim=-1), mask=mask1, ignore_index=vocab.padding)
    bond_loss = nll_loss(e_p, torch.argmax(e1, dim=-1), mask=pair_mask)
    bond_loss_reg = nll_loss(ey_p, torch.argmax(e1, dim=-1), mask=pair_mask)
    charge_loss = nll_loss(c_p, torch.argmax(c1, dim=-1), mask=mask1, ignore_index=vocab.padding)
    charge_loss_reg = nll_loss(cy_p, torch.argmax(c1, dim=-1), mask=mask1, ignore_index=vocab.padding)

    adj_x_p = torch.zeros((x_p.shape[0], x_p.shape[1], x_p.shape[1]), device=x_p.device)
    adj_x_1 = torch.zeros((x1.shape[0], x1.shape[1], x1.shape[1]), device=x1.device)
    adj_y_p = torch.zeros((y_p.shape[0], y_p.shape[1], y_p.shape[1]), device=y_p.device)
    adj_y_1 = torch.zeros((y1.shape[0], y1.shape[1], y1.shape[1]), device=y1.device)
    
    for i in range(len(x_p)):

        mask_adja = (~mask1[i].unsqueeze(-1) * ~mask1[i].unsqueeze(0)).float()
        
        adj_x_1[i] = torch.cdist(x1[i], x1[i]) * mask_adja
        adj_y_1[i] = torch.cdist(y1[i], y1[i]) * mask_adja

        adj_x_p[i] = torch.cdist(x_p[i], x_p[i]) * mask_adja
        adj_y_p[i] = torch.cdist(y_p[i], y_p[i]) * mask_adja

    mask_bonds = (torch.argmax(e1, dim=-1) > 0).float() # constraint for bonds distances
    adj_constraint_loss_x = (torch.abs(adj_x_p - adj_x_1) \
                           * mask_bonds).mean(dim=(1, 2))
    adj_constraint_loss_y = (torch.abs(adj_y_p - adj_y_1) \
                           * mask_bonds).mean(dim=(1, 2))

    coord_loss_x = coord_loss_x + adj_constraint_loss_x * 1.3
    coord_loss_y = coord_loss_y + adj_constraint_loss_y * 1.3

    coord_loss_x = coord_loss_x.mean() / world_size
    coord_loss_y = coord_loss_y.mean() / world_size
    type_loss = type_loss.mean() / world_size
    bond_loss = bond_loss.mean() / world_size
    charge_loss = charge_loss.mean() / world_size
    regularization_loss_bonds = regularization_loss_bonds.mean() / world_size + (bond_loss_reg.mean() / world_size) * 0.5
    regularization_loss_types = regularization_loss_types.mean() / world_size + type_loss_reg.mean() / world_size
    regularization_loss_charges = regularization_loss_charges.mean() / world_size + charge_loss_reg.mean() / world_size

    loss = (coord_loss_x * 1.0 +
            coord_loss_y * 1.3 +
            type_loss * 0.2 + 
            bond_loss * 1.0 + 
            charge_loss * 1.0 +
            regularization_loss_bonds * 1.0 + 
            regularization_loss_types * 1.0 +
            regularization_loss_charges * 1.0)


    return loss, coord_loss_x.item(), coord_loss_y.item(), type_loss.item(), bond_loss.item(), charge_loss.item(), regularization_loss_bonds.item(), regularization_loss_types.item(), regularization_loss_charges.item(), local_skip_update


def main(rank, world_size, config_path, port):

    # identifier = datetime.now().strftime("%Y %m %d %H:%M").replace(" ", "_")
    identifier = datetime.now().strftime("%Y %m %d %H").replace(" ", "_")

    config = yaml.safe_load(open(config_path, "rb"))
    train_config = config["train"]
    model_version = train_config["model_version"]
    type_model = train_config["type_model"]
    chpt_path = train_config["checkpoint_path"]
    path_legacy_model_cpk = train_config["path_legacy_model_cpk"]
    path_legacy_model_opt = train_config["path_legacy_model_cpk"]
    dataset_name = train_config["dataset"]
    run_name = "model.version=" + model_version + "." + type_model
    save_path = Path(f"{chpt_path}/{run_name}.{dataset_name}")

    if not os.path.exists(chpt_path):
        save_path.mkdir(parents=True, exist_ok=True)

    master_node = rank == 0

    if world_size > 1:
        ddp_setup(rank, world_size, port)
    if rank == 0:
        logger = lambda x:print(x)
    else:
        logger = lambda x:x

    nll_loss = getattr(importlib.import_module(f"src.train.utils"), "nll_loss")
    vocabulary = getattr(importlib.import_module(f"src.data.vocabulary"), "Vocabulary")
    if str(dataset_name) == "geom":
        vocabulary = vocabulary()
    elif str(dataset_name) == "qm9":
        vocabulary = vocabulary(atom_types={"H", "C", "N", "O", "F"}) 
    else:
        print("Unknown dataset_name")

    GeometricInterpolant = getattr(importlib.import_module(f"src.data.interpolate"), "GeometricInterpolant")
    NoiseSampler = getattr(importlib.import_module(f"src.data.interpolate"), "NoiseSampler")
    TimeSampler = getattr(importlib.import_module(f"src.data.interpolate"), "TimeSampler")
    CheckpointManager = getattr(importlib.import_module(f"src.train.checkpoint"), "CheckpointManager")

    # Dataset class
    if str(dataset_name) == "geom":
        dataset_class = getattr(importlib.import_module(f"src.data.molecule"), "Molecule")
    elif str(dataset_name) == "qm9":
        dataset_class = getattr(importlib.import_module(f"src.data.molecule"), "FastPairsConfQM9")
    else:
        print("Unknown dataset_name")

    if str(dataset_name) ==  "geom":
        get_molecule_dataloader = getattr(importlib.import_module(f"src.data.molecule"), "get_dataloader")
    elif str(dataset_name) == "qm9":
        get_molecule_dataloader = getattr(importlib.import_module(f"src.data.molecule"), "get_dataloader_qm9") 
    else:
        print("Unknown dataset_name")

    whereisthemodel = f"src.models.flexiflow.ff"
    model_class = getattr(importlib.import_module(whereisthemodel), "FlexiFlow")

    batch_size = train_config["batch_size"]
    epochs = train_config["epochs"]
    seed = train_config["seed"]
    max_gradient_norm = train_config["max_gradient_norm"]

    if str(dataset_name) == "geom":  
        molecule_dataset = dataset_class(str(config["molecule_dataset"]["path"]), lazy=False)
    elif str(dataset_name) == "qm9":
        def read_from_file(filepath):
            bytes = filepath.read_bytes()
            return pickle.loads(bytes)
        output_path = Path(str(config["molecule_dataset"]["path"]))
        molecule_dataset = dataset_class(read_from_file(output_path))
    else:
        print("Unknown dataset_name when loading dataset, also the path can be wrong")

    model, optimizer, scheduler, ema_model, checkpoint_manager = CheckpointManager.restore_last_checkpoint(save_path, world_size=world_size, device=f'cuda:{rank}')

    if master_node:
        # writer = SummaryWriter(save_path / f"{identifier}_logs")
        # checkpoint_monitor = ChekpointMonitor(f"{str(save_path)}", "models")
        train_logger = Logger(f"train_rank={rank}", log_dir=save_path / f"{identifier}_logs")
        train_logger.info("epoch,batch,loss,coord_loss_x,coord_loss_y,type_loss,bond_loss,charge_loss,reg_loss_bonds,reg_loss_types,reg_loss_charges")
        model_loading_logger = Logger(f"model_loading_rank={rank}", log_dir=save_path / f"{identifier}_logs")
        model_loading_logger.info(f"Model version: {model_version}, Type of model: {type_model}")
        model_loading_logger.info(f"Checkpoint path: {save_path}")
        model_loading_logger.info(f"World size: {world_size}, Rank: {rank}, Port: {port}")
        model_loading_logger.info(f"Batch size: {batch_size}, Epochs: {epochs}, Seed: {seed}")
        model_loading_logger.info(f"Max gradient norm: {max_gradient_norm}")
        model_loading_logger.info(f"Model config: {config['model']}")
        model_loading_logger.info(f"Train config: {config['train']}")
        model_loading_logger.info(f"Flow config: {config['flow']}")

    batch_id = None
    if model is None:
        if master_node:
            model_loading_logger.info("No checkpoint found, training from scratch")
        model = model_class(**config["model"]).to(rank)
        if path_legacy_model_cpk != 'None':
            cpk = torch.load(path_legacy_model_cpk, map_location=f'cuda:{rank}')
            model.load_state_dict(cpk["model"])
            if master_node:
                model_loading_logger.info(f"Model correctly loaded from: {path_legacy_model_cpk}")
        avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
        if world_size > 1:
            model = DDP(model, device_ids=[rank])
            ema_model = torch.optim.swa_utils.AveragedModel(model.module, multi_avg_fn=avg_fn)
            if master_node:
                # print("Number of model parameters: ", sum(p.numel() for p in model.module.parameters() if p.requires_grad)/1e6, "M")
                model_loading_logger.info(f"Number of model parameters: {sum(p.numel() for p in model.module.parameters() if p.requires_grad)/1e6} M")

        else:
            ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=avg_fn)
            # print("Number of model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M")
            if master_node:
                model_loading_logger.info(f"Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6} M")

        checkpoint_manager = CheckpointManager(vocabulary, model, optimizer, str(save_path), scheduler=scheduler, swa=ema_model, seed=seed, device=rank)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, amsgrad=True, weight_decay=0.0)
        if path_legacy_model_opt != 'None':
            opt_cpk = torch.load(path_legacy_model_opt, map_location=f'cuda:{rank}')
            if "optimizer" in opt_cpk:
                for key in opt_cpk["optimizer"]["state"]:
                    for param in opt_cpk["optimizer"]["state"][key]:
                        if isinstance(opt_cpk["optimizer"]["state"][key][param], torch.Tensor):
                            opt_cpk["optimizer"]["state"][key][param] = opt_cpk["optimizer"]["state"][key][param].to(rank)
                optimizer.load_state_dict(opt_cpk["optimizer"])
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=config["train"]["warm_up_steps"])
        if path_legacy_model_cpk != 'None':
            if "scheduler" in cpk:
                scheduler.load_state_dict(cpk["scheduler"])
    else:

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=config["train"]["warm_up_steps"])
            checkpoint_manager.scheduler = scheduler

        if master_node:
            model_loading_logger.info("Checkpoint found, training resumed")
            model_loading_logger.info(f"Model loaded from {save_path}")


    start_batch = checkpoint_manager.batch
    start_epoch = checkpoint_manager.epoch
    seed = checkpoint_manager.seed
    vocabulary = checkpoint_manager.vocabulary

    coord_noise_std = config["flow"].get("coord_noise_std", 0.2)
    scale_ot_factor = config["flow"].get("scale_ot_factor", 0.25645364536453646)
    batch_ot = config["flow"].get("batch_ot", False)
    max_interp_steps = config["flow"].get("max_interp_steps", 1)
    scale_ot = config["flow"].get("scale_ot", True)
    sample_charges = config["flow"].get("sample_charges", False)
    symmetric_adj = config["flow"].get("symmetric_adj", False)   

    interporlator = GeometricInterpolant(vocabulary, coord_noise_std=coord_noise_std, batch_ot=batch_ot, max_steps=max_interp_steps)
    noise_sampler = NoiseSampler(vocabulary, scale_ot=scale_ot, scale_ot_factor=scale_ot_factor, sample_charges=sample_charges, symmetric_adj=symmetric_adj)

    time_sampler = TimeSampler(time_alpha=2.0, time_beta=1.0)

    n_batches = len(molecule_dataset) // (batch_size * world_size) if world_size > 1 else len(molecule_dataset) // batch_size

    model = model.train()
    step = 0
    if master_node:
        model_loading_logger.info(f"Starting training from epoch {start_epoch}, batch {start_batch}")
        model_loading_logger.info(f"Number of batches per epoch: {n_batches}")
    for epoch in range(start_epoch, epochs):

        dataloader = get_molecule_dataloader(molecule_dataset, batch_size, start_epoch=epoch, start_batch=start_batch, seed=seed, rank=rank, world_size=world_size)

        n_batches = len(dataloader)
        
        for batch_id, batch in enumerate(dataloader):
            optimizer.zero_grad()

            if isinstance(model, DDP):
                loss, coord_loss_x, coord_loss_y, type_loss, bond_loss, charge_loss, reg_loss_bonds, reg_loss_types, reg_loss_charges, local_skip_update = loss_fn(model.module if world_size > 1 else model,
                                                                                                                                                                batch, noise_sampler, interporlator, vocabulary,
                                                                                                                                                                nll_loss, time_sampler, rank, batch_id, world_size)
            else:
                loss, coord_loss_x, coord_loss_y, type_loss, bond_loss, charge_loss, reg_loss_bonds, reg_loss_types, reg_loss_charges, local_skip_update = loss_fn(model.module if world_size > 1 else model,
                                                                                                                                                                batch, noise_sampler, interporlator, vocabulary,
                                                                                                                                                                nll_loss, time_sampler, rank, batch_id, world_size)

            # skip_update = should_skip_update(local_skip_update, rank)

            if loss is not None:
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient_norm)
                
                # if not skip_update:

                optimizer.step()

                if isinstance(model, DDP):
                    ema_model.update_parameters(model.module)
                else:
                    ema_model.update_parameters(model)

                scheduler.step()

                if master_node:

                    if (step + 1) % 1000 == 0: 
                        checkpoint_manager.save_tmp(model.module if world_size > 1 else model, optimizer, scheduler, epoch, batch_id, swa=ema_model)
                        model_loading_logger.info(f"Temporary checkpoint saved at epoch {epoch}, batch {batch_id}")

                    if (step + 1) % 10 == 0:

                        train_logger.info(
                            f"{epoch},{batch_id}/{n_batches},{loss.item():.4f},{coord_loss_x:.4f},{coord_loss_y:.4f},{type_loss:.4f},{bond_loss:.4f},{charge_loss:.4f},{reg_loss_bonds:.4f},{reg_loss_types:.4f},{reg_loss_charges:.4f}"
                        )

                step = step + 1

            torch.cuda.empty_cache()

        if (batch_id is not None) and (rank == 0):

            checkpoint_manager.save(model.module if world_size > 1 else model, optimizer, scheduler, epoch, batch_id, swa=ema_model)

        start_batch = 0

    if world_size > 1:
        destroy_process_group()


def parser_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="unconditional.yml")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    
    _args = parser_arguments()
    if _args.config is not None:
        config_path = _args.config
    else:
        config_path = "unconditional.yml"

    print(f"GPU: {world_size:d}")
    if world_size == 1:
        main(0, world_size, config_path, port=0)
    else:
        # if gpu 0 sleep for 1 second to avoid  
        rank = int(os.environ.get("SLURM_PROCID", 0))
        port = get_distributed_port()
        print(f"Using port: {port}, GPU: {rank}")

        mp.spawn(main, args=(world_size, config_path, port+rank), nprocs=world_size)