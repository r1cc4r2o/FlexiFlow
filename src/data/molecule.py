from pathlib import Path
import os

import numpy as np
import h5py
import torch

from tqdm import tqdm
import pickle

from src.data.interpolate import find_rigid_alignment

def batch_rotation_matrices(alphas, betas, gammas):
    """Generate a batch of 3D rotation matrices for each batch in PyTorch."""
    cos = torch.cos
    sin = torch.sin

    R_x = torch.stack([
        torch.tensor([[1, 0, 0], 
                      [0, cos(a), -sin(a)], 
                      [0, sin(a), cos(a)]]) 
        for a in alphas
    ])

    R_y = torch.stack([
        torch.tensor([[cos(b), 0, sin(b)], 
                      [0, 1, 0], 
                      [-sin(b), 0, cos(b)]]) 
        for b in betas
    ])

    R_z = torch.stack([
        torch.tensor([[cos(g), -sin(g), 0], 
                      [sin(g), cos(g), 0], 
                      [0, 0, 1]]) 
        for g in gammas
    ])

    # Compute the final rotation matrix R = R_z @ R_y @ R_x (batch-wise)
    R = torch.bmm(R_z, torch.bmm(R_y, R_x))  # Shape: (4, 3, 3)
    return R


class MoleculeBatch:
    def __init__(self, x, y, c, a, mask, names=None):
        self.x = x
        self.y = y
        self.c = c
        self.a = a
        self.mask = mask
        self.names = names

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.c = self.c.to(device)
        if self.a is not None:
            self.a = self.a.to(device)
        self.mask = self.mask.to(device)
        return self

    def __iter__(self):
        # Return a fresh generator each time
        for i in range(len(self)):
            m = self.mask[i]
            mi = (~m).long().sum()
            x = self.x[i][:mi]
            y = self.y[i][:mi]
            c = self.c[i][:mi]
            if self.a is not None:
                a = self.a[i][:mi]
            else:
                a = None
            yield x, y, c, a, m
            
    def __len__(self):
        return len(self.x)

    @property
    def com(self):
        inv_mask = (~self.mask).float()
        com_x, com_y= (self.x * inv_mask.unsqueeze(-1)).sum(1, keepdims=True), \
                            (self.y * inv_mask.unsqueeze(-1)).sum(1, keepdims=True)
        com_x, com_y = com_x / inv_mask.unsqueeze(-1).sum(1, keepdims=True).clamp(min=1.0), \
                            com_y / inv_mask.unsqueeze(-1).sum(1, keepdims=True).clamp(min=1.0)
        return com_x, com_y

    def remove_com(self, com_x=None, com_y=None):
        if com_x is None or com_y is None:
            com_x, com_y = self.com
        inv_mask = (~self.mask).float().unsqueeze(-1)
        x = (torch.clone(self.x) - com_x) * inv_mask
        y = (torch.clone(self.y) - com_y) * inv_mask

        if self.a is None:
            a = None
        else:
            a = torch.clone(self.a)
        return MoleculeBatch(x, y, torch.clone(self.c), a, torch.clone(self.mask))

    @property
    def num_atoms(self):
        return int(self.mask.shape[1])

    def random_rotate(self, return_rotation_matrix=False):
        alphas = torch.rand(len(self.x)) * 2.0 * np.pi
        betas  = torch.rand(len(self.x)) * 2.0 * np.pi
        gammas = torch.rand(len(self.x)) * 2.0 * np.pi
        
        R = batch_rotation_matrices(alphas, betas, gammas)  # (N, 3, 3)
        R = R.to(self.x.device)

        mb = self.rotate(R)
        
        if return_rotation_matrix:
            return mb, R
        return mb

    def rotate(self, R):
        x = torch.clone(self.x)
        y = torch.clone(self.y)
        m = torch.clone(self.mask)

        # Apply rotation: Use batch matrix multiplication
        xr = torch.bmm(x, R.transpose(1, 2))  # (N, C, 3) @ (N, 3, 3)^T -> (N, C, 3)
        xr = xr * (~m).float().unsqueeze(-1)
        yr = torch.bmm(y, R.transpose(1, 2))  # (N, C, 3) @ (N, 3, 3)^T -> (N, C, 3)
        yr = yr * (~m).float().unsqueeze(-1)

        if self.a is None:
            a = None
        else:
            a = torch.clone(self.a)
        return MoleculeBatch(xr, yr, torch.clone(self.c), a, m)
        

class Molecule(torch.utils.data.Dataset):
    def __init__(self, prefix, buckets=None, lazy=True):
        self.h5files = {}
        self.lengths = {}
        self.n = 0
        self.lazy = lazy
        if buckets is None:
            buckets = [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 96, 192]
        self.buckets = np.array(buckets)
        for fname in Path(prefix).glob("*.h5"):
            n_atoms = int(os.path.splitext(fname.name)[0])
            if lazy:
                self.h5files[n_atoms] = h5py.File(fname, "r")
            else:
                h5 = h5py.File(fname, "r")
                self.h5files[n_atoms] = {}
                self.h5files[n_atoms]["X"] = h5["X"][:]
                self.h5files[n_atoms]["C"] = h5["C"][:]
                self.h5files[n_atoms]["A"] = h5["A"][:]
                h5.close()
                
            self.lengths[n_atoms] = len(self.h5files[n_atoms]["X"])
            self.n += self.lengths[n_atoms]
        self.idx_bounds = np.zeros((len(self.lengths), 3), np.uint32)

        n = 0
        for i, k in enumerate(sorted(self.lengths.keys())):
            self.idx_bounds[i] = [n, n+self.lengths[k], k]
            n = n + self.lengths[k]
        # map i -> start_internal_index, end_internal_index, n_atoms

        self.paired_data_xy = np.load("../../cache/dataloader_fixed_targetssets_geomdrugs.npy")


    def __getbothitems__(self, i, j):
        i, j = i % len(self), j % len(self)
        idx_i, idx_j = (self.idx_bounds[...,0] > i).argmax() - 1, (self.idx_bounds[...,0] > j).argmax() - 1
        if idx_i < 0:
            idx_i = len(self.idx_bounds) - 1
        if idx_j < 0:
            idx_j = len(self.idx_bounds) - 1
        bucket_idx_i, bucket_idx_j = i - self.idx_bounds[idx_i][0], j - self.idx_bounds[idx_j][0]
        n_atoms = int(self.idx_bounds[idx_i][2])
        x, y = torch.tensor(self.h5files[n_atoms]["X"][bucket_idx_i]), torch.tensor(self.h5files[n_atoms]["X"][bucket_idx_j])
        c = torch.tensor(self.h5files[n_atoms]["C"][bucket_idx_i])
        a = torch.tensor(self.h5files[n_atoms]["A"][bucket_idx_i])
        return x, c, a, y
    
    def __getitem__(self, i):
        data_xy = self.paired_data_xy[np.logical_or(self.paired_data_xy[:,0] == i, self.paired_data_xy[:,1] == i)]
        i_target, i_ensemble = data_xy[0] if len(data_xy) == 1 else data_xy[np.random.randint(len(data_xy))]
            
        x, c, a, y = self.__getbothitems__(i_target, i_ensemble)

        # x, y = x.cuda(), y.cuda()
        x = x - x.mean(dim=0, keepdim=True)  # Center the coordinates
        y = y - y.mean(dim=0, keepdim=True)  # Center the coordinates
        R, _t = find_rigid_alignment(x,y)
        y = y.mm(R.T) + _t
        # x, y = x.cpu(), y.cpu()

        return x, c, a, y 

    def __len__(self):
        return self.n

    def __del__(self):
        if self.lazy:
            for k, v in self.h5files.items():
                v.close()

    @staticmethod
    def collate_fn(batch):
        x, c, a, y = zip(*batch)
        xb = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0).float()
        yb = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0.0).float()
        cb = torch.nn.utils.rnn.pad_sequence(c, batch_first=True, padding_value=0).long()
        ab = torch.zeros((xb.shape[0], xb.shape[1], xb.shape[1]), device=xb.device).long()
        mask = torch.zeros((xb.shape[0], xb.shape[1]), dtype=torch.bool, device=xb.device)
        for i, _a in enumerate(a):
            ab[i, :_a.shape[0], :_a.shape[0]] = _a
            mask[i, _a.shape[0]:] = True
        return MoleculeBatch(**{"x": xb, "y": yb, "c": cb, "a": ab, "mask": mask})

    @property
    def scaling_ot_factor(self):
        stats = {}
        for i in range(len(self)):
            x = self[i][0]
            natoms = len(x)
            if natoms not in stats:
                stats[natoms] = [0.,0.,0.]
            mean, var, n =  self.update_stats(*stats[natoms], x.numpy().ravel())
            stats[natoms] = [mean, var, n]
        x, y = [], []
        for k,v in stats.items():
            x.append(k)
            y.append(v[1])
        x = np.array(x)
        y = np.array(y) ** 0.5
        k_best = np.mean(y / np.log(x+1))
        return k_best

    @staticmethod    
    def update_stats(mean_prev, var_prev, n, x):
        offset = len(x)
  
        # Update mean
        mean_new = (n * mean_prev + np.sum(x)) / (n + offset)

        # Update variance
        if n > 0:
            var_new = ( n * var_prev + ((x - mean_new)**2).sum() + n * (mean_new - mean_prev)**2 ) / ( n + offset)
        else:
            if len(x) > 1:
                var_new = np.var(x)
            else:
                var_new = 0

        # Update standard deviation
        n += offset
        return mean_new, var_new, n



class FastPairsConfQM9(torch.utils.data.Dataset):
    def __init__(self, data, silence_x=False):
        self.x, self.y, self.c, self.a = [], [], [], []
        for item in tqdm(data):
            item = pickle.loads(item)
            closest_to_average_id = int(item['conformers']['closest_to_average_id'])
            for idx, conf in enumerate(item['conformers']['all_conformers']):
                if idx == closest_to_average_id:
                    continue
                self.x.append(item['conformers']['all_conformers'][closest_to_average_id]['positions'])
                self.y.append(conf['positions'])
                self.c.append(torch.concat([item['atomics'].unsqueeze(1), item['charges'].unsqueeze(1)], dim=1))
                self.a.append(self.edge_list_to_adjacency_matrix(item['bonds']))
        del data

        # Set x to zero if True
        self.silence_x = silence_x

    def __len__(self):
        return len(self.x)

    def edge_list_to_adjacency_matrix(self, edge_tensor):
        max_node = edge_tensor[:, :2].max().item()
        num_nodes = max_node + 1
        adj_matrix = torch.zeros(num_nodes, num_nodes, dtype=edge_tensor.dtype)
        sources = edge_tensor[:, 0]
        targets = edge_tensor[:, 1]
        bond_types = edge_tensor[:, 2]
        adj_matrix[sources, targets] = bond_types
        adj_matrix[targets, sources] = bond_types
        return adj_matrix
    
    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        R, t = find_rigid_alignment(y, x)
        y = y.mm(R.T) + t
        if self.silence_x:
            x = torch.zeros_like(x)
        return x, self.c[idx], self.a[idx], y
    
    @staticmethod
    def collate_fn(batch):
        x, c, a, y = zip(*batch)
        xb = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0).float()
        yb = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0.0).float()
        cb = torch.nn.utils.rnn.pad_sequence(c, batch_first=True, padding_value=0).long()
        ab = torch.zeros((xb.shape[0], xb.shape[1], xb.shape[1]), device=xb.device).long()
        mask = torch.zeros((xb.shape[0], xb.shape[1]), dtype=torch.bool, device=xb.device)
        for i, _a in enumerate(a):
            ab[i, :_a.shape[0], :_a.shape[0]] = _a
            mask[i, _a.shape[0]:] = True
        return MoleculeBatch(**{"x": xb, "y": yb, "c": cb, "a": ab, "mask": mask})



class MolecularBucketSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, seed, start_epoch=0, start_batch=0, rank=0, world_size=1):
        self.idx_bounds = dataset.idx_bounds
        self.batch_size = batch_size
        self.buckets = dataset.buckets
        self.rank = rank
        self.world_size = world_size
        
        backup_state = np.random.get_state()
        
        start_index = start_batch * batch_size
        np.random.seed((seed + start_epoch) % (2**32-1))
        indices = self._get_indices()
        li = len(indices) // self.world_size * self.rank
        ri = li + len(indices) // self.world_size
        indices = indices[li:ri]
        self.indices = indices[:, start_index:].ravel()
        
        np.random.set_state(backup_state)

    def _get_indices(self):
        buckets = self.buckets
        batch_size = self.batch_size
        idx = self.idx_bounds
        all_idx = []
        for lb, rb in zip(buckets[:-1], buckets[1:]):
            cluster = (idx[...,-1] < rb) & (idx[...,-1] >= lb)
            if cluster.sum() == 0:
                continue
            li, ri = idx[cluster][:, :2].min(), idx[cluster][:, :2].max()
            local_idx = np.arange(li, ri)
            np.random.shuffle(local_idx)
            n_batches = len(local_idx) // batch_size // self.world_size * self.world_size
            all_idx += [local_idx[:n_batches * batch_size].reshape(-1, batch_size)]
        all_idx = np.concatenate(all_idx, 0)
        all_idx_idx = np.arange(len(all_idx))
        np.random.shuffle(all_idx_idx)
        all_idx = all_idx[all_idx_idx]    
        return all_idx
        
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    


class SimpleMolecularSamplerGeom(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, seed, start_epoch=0, start_batch=0, rank=0, world_size=1):
        self.nr_indices = len(dataset)
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

        # save/restore RNG state so we don't disturb caller's numpy RNG
        backup_state = np.random.get_state()

        # offset in samples (not in batches)
        start_index = (start_batch * batch_size) // world_size

        # deterministic shuffle per epoch
        np.random.seed((seed + start_epoch) % (2**32 - 1))

        id_paired_with_atoms_counts = np.load("/home/jovyan/metamol/cache/paired_ids_counts_atoms_geomdrugs.npy")
        counts = id_paired_with_atoms_counts[:, 1]
        mean, std = np.mean(counts), np.std(counts)
        all_idx = id_paired_with_atoms_counts[(counts > mean -  std) & (counts < mean + std)][:,0]

        # all_idx = np.arange(self.nr_indices)
        np.random.shuffle(all_idx)

        # make total usable indices divisible by world_size, then slice for each worker
        indices_per_worker = self.nr_indices // self.world_size
        total_usable = indices_per_worker * self.world_size
        truncated = all_idx[:total_usable]

        li = self.rank * indices_per_worker
        ri = li + indices_per_worker
        worker_indices = truncated[li:ri]

        # apply start_index offset (may produce empty array)
        if start_index >= len(worker_indices):
            worker_indices = np.empty(0, dtype=int)
        else:
            worker_indices = worker_indices[start_index:]

        self.indices = worker_indices.copy()

        # restore RNG state
        np.random.set_state(backup_state)

    def __iter__(self):
        # yield python ints
        return iter(self.indices.tolist())

    def __len__(self):
        return len(self.indices)



class SimpleMolecularSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, seed, start_epoch=0, start_batch=0, rank=0, world_size=1):
        self.nr_indices = len(dataset)
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

        # save/restore RNG state so we don't disturb caller's numpy RNG
        backup_state = np.random.get_state()

        # offset in samples (not in batches)
        start_index = (start_batch * batch_size) // world_size

        # deterministic shuffle per epoch
        np.random.seed((seed + start_epoch) % (2**32 - 1))
        all_idx = np.arange(self.nr_indices)
        np.random.shuffle(all_idx)

        # make total usable indices divisible by world_size, then slice for each worker
        indices_per_worker = self.nr_indices // self.world_size
        total_usable = indices_per_worker * self.world_size
        truncated = all_idx[:total_usable]

        li = self.rank * indices_per_worker
        ri = li + indices_per_worker
        worker_indices = truncated[li:ri]

        # apply start_index offset (may produce empty array)
        if start_index >= len(worker_indices):
            worker_indices = np.empty(0, dtype=int)
        else:
            worker_indices = worker_indices[start_index:]

        self.indices = worker_indices.copy()

        np.random.set_state(backup_state)

    def __iter__(self):
        return iter(self.indices.tolist())

    def __len__(self):
        return len(self.indices)


def get_dataloader(dataset, batch_size, start_epoch, start_batch, seed=None, rank=0, world_size=1, num_workers=0):
    sampler = SimpleMolecularSamplerGeom(dataset, batch_size, seed, start_epoch, start_batch, rank=rank, world_size=world_size)
    collate_fn = None
    if hasattr(dataset.__class__, "collate_fn"): 
        collate_fn = dataset.__class__.collate_fn
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0, drop_last=True)


def get_dataloader_qm9(dataset, batch_size, start_epoch, start_batch, seed=None, rank=0, world_size=1, num_workers=0):
    sampler = SimpleMolecularSampler(dataset, batch_size, seed, start_epoch, start_batch, rank=rank, world_size=world_size)
    collate_fn = None
    if hasattr(dataset.__class__, "collate_fn"): 
        collate_fn = dataset.__class__.collate_fn
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0, drop_last=True)
