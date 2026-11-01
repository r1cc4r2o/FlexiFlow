from pathlib import Path
import os

import numpy as np
import h5py
import torch


class ProteinLigand(torch.utils.data.Dataset):
    def __init__(self, data_path, min_qed=0.5, min_pocket_size=120, max_pocket_size=500):
        if isinstance(data_path, str):
            data_path = [data_path]
        self.data = {}
        for dp in data_path:
            self.data = {**self.data, **np.load(dp, allow_pickle=True)[()]}
        to_delete = set()
        for k, v in self.data.items():
            if v["qed"] < min_qed:
                to_delete.add(k)
            if (len(v["protein"][0]) > max_pocket_size) or (len(v["protein"][0]) < min_pocket_size):
                to_delete.add(k)
        for k in to_delete:
            del self.data[k]
        self.idx_to_key = {i:k for i,k in enumerate(self.data)}
        
    def __getitem__(self, i):
        k = self.idx_to_key[i]
        px, pc = self.data[k]["protein"]
        x, c, a = self.data[k]["ligand"]
        return torch.tensor(x), torch.tensor(c), torch.tensor(a), torch.tensor(px), torch.tensor(pc), k

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        x, c, a, px, pc, pdb_id = zip(*batch)
        xb = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0).float()
        cb = torch.nn.utils.rnn.pad_sequence(c, batch_first=True, padding_value=0).long()
        ab = torch.zeros((xb.shape[0], xb.shape[1], xb.shape[1]), device=xb.device).long()
        mask = torch.zeros((xb.shape[0], xb.shape[1]), dtype=torch.bool, device=xb.device)
        for i, _a in enumerate(a):
            ab[i, :_a.shape[0], :_a.shape[0]] = _a
            mask[i, _a.shape[0]:] = True

        pxb = torch.nn.utils.rnn.pad_sequence(px, batch_first=True, padding_value=0.0).float()
        pcb = torch.nn.utils.rnn.pad_sequence(pc, batch_first=True, padding_value=0).long()
        pmask = torch.zeros((pxb.shape[0], pxb.shape[1]), dtype=torch.bool, device=pxb.device)
        for i, _a in enumerate(px):
            pmask[i, _a.shape[0]:] = True
        
        return {"x": xb, "c": cb, "a": ab,  "mask": mask, "pdb_id": pdb_id}, {"x": pxb, "c": pcb, "a": None, "mask": pmask}


class ProteinLigandBucketSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, seed, start_epoch=0, start_batch=0):
        self.batch_size = batch_size
        backup_state = np.random.get_state()
        start_index = start_batch * batch_size
        np.random.seed((seed + start_epoch) % (2**32-1))
        self.indices = np.arange(len(dataset))
        np.random.shuffle(self.indices)
        self.indices = self.indices[start_index:]
        np.random.set_state(backup_state)
        
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_dataloader(dataset, batch_size, start_epoch, start_batch, seed=None):
    sampler = ProteinLigandBucketSampler(dataset, batch_size, seed, start_epoch, start_batch)
    collate_fn = None
    if hasattr(dataset.__class__, "collate_fn"): 
        collate_fn = dataset.__class__.collate_fn
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, collate_fn=collate_fn)
