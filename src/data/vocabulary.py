import json

from rdkit import Chem
import torch


class Vocabulary:
    def __init__(self, atom_types=None, charge_types=None, bond_types=None, std=None):
        # assume paddig 0
        self.padding = 0
        if std is None:
            self.std = 2.407038688659668
        else:
            self.std = std
        if atom_types is None:
            self.atom_types = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
        else:
            self.atom_types = atom_types
        
        if charge_types is None:
            self.charge_types = { 0, -1, 1 }
        else:
            self.charge_types = charge_types

        if bond_types is None:
            self.bond_types = { 0, 1, 2, 3, 4 }
        else:
            self.bond_types = bond_types

        pt = Chem.GetPeriodicTable()
        self.atom_num_types = {pt.GetAtomicNumber(k) for k in self.atom_types}

        self.atom_map = torch.zeros(max(self.atom_num_types) + 1, dtype=torch.long)
        self.atom_inv_map = torch.zeros(len(self.atom_num_types) + 1, dtype=torch.long)
        for i, k in enumerate(sorted(list(self.atom_num_types))):
            self.atom_map[k] = i + 1 
            self.atom_inv_map[i + 1] = k


        self.min_charge_key = min(self.charge_types)
        self.max_charge_key = max(self.charge_types)

        self.charge_map = torch.full((self.max_charge_key - self.min_charge_key  + 1,), -1, dtype=torch.long)
        self.charge_inv_map = torch.zeros(len(self.charge_types) + 1, dtype=torch.long)
        for i, k in enumerate(sorted(list(self.charge_types))):
            self.charge_map[k - self.min_charge_key ] = i + 1
            self.charge_inv_map[i + 1] = k

        self.bond_map = torch.zeros(max(self.bond_types) + 1, dtype=torch.long)
        self.bond_inv_map = torch.zeros(len(self.bond_types) + 1, dtype=torch.long)
        for i, k in enumerate(sorted(list(self.bond_types))):
            self.bond_map[k] = i
            self.bond_inv_map[i] = k


    def validate_batch(self, batch):
        mask = batch.mask
        
        at = set(batch.c[~mask][...,0].detach().cpu().numpy().ravel())
        ct = set(batch.c[~mask][...,1].detach().cpu().numpy().ravel())
        bt = set(batch.a.detach().cpu().numpy().ravel())

        return len(at & self.atom_num_types) == len(at) and len(ct & self.charge_types) == len(ct) and len(bt & self.bond_types) == len(bt)

    def encode(self, batch):
        mask = batch.mask
        x = torch.clone(batch.x) / self.std
        y = torch.clone(batch.y) / self.std
        c = torch.clone(batch.c)
        a = None
        
        c[...,0][~mask] = self.atom_map[c[...,0][~mask]]
        c[...,1][~mask] = self.charge_map[c[...,1][~mask] - self.min_charge_key]
        x[mask] = self.padding
        y[mask] = self.padding
        c[mask] = self.padding
        
        if batch.a is not None:
            a = torch.clone(batch.a)
            pair_mask = torch.logical_or(mask[...,None], mask[:,None])
            a[pair_mask] = self.padding
            a_shape = a.shape
            a = self.bond_map[a.flatten()].view(*a_shape)
            
        return {"x":x,"y":y, "c": c, "a": a, "mask": mask}
        
        
    def decode(self, batch):
        mask = batch.mask
        pair_mask = torch.logical_or(mask[...,None], mask[:,None])
        x = torch.clone(batch.x) * self.std
        y = torch.clone(batch.y) * self.std
        c = torch.clone(batch.c)
        a = None
        c[...,0][~mask] = self.atom_inv_map[c[...,0][~mask]]
        c[...,1][~mask] = self.charge_inv_map[c[...,1][~mask]]
        # add padding
        x[mask] = self.padding
        y[mask] = self.padding
        c[mask] = self.padding
        
        if batch.a is not None:
            a = torch.clone(batch.a)
            a_shape = a.shape
            a = self.bond_inv_map[a.view(-1)].view(*a_shape)
            a[pair_mask] = self.padding

        return {"x":x, "y":y, "c": c, "a": a, "mask": mask}

    def get_state_dict(self):
        return { 
                 "atom_types": self.atom_types,
                 "charge_types": self.charge_types,
                 "bond_types": self.bond_types,
                 "std": self.std,
               }
        

    @classmethod
    def load_from_state_dict(cls, save_dict):
        vocabulary = cls(**save_dict)
        return vocabulary

    
    @classmethod
    def load_from_file(cls, file):
        save_dict = json.load(open(file))
        vocabulary = cls.load_from_state_dict(save_dict)
        return vocabulary