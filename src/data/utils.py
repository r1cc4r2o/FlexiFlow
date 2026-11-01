import random

from rdkit import Chem
from rdkit.Geometry import Point3D
from torch.utils.data import DataLoader, Sampler
import numpy as np
import torch


_T = torch.Tensor


def to_rdkit(x, c, a):
    bond_types = {
      1: Chem.BondType.SINGLE,
      2: Chem.BondType.DOUBLE,
      3: Chem.BondType.TRIPLE,
      4: Chem.BondType.AROMATIC,
    }
    
    mol = Chem.RWMol()
    for atom_num, charge in c:
        atom = Chem.Atom(int(atom_num))
        atom.SetFormalCharge(int(charge))
        mol.AddAtom(atom)

    frontier = set()
    for (i,j) in zip(*np.where(a)):
        if (not (int(i), int(j)) in frontier) and (not (int(j), int(i)) in frontier):
            v = a[i,j]
            mol.AddBond(int(i), int(j), order=bond_types[int(v)])
            frontier.add((int(i), int(j)))
            frontier.add((int(j), int(i)))
    
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (_x, _y, _z) in enumerate(x):
        conf.SetAtomPosition(i, Point3D(float(_x), float(_y), float(_z)))
    mol.AddConformer(conf, assignId=True)
    mol = Chem.Mol(mol)
    mol.UpdatePropertyCache()
    Chem.GetSSSR(mol)
    return mol


class FixedStartSampler(Sampler):
    def __init__(self, dataset, batch_size, seed, start_epoch=0, start_batch=0):
        state = np.random.get_state()

        start_index = start_batch * batch_size
        indices = np.arange(len(dataset))
        np.random.seed((seed + start_epoch) % (2**32-1))
        np.random.shuffle(indices)
        self.indices = indices[start_index:]

        np.random.set_state(state)
        

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_dataloader(dataset, batch_size, start_epoch, start_batch, seed=None, drop_last=True):
    sampler = FixedStartSampler(dataset, batch_size, seed, start_epoch, start_batch)
    collate_fn = None
    if hasattr(dataset.__class__, "collate_fn"): 
        collate_fn = dataset.__class__.collate_fn
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=drop_last, shuffle=False, collate_fn=collate_fn)


def set_seed(seed):
    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you use multi-GPU

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
