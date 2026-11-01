from typing import Callable
from typing import Optional

from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
import numpy as np
import torch


from src.data.molecule import MoleculeBatch
from src.data.vocabulary import Vocabulary


def find_rigid_alignment(A, B):
    """
    R, t = find_rigid_alignment(A, B)
    NewA = A.mm(R.T) + t
    """ 
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


class MolHelper:
    def __init__(self, x, c, a, mask, vocab):
        self.x = np.asarray(x)
        self.c = np.asarray(c)
        self.a = np.asarray(a)
        self.mask = np.asarray(mask)
        self.vocab = vocab

    def align(self, mol_ref, max_steps):
        x = self.x[~mol_ref.mask]
        x_ref = mol_ref.x[~mol_ref.mask]
        x_aligned, cost, idx = icp(x, x_ref, max_steps=max_steps)
        c = np.array(self.c[idx])
        a = np.array(self.a[idx,:][:,idx])
        return MolHelper(x_aligned, c, a, self.mask[:len(x_aligned)], self.vocab), cost

    @classmethod
    def create_from_batch(cls, batch, vocab):
        """
        Return a list of MolHelper objects
        """
        mols = []
        for i in range(len(batch)):
            x = batch.x[i]
            c = batch.c[i]
            a = batch.a[i]
            mask = batch.mask[i]
            mols.append(cls(x, c, a, mask, vocab))
        return mols

    def __len__(self):
        return (~self.mask).sum()



def icp(x, x_ref, max_steps=1, use_neighbors=False):
    """
    Perform the Iterative Closest Point (ICP) algorithm to align `x` to `x_ref`.
    Note `x_ref` is kept fixed.

    Parameters
    ----------
    x : numpy.ndarray
        A (N, 3) array representing the source point set in 3D space, where N is the number of points.
    y : numpy.ndarray
        A (N, 3) array representing additional atomic coordinates or features.
    x_ref : numpy.ndarray
        A (N, 3) array representing the target point set in 3D space, where N is the number of points.
    max_steps : int
        The maximum number of iterations to run the ICP algorithm.
    use_neighbors: bool, optional
        If True, use 4 neighbors to estimate the context of an atom 
        Defaults to False

    Returns
    -------
    x_aligned : numpy.ndarray
        A (N, 3) array of the source points `x` after applying the alignment transformation.
    cost : float
        The final mean square error (RMSE) between the aligned source points and the target points.
    idx: numpy.ndarray
        A (N) array containing the indices to align `x` to `x_ref`
    """
    _x = np.array(x_ref).copy().astype(np.float32)
    _y = np.array(x).copy().astype(np.float32)
    _x = _x - _x.mean(0, keepdims=True)
    com_y = _y.mean(0, keepdims=True)
    _y = _y - com_y
    cost = (((_x - _y)**2).sum(-1)**0.5).sum()

    idx = np.arange(len(_y))
    for i in range(max_steps):
        cost_matrix = ((_x[:,None] - _y[None])**2).sum(-1)**0.5
        
        # add neighbors info
        cx = ((_x[:,None] - _x[None])**2).sum(-1)
        cy = ((_y[:,None] - _y[None])**2).sum(-1)
        cx = np.sort(cx, axis=1)[...,:5].sum(axis=1)[:,None]
        cy = np.sort(cy, axis=1)[...,:5].sum(axis=1)[None]
        cost_matrix += np.abs(cx**0.5 - cy**0.5)

        _, permuted_idxs = linear_sum_assignment(cost_matrix)

        _y = _y[permuted_idxs]
        idx = idx[permuted_idxs]
    
        rot, _ = Rotation.align_vectors(_x, _y)
        _y = rot.apply(_y)
        com_y = rot.apply(com_y)
        new_cost = (((_x - _y)**2).sum(-1)**0.5).sum()
        if np.abs(new_cost - cost) < 1e-3:
            break
        cost = new_cost
    return _y + com_y, cost, idx


class NoiseSampler:
    def __init__(
        self,
        vocab: Vocabulary,
        scale_ot: bool = False,
        scale_ot_factor: float = 0.2,
        sample_charges: bool = False,
        symmetric_adj: bool = False,
    ):
        self.vocab = vocab
        self.scale_ot = scale_ot
        self.scale_ot_factor = scale_ot_factor
        self.sample_charges = sample_charges
        self.symmetric_adj = symmetric_adj
        self.n_atom_types_min = vocab.atom_map[vocab.atom_map > 0].min()
        self.n_atom_types_max = vocab.atom_map[vocab.atom_map > 0].max()
        
        self.n_charge_types_min = vocab.charge_map[vocab.charge_map > 0].min()
        self.n_charge_types_max = vocab.charge_map[vocab.charge_map > 0].max()
        
        self.n_bond_types_min = vocab.bond_map[vocab.bond_map >= 0].min()
        self.n_bond_types_max = vocab.bond_map[vocab.bond_map >= 0].max()

    
    def sample(self, n_atoms: int):
        x_0 = torch.randn(n_atoms, 3)
        y_0 = torch.randn(n_atoms, 3)
        # if self.scale_ot:
        #    x_0 = x_0 * np.log(n_atoms + 1) * self.scale_ot_factor
            
        a_0 = torch.randint(self.n_atom_types_min, self.n_atom_types_max+1, (n_atoms,1))
        if self.sample_charges:
            c_0 = torch.randint(self.n_charge_types_min, self.n_charge_types_max + 1, (n_atoms,1))
        else:
            c_0 = torch.ones_like(a_0) * self.vocab.charge_map[0]
            
        if self.symmetric_adj:
            adj_0 = torch.zeros((n_atoms, n_atoms), dtype=a_0.dtype)
            rows, cols = torch.triu_indices(n_atoms, n_atoms, offset=1)
            values = torch.randint(self.n_bond_types_min, self.n_bond_types_max + 1, (len(rows),))
            adj_0[rows, cols] = values
            adj_0[cols, rows] = values
        else:
            adj_0 = torch.randint(self.n_bond_types_min, self.n_bond_types_max + 1, (n_atoms,n_atoms))
            
        return x_0, y_0, torch.cat((a_0, c_0), -1), adj_0


    def sample_batch(self, n_atoms: list):
        batch = []
        n_examples = len(n_atoms)
        max_n_atoms = max(n_atoms)
        x = torch.zeros((n_examples, max_n_atoms, 3), dtype=torch.float32)
        y = torch.zeros((n_examples, max_n_atoms, 3), dtype=torch.float32)
        c = torch.zeros((n_examples, max_n_atoms, 2), dtype=torch.int64)
        a = torch.zeros((n_examples, max_n_atoms, max_n_atoms), dtype=torch.int64)
        mask = torch.zeros((n_examples, max_n_atoms), dtype=torch.bool)
        for i, na in enumerate(n_atoms):
            _x, _y, _c, _a = self.sample(na)
            x[i, :len(_x)] = _x
            y[i, :len(_x)] = _y
            c[i, :len(_c)] = _c 
            a[i, :len(_a), :len(_a)] = _a
            mask[i, len(_x):] = True

        pair_mask = torch.logical_or(mask[...,None], mask[:,None])
        x[mask] = self.vocab.padding
        y[mask] = self.vocab.padding
        c[mask] = self.vocab.padding
        a[pair_mask] = self.vocab.padding
        return MoleculeBatch(**{"x": x, "y": y, "c": c, "a": a, "mask": mask})


class GeometricInterpolant:
    def __init__(
        self,
        vocab: Vocabulary,
        coord_noise_std: float = 0.2,
        batch_ot: bool = False,
        max_steps: int = 1,
    ):

        self.coord_noise_std = coord_noise_std
        self.batch_ot = batch_ot
        self.vocab = vocab
        self.max_steps = max_steps

    def match(self, batch_mols, batch_mols_ref):
        """
        Align a batch of molecules to a reference batch.

        Parameters
        ----------
        batch_mols : dict
            A dictionary containing molecular data with the following keys:
            - `x` : torch.tensor 
                A (N, 3) array representing atomic coordinates of the molecules.
            - `y` : torch.tensor 
                A (N, 3) array representing additional atomic coordinates or features.
            - `c` : torch.tensor 
                A (N, 2) array representing atomic types and charges.
            - `a` : torch.tensor 
                A (N, N) array representing adjacency matrix.
            - `mask` : torch.tensor 
                A (N,) array representing mask indicating valid entries in the batch. (0 keep, 1 mask)
        batch_mols_ref : dict
            A dictionary containing reference molecular data with the same keys and structure as `batch_mols`.

        Returns
        -------
        mols_aligned : dict
            A dictionary containing the transformed version of `batch_mols`, with the same keys {x, y, c, a, mask}, aligned to match `batch_mols_ref`.
        """

        mols = MolHelper.create_from_batch(batch_mols, self.vocab)
        mols_ref = MolHelper.create_from_batch(batch_mols_ref, self.vocab)

        y, y_ref, _mask = batch_mols.y, batch_mols_ref.y, batch_mols_ref.mask
        ya = torch.ones_like(batch_mols.y, dtype=torch.float32) * self.vocab.padding
        for _idx in range(len(y_ref)):
            R, _translation = find_rigid_alignment(y[_idx][~_mask[_idx]], y_ref[_idx][~_mask[_idx]])
            ya[_idx][~_mask[_idx]] = y[_idx][~_mask[_idx]].mm(R.T) + _translation

        assert len(mols) == len(mols_ref)

        xa = torch.ones_like(batch_mols.x) * self.vocab.padding
        ca = torch.ones_like(batch_mols.c) * self.vocab.padding
        aa = torch.ones_like(batch_mols.a) * self.vocab.padding
        mask = batch_mols.mask


        if self.batch_ot:
            mols_aligned = {}
            cost_matrix = np.zeros((len(mols_ref), len(mols)), dtype=np.float32)
            for i, m_ref in enumerate(mols_ref):
                for j, m in enumerate(mols):
                    m_aligned, cost = m.align(m_ref, max_steps=self.max_steps)
                    cost_matrix[i, j] = cost
                    mols_aligned[(i,j)] = m_aligned
                    
            row_idx, col_idx = linear_sum_assignment(cost_matrix)            
            for i, j in zip(row_idx, col_idx):
                m_aligned = mols_aligned[(i,j)]
                xa[i, :len(m_aligned)] = torch.tensor(m_aligned.x, dtype=xa.dtype)
                ca[i, :len(m_aligned)] = torch.tensor(m_aligned.c, dtype=ca.dtype)
                aa[i, :len(m_aligned), :len(m_aligned)] = torch.tensor(m_aligned.a, dtype=aa.dtype)
 
        else:
            for i, (m_ref, m) in enumerate(zip(mols_ref, mols)):
                m_aligned, cost = m.align(m_ref, max_steps=self.max_steps)
                xa[i, :len(m_aligned)] = torch.tensor(m_aligned.x, dtype=xa.dtype)
                ca[i, :len(m_aligned)] = torch.tensor(m_aligned.c, dtype=ca.dtype)
                aa[i, :len(m_aligned), :len(m_aligned)] = torch.tensor(m_aligned.a, dtype=aa.dtype)
        
        return MoleculeBatch(**{"x": xa, "y": ya, "c": ca, "a": aa, "mask": mask})

    def __call__(self, mols0, mols1, t):
        return self.interpolate(mols0, mols1, t)

    def interpolate(self, mols0, mols1, t):
        """
        Interpolate a batch of molecules from mols0 and mols1. 
        First mols0 is aligned to best match mols1.

        Parameters
        ----------
        mols0 : dict
            A dictionary containing molecular data with the following keys:
            - `x` : torch.tensor 
                A (N, 3) array representing atomic coordinates of the molecules.
            - `y` : torch.tensor 
                A (N, 3) array representing additional atomic coordinates or features.
            - `c` : torch.tensor 
                A (N, 2) array representing atomic types and charges.
            - `a` : torch.tensor 
                A (N, N) array representing adjacency matrix.
            - `mask` : torch.tensor 
                A (N,) array representing mask indicating valid entries in the batch. (0 keep, 1 mask)
        mols1 : dict
            A dictionary containing reference molecular data with the same keys and structure as `batch_mols`.

        t: torch.tensor 
            A (N,) array containing values in [0,1]

        Returns
        -------
        mols0_aligned : dict
            A dictionary containing the transformed version of `mols0`, with the same keys {x, y, c, a, mask}, aligned to match `mols1`.
        molst : dict
            A dictionary containing the interpolated version of `mols0_aligned` and `mols1`, with the same keys {x, y, c, a, mask}.
        """

        mols0_aligned = self.match(mols0, mols1)
        t = t.ravel()
        tx = t.view((-1,) + (1,) * (len(mols0_aligned.x.shape) - 1))
        tc = t.view((-1,) + (1,) * (len(mols0_aligned.c.shape) - 1))
        ta = t.view((-1,) + (1,) * (len(mols0_aligned.a.shape) - 1))
                    
        xt = mols0_aligned.x * (1-tx) + mols1.x * tx
        xt = xt + torch.randn_like(xt) * self.coord_noise_std

        yt = mols0_aligned.y * (1-tx) + mols1.y * tx
        yt = yt + torch.randn_like(yt) * self.coord_noise_std

        cat_mask = torch.rand_like(mols1.c.float()) > tc
        ct = mols1.c.detach().clone()
        ct[cat_mask] = mols0_aligned.c[cat_mask]

        cat_mask = torch.rand_like(mols1.a.float()) > ta
        at = mols1.a.detach().clone()
        at[cat_mask] = mols0_aligned.a[cat_mask]

        mask = mols1.mask
      
        pair_mask = torch.logical_or(mask[...,None], mask[:,None])
        xt[mask] = self.vocab.padding
        yt[mask] = self.vocab.padding
        ct[mask] = self.vocab.padding
        at[pair_mask] = self.vocab.padding
        molst = MoleculeBatch(**{"x": xt, "y": yt, "c": ct, "a": at, "mask": mask})
        return mols0_aligned, molst
    
    
class TimeSampler:
    def __init__(self,time_alpha: float = 1.0, time_beta: float = 1.0, fixed_time: Optional[float] = None):
        if fixed_time is not None and (fixed_time < 0 or fixed_time > 1):
            raise ValueError(f"fixed_time must be between 0 and 1 if provided.")
        if fixed_time is None:
            self.time_dist = torch.distributions.Beta(time_alpha, time_beta)
        self.fixed_time = fixed_time
    
    def sample_time(self, batch_size: int = 1) -> torch.tensor:
        if self.fixed_time:
            times = torch.tensor([self.fixed_time] * batch_size)
        else:
            times = self.time_dist.sample((batch_size,))
        return times