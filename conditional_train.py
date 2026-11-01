import numpy as np
import torch
from rdkit import Chem
import os
from datetime import datetime
from pathlib import Path
from natsort import natsorted

from src.data.interpolate import NoiseSampler, GeometricInterpolant, TimeSampler
from src.data.molecule import Molecule
from src.data.molecule import simple_get_dataloader as get_dataloader
from src.data.protein_molecule import ProteinLigand
from src.data.protein_molecule import get_dataloader as get_dataloader_pl

from src.data.vocabularyp import Vocabulary

from src.models.flexiflow.ff import FlexiFlow
from src.train.checkpoint import CheckpointManager
from src.utils.prepare_protein_ligand_dataset import ligand2npy


from func_timeout import func_timeout
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")



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



class DefaultParams:
    def __init__(self):
        self.n_layers = 11
        self.d_model = 384
        self.d_message = 128
        self.d_edge = 128
        self.n_coord_sets = 64
        self.n_attn_heads = 32
        self.d_message_hidden = 128
        self.coord_norm = "length"
        self.size_emb = 64
        self.max_atoms = 256
        self.atom_vocab_size = 13
        self.charge_vocab_size = 4
        self.edge_vocab_size = 5
        self.self_conditioning = False
        self.remove_com = False
        self.protein_conditioning = True
        self.max_patoms = 501
        self.patom_vocab_size = 6
        self.gramian = False

    @property
    def kwargs(self):
        return self.__dict__



def cloud2mol(x, c, a, y=None):
    mol = AllChem.RWMol()
    for atom_num, charge in c:
        atom = Chem.Atom(int(atom_num))
        atom.SetFormalCharge(int(charge))
        mol.AddAtom(atom)
        
    frontier = set()
    for i, j in np.vstack(np.where(a)).T:
        if i == j:
            continue
        if (i,j) in frontier:
            continue
        if (j,i) in frontier:
            continue
        if a[i,j] == 1:
            bt = Chem.BondType.SINGLE
        elif a[i,j] == 2:
            bt = Chem.BondType.DOUBLE
        elif a[i,j] == 3:
            bt = Chem.BondType.TRIPLE
        elif a[i,j] == 4:
            bt = Chem.BondType.AROMATIC
        else:
            continue
            
        mol.AddBond(int(i),int(j),order=bt)
        frontier.add((i,j))
        frontier.add((j,i))

    conformer2add = [x] if y is None else [x, y]

    for c_xy in conformer2add:
        conf = Chem.Conformer()
        for i, p in enumerate(c_xy):
            point = Point3D(float(p[0]), float(p[1]), float(p[2]))
            conf.SetAtomPosition(i, point)
        mol.AddConformer(conf)

    mol = Chem.Mol(mol)
    return mol



def prepare_input(batch, t, vocab, generate_random_fragments=False):
    frag_mask = torch.zeros_like(batch["mask"])
    if generate_random_fragments:
        np_mols = vocab.decode(batch)
        for i, (pos,c,adj,mask) in enumerate(zip(np_mols["x"], np_mols["c"], np_mols["a"], np_mols["mask"])):
            if np.random.rand() > 0.5:
                try:
                    mol = cloud2mol(pos[~mask], c[~mask], adj[:,~mask][~mask,:])
                    mol.UpdatePropertyCache()
                    Chem.GetSSSR(mol)
                except BaseException:
                    pass
    
    h = torch.nn.functional.one_hot(batch["c"][...,0], num_classes=len(vocab.atom_types)+1)
    h2 = torch.nn.functional.one_hot(batch["c"][...,1], num_classes=len(vocab.charge_types)+1)
    e = torch.nn.functional.one_hot(batch["a"], num_classes=len(vocab.bond_types))
    t = t.view(-1, 1, 1).expand(-1, h.size(1), -1)
    features = torch.cat((t, h.float()), dim=2)
    return batch["x"], batch["y"], features, h2, e, batch["mask"]#, frag_mask


def prepare_protein_input(batch, vocab):
    features = torch.nn.functional.one_hot(batch["c"][...,0], num_classes=len(vocab.atom_types)+1)
    h2 = torch.nn.functional.one_hot(batch["c"][...,1], num_classes=len(vocab.charge_types)+1)
    return batch["x"], features, h2, None, batch["mask"]


def nll_loss(y_hat, y, mask=None, ignore_index=-100, logits=True):
    y_hat_shape = y_hat.shape
    y_shape = y.shape
    y = y.ravel()
    y_hat = y_hat.view(-1, y_hat_shape[-1])
    if logits:
        y_hat_log = torch.nn.functional.log_softmax(y_hat, -1)
    else:
        y_hat_log = y_hat
    
    _nlls = torch.nn.functional.nll_loss(y_hat_log, y, ignore_index=ignore_index, reduction="none")
    if mask is None:
        _nlls = _nlls.view(y_hat_shape[:-1])
        _nlls = _nlls.mean(dim=tuple(range(1, len(_nlls.shape))))
        return _nlls

    _nlls = _nlls * (~mask).view(-1).float()
    _nlls = _nlls.view(y_hat_shape[:-1])
    _nlls = _nlls.sum(dim=tuple(range(1, len(_nlls.shape))))
    _den = (~mask).sum(dim=tuple(range(1, len(mask.shape))))
    _den = torch.clip(_den.float(), 1.)
    return _nlls / _den


def remove_hs(batch, err_counter):
    batch_pos = batch["x"].cpu().numpy()
    batch_c= batch["c"].cpu().numpy()
    batch_adj = batch["a"].cpu().numpy()
    batch_mask = batch["mask"].cpu().numpy()
    batch_id = batch["pdb_id"]
    
    batch_pos_nohs = []
    batch_c_nohs = []
    batch_adj_nohs = []
    for pos, c, adj, mask, pid in zip(batch_pos, batch_c, batch_adj, batch_mask, batch_id):
        mol = cloud2mol(pos[~mask], c[~mask], adj[:,~mask][~mask,:])
        try:
            mol_no_hs = Chem.RemoveHs(mol)
            x_nohs, c_nohs, a_nohs = ligand2npy(mol_no_hs)
            batch_pos_nohs.append(torch.from_numpy(x_nohs))
            batch_c_nohs.append(torch.from_numpy(c_nohs))
            batch_adj_nohs.append(torch.from_numpy(a_nohs))
        except Exception as err:
            err_counter += 1
            print(err)
            print(f"error counter for {pid}: {err_counter}")
        
    xb = torch.nn.utils.rnn.pad_sequence(batch_pos_nohs, batch_first=True, padding_value=0.0).float()
    cb = torch.nn.utils.rnn.pad_sequence(batch_c_nohs, batch_first=True, padding_value=0).long()
    ab = torch.zeros((xb.shape[0], xb.shape[1], xb.shape[1]), device=xb.device).long()
    mask = torch.zeros((xb.shape[0], xb.shape[1]), dtype=torch.bool, device=xb.device)
    for i, _a in enumerate(batch_adj_nohs):
        ab[i, :_a.shape[0], :_a.shape[0]] = _a
        mask[i, _a.shape[0]:] = True
    return {"x":xb, "c": cb, "a": ab, "mask": mask}, err_counter


class Trainer:
    def __init__(self, model, optimizer, scheduler, vocabulary, protein_vocabulary=None, remove_hs=False, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.vocabulary = vocabulary
        self.protein_vocabulary = protein_vocabulary
        self.remove_hs = remove_hs
        self.interporlator = GeometricInterpolant(self.vocabulary, coord_noise_std=0.2, batch_ot=False, max_steps=1)
        self.time_sampler = TimeSampler(time_alpha=2.0, time_beta=1.0)
        scale_ot_factor = 0.25645364536453646
        self.noise_sampler = NoiseSampler(self.vocabulary, scale_ot=True, scale_ot_factor=scale_ot_factor, sample_charges=False, symmetric_adj=False)
        avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
        self.ema_model = torch.optim.swa_utils.AveragedModel(self.model, multi_avg_fn=avg_fn)
        self.max_gradient_norm = 1.0
        self.device = device

    def _remove_com(self, batch, com=None, com_y=None):
        if com is None:
            num = (batch["x"] * (~batch["mask"]).float().unsqueeze(-1)).sum(1)
            if "y" in batch and batch["y"] is not None:
                num_y = (batch["y"] * (~batch["mask"]).float().unsqueeze(-1)).sum(1)
            den = (~batch["mask"]).sum(1,keepdims=True).float()
            den = torch.clip(den, 1.)
            com = (num / den).unsqueeze(1)
            if "y" in batch and batch["y"] is not None:
                com_y = (num_y / den).unsqueeze(1)
        batch["x"] = (batch["x"] - com) * (~batch["mask"]).float().unsqueeze(-1)
        if "y" in batch and batch["y"] is not None:
            batch["y"] = (batch["y"] - com_y) * (~batch["mask"]).float().unsqueeze(-1)
            return batch, com, com_y
        return batch, com


    def training_step(self, batch, batch_protein=None):
        self.optimizer.zero_grad()
        if self.remove_hs:
            batch, err_counter = remove_hs(batch, err_counter)

        if "y" not in batch or batch["y"] is None:
            batch["y"] = torch.clone(batch["x"])

        try:
            bs, num_atoms = batch["x"].shape[0:2]
            enc_noise = self.noise_sampler.sample_batch([num_atoms]*bs)
            enc_batch_protein = None
            enc_batch = self.vocabulary.encode(batch)
            
            if batch_protein is not None:
                enc_batch_protein = self.protein_vocabulary.encode(batch_protein)
        except:
            return None, None, None, None, None, None, None, None

        if enc_batch_protein is not None:
            enc_batch_protein, com = self._remove_com(enc_batch_protein)
            enc_batch, com, com_y = self._remove_com(enc_batch, com=com, com_y=com)
            x_prot, h_prot, _, _, mask_prot = prepare_protein_input(enc_batch_protein, self.protein_vocabulary)
            x_prot = x_prot.to(self.device)
            h_prot = h_prot.to(self.device)
            mask_prot = mask_prot.to(self.device)
        else:
            enc_batch, com, com_y = self._remove_com(enc_batch)
            x_prot = None
            h_prot = None
            mask_prot = None

        t = self.time_sampler.sample_time(len(enc_batch["x"]))
        enc_noise_align, enc_t = self.interporlator.interpolate(enc_noise, enc_batch, t) 
        x_t, y_t, h_t, _, e_t, mask_t = prepare_input(enc_t, t, self.vocabulary)
        x1, y1, h1, c1, e1, mask1 = prepare_input(enc_batch, t, self.vocabulary) # frag_mask placeholder frag_mask

        x1 = x1.to(self.device)
        y1 = y1.to(self.device)
        h1 = h1.to(self.device)
        e1 = e1.to(self.device)
        c1 = c1.to(self.device)
        mask1 = mask1.to(self.device)
        
        x_t = x_t.to(self.device)
        y_t = y_t.to(self.device)
        h_t = h_t.to(self.device)
        e_t = e_t.to(self.device)
        mask_t = mask_t.to(self.device)
        x_p, y_p, h_p, hy_p, c_p, cy_p, e_p, ey_p = self.model(x=x_t, y=y_t, h=h_t, e=e_t, mask=mask_t, px=x_prot, ph=h_prot, pmask=mask_prot) # , frag_mask=frag_mask

        pair_mask = torch.logical_or(mask1[...,None], mask1[:,None])
        coord_loss_x = (x_p - x1)**2 * (~mask1).unsqueeze(-1) 
        coord_loss_y = (y_p - y1)**2 * (~mask1).unsqueeze(-1)
        coord_loss_x = coord_loss_x.mean(dim=(1, 2)) 
        coord_loss_y = coord_loss_y.mean(dim=(1, 2))
        
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

        coord_loss_x = coord_loss_x.mean()
        coord_loss_y = coord_loss_y.mean()
        type_loss = type_loss.mean()
        bond_loss = bond_loss.mean()
        charge_loss = charge_loss.mean()
        regularization_loss_bonds = regularization_loss_bonds.mean() + bond_loss_reg.mean() * 0.5
        regularization_loss_types = regularization_loss_types.mean() + type_loss_reg.mean()
        regularization_loss_charges = regularization_loss_charges.mean() + charge_loss_reg.mean()

        loss = (coord_loss_x * 1.0 +
                coord_loss_y * 1.3 +
                type_loss * 0.2 + 
                bond_loss * 1.0 + 
                charge_loss * 1.0 +
                regularization_loss_bonds * 1.0 + 
                regularization_loss_types * 1.0 +
                regularization_loss_charges * 1.0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_gradient_norm)
        self.optimizer.step()
        self.ema_model.update_parameters(self.model)
        self.scheduler.step()

        return coord_loss_x.item(), coord_loss_y.item(), type_loss.item(), bond_loss.item(), charge_loss.item(), regularization_loss_bonds.item(), regularization_loss_types.item(), regularization_loss_charges.item()

if __name__ == "__main__":
    """ 
    in FM 0 means noise, 1 means real
    """
    dataset_path = "/datasets/geom/train"
    # pretrained_checkpoint = None
    pretrained_checkpoint = False 
    import sys
    task = "pdbbind"
    if task == "pdbbind":
        protein_ligand_dataset_path = ["cache/pdbbind_train.npy"]
    else:
        raise ValueError(f"{task} not supported")
    epochs = 300
    # might change with data loader
    resume = False
    start_epoch = 0
    start_batch = 0
    batch_size = 2
    seed = 1234
    DEVICE = "cuda"
    warm_up_steps = 10000
    # might change with data loader
    REMOVE_HS = False
    err_counter = 0
    if REMOVE_HS:
        chpt_path = f'../checkpoints_protein_{task}_nohs'
    else:
        chpt_path = f'../checkpoints_protein_{task}_hs'
    print(chpt_path) 
    params = DefaultParams()

    identifier = datetime.now().strftime("%Y %m %d %H").replace(" ", "_")
    logger = Logger(f"train", log_dir=Path(chpt_path) / f"{identifier}_logs")
    logger_debug = Logger(f"debug", log_dir=Path(chpt_path) / f"{identifier}_logs")

    _tmp_paths = natsorted([ f for f in Path(chpt_path).glob("chpt_*.pt")])[::-1]
    for _p_ in _tmp_paths:
        logger_debug.info(_p_)

    ligand_dataset = Molecule(str(dataset_path), lazy=False, paired_data_xy=np.load("cache/dataloader_fixed_targetssets_geomdrugs.npy"))
    protein_ligand_dataset = ProteinLigand(
            protein_ligand_dataset_path,
            min_qed=0.0,
            min_pocket_size=120,
            max_pocket_size=params.max_patoms-1)
    
    
    vocab = Vocabulary()
    protein_vocabulary = Vocabulary({"H", "C", "N", "O", "S"})
            
    # Load model
    model, optimizer, scheduler, ema_model, checkpoint_manager = CheckpointManager.restore_last_checkpoint(chpt_path)
    batch_id = None
    if model is None:
        model = FlexiFlow(**params.kwargs)
        if pretrained_checkpoint:
            model = FlexiFlow.load_from_file(pretrained_checkpoint, device=DEVICE, instance=model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, amsgrad=True, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=warm_up_steps)
        checkpoint_manager = CheckpointManager(vocab, model, optimizer, str(chpt_path), scheduler=scheduler, protein_vocabulary=protein_vocabulary, seed=seed, device=DEVICE)
    else:
        start_batch = checkpoint_manager.batch
        start_epoch = checkpoint_manager.epoch
        seed = checkpoint_manager.seed
        DEVICE = checkpoint_manager.device
        print(f"model loaded from {chpt_path}")
        resume = True
    
    model = model.train()
    model = model.to(DEVICE)
    trainer = Trainer(model, optimizer, scheduler, vocabulary=vocab, protein_vocabulary=protein_vocabulary, remove_hs=REMOVE_HS, device=DEVICE)
    if resume:
        trainer.ema_model = ema_model

    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, epochs):

        print(f"Epoch {epoch} started")
        
        dataloader = get_dataloader_pl(protein_ligand_dataset, batch_size, start_epoch=epoch, start_batch=start_batch, seed=seed)
        ligand_dataloader = get_dataloader(ligand_dataset, batch_size, start_epoch=epoch, start_batch=start_batch, seed=seed)
        iterator = iter(ligand_dataloader)

        err_counter = 0
        for batch_id, (batch, batch_protein) in enumerate(dataloader):

            coord_loss_x, coord_loss_y, type_loss, bond_loss, charge_loss, reg_bond_loss, reg_type_loss, reg_charge_loss = trainer.training_step(batch, batch_protein)
            if coord_loss_x is not None:
                if (batch_id + 1) % 10 == 0:
                    log_str = f"({epoch:d}/{batch_id:d})" + \
                            f" losses: coord={coord_loss_x:.2f}" + \
                                f",{coord_loss_y:.2f}" + \
                            f" type={type_loss:.2f}" + \
                                f",{reg_type_loss:.2f}" + \
                            f" bond={bond_loss:.2f}" + \
                                f",{reg_bond_loss:.2f}" + \
                            f" charge={charge_loss:.2f}"+ \
                                f",{reg_charge_loss:.2f}"+ \
                            f" protein"
                    logger.info(log_str)
                    # print(log_str)

            # Safe guard
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(ligand_dataset)
                batch = next(iterator)

            coord_loss_x, coord_loss_y, type_loss, bond_loss, charge_loss, reg_bond_loss, reg_type_loss, reg_charge_loss = trainer.training_step(batch, None)
            if coord_loss_x is not None:
                if (batch_id + 1) % 10 == 0:
                    
                    log_str = f"({epoch:d}/{batch_id:d})" + \
                            f" losses: coord={coord_loss_x:.2f}" + \
                                f",{coord_loss_y:.2f}" + \
                            f" type={type_loss:.2f}" + \
                                f",{reg_type_loss:.2f}" + \
                            f" bond={bond_loss:.2f}" + \
                                f",{reg_bond_loss:.2f}" + \
                            f" charge={charge_loss:.2f}"+ \
                                f",{reg_charge_loss:.2f}"+ \
                            f" ligand"
                    logger.info(log_str)
                    # print(log_str)
            
            if (batch_id) % 100 == 0:
                checkpoint_manager.save_tmp(model, optimizer, scheduler, epoch, batch_id, swa=trainer.ema_model)
                print(f"Checkpoint saved at epoch {epoch} batch {batch_id}")

        if batch_id is not None:
            checkpoint_manager.save(model, optimizer, scheduler, epoch, batch_id, swa=trainer.ema_model)

        start_batch = 0