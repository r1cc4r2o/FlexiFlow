from pathlib import Path
from typing import List
import csv
import os


from rdkit import Chem
from rdkit.Chem.QED import qed
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import numpy as np
import h5py
import torch


def extract_pocket(protein: Chem.Mol,
                   ligand: Chem.Mol,
                   cutoff: float = 3.5, 
                   min_residue_size: float = 10 ) -> List[np.ndarray]:
    # https://www.biorxiv.org/content/10.1101/2024.02.25.581968v4.full.pdf
    pX = protein.GetConformer().GetPositions()
    lX = ligand.GetConformer().GetPositions()
    d = np.sum((pX[:,None] - lX[None])**2, -1).min(-1)**0.5
    pocket = set(np.where(d < cutoff)[0].tolist())
    
    residue_sizes = {}
    for atom in protein.GetAtoms():
        res_info = atom.GetPDBResidueInfo()
        if res_info.GetResidueNumber() not in residue_sizes:
            residue_sizes[res_info.GetResidueNumber()] = 0
        residue_sizes[res_info.GetResidueNumber()] += 1
    
    to_delete = set()
    for atom_id in pocket:
        atom = protein.GetAtomWithIdx(int(atom_id))
        res_info = atom.GetPDBResidueInfo()
        if residue_sizes[res_info.GetResidueNumber()] < min_residue_size:
            to_delete.add(atom_id)
    for atom_id in to_delete:
        pocket.remove(atom_id)
    
    residues_ids = set()
    for atom_id in pocket:
        atom = protein.GetAtomWithIdx(int(atom_id))
        res_info = atom.GetPDBResidueInfo()
        residues_ids.add(res_info.GetResidueNumber())
        
    for atom in protein.GetAtoms():
        res_info = atom.GetPDBResidueInfo()
        if res_info.GetResidueNumber() in residues_ids:
            pocket.add(atom.GetIdx())
    
    conf = protein.GetConformer()
    pocket_x = np.zeros((len(pocket), 3), dtype=np.float32)
    pocket_c = np.zeros((len(pocket), 2), dtype=np.int64)
    for i,atom_id in enumerate(sorted(list(pocket))):
        pos = conf.GetAtomPosition(atom_id)
        pocket_x[i] = [pos.x, pos.y, pos.z]
        pocket_c[i, 0] = protein.GetAtomWithIdx(atom_id).GetAtomicNum()
        pocket_c[i, 1] = protein.GetAtomWithIdx(atom_id).GetFormalCharge()
        
    return pocket_x, pocket_c


def ligand2npy(ligand: Chem.Mol) -> List[np.ndarray]:
    conf = ligand.GetConformer()
    n_atoms = ligand.GetNumAtoms()
    ligand_x = np.zeros((n_atoms, 3), dtype=np.float32)
    ligand_c = np.zeros((n_atoms, 2), dtype=np.int8)
    ligand_a = np.zeros((n_atoms, n_atoms), dtype=np.uint8)
    for atom in ligand.GetAtoms():
        atom_id = atom.GetIdx()
        pos = conf.GetAtomPosition(atom_id)
        ligand_x[atom_id] = [pos.x, pos.y, pos.z]
        ligand_c[atom_id, 0] = atom.GetAtomicNum()
        ligand_c[atom_id, 1] = atom.GetFormalCharge()

    bond_type_dict = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4
    }

    for bond in ligand.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        ligand_a[atom1, atom2] = bond_type_dict.get(bond_type, 0)
        ligand_a[atom2, atom1] = ligand_a[atom1, atom2]
    return ligand_x, ligand_c, ligand_a


def process_datasets(file_paths: List[str], output_files: List[str], qed_thres : List[float], test_ids_lsts: List[List[str]] = [[]]) -> None:
    assert len(file_paths) == len(output_files)
    for f, output_file, test_ids, qed_thre, remove_hs in zip(file_paths, output_files, test_ids_lsts, qed_thres):
        dataset = {}
        if 'test' in output_file:
            mode = 'test'
        else:
            mode = 'train'
        print(f'processing {f} in {mode} mode')
        skipped_counter = 0
        for protein_path in Path(f).glob("*/protein.pdb"):
            protein_id = protein_path.parent.name
            ligand_path = protein_path.parent / "ligand.sdf"
            score_path = protein_path.parent / "score.csv"
            grid_path = protein_path.parent / "grid.zip"
            if mode == 'test':
                if test_ids:
                    if protein_id not in test_ids:
                        continue
                    if not grid_path.exists():
                        continue
                else:
                    raise ValueError(f'Data preparation in test mode, but no test ids provided')
            if (not ligand_path.exists()) or (not score_path.exists()):
                continue
            score = []
            protein = None
            ligand = None
            try:
                with open(score_path) as file:
                    reader = csv.DictReader(file)
                    score = [row for row in reader]
                mols = []
                with Chem.SDMolSupplier(
                    ligand_path,
                    removeHs=False, 
                    sanitize=False
                ) as suppl:
                    for mol in suppl:
                        if mol is None:
                            continue
                        mols.append(mol)
                protein = Chem.MolFromPDBFile(
                    str(protein_path), 
                    removeHs=False, 
                    sanitize=False)
            except Exception as err:
                print(err)
                continue

            if len(score) != 1:
                continue
            if len(mols) != 1:
                continue
            ligand = mols[0]
            if protein is None:
                print(f'{protein_path} protein is none')
                continue
            if ligand is None:
                print(f'{ligand_path} ligand is none')
                continue

            qed_score = qed(ligand)
            docking_score = float(score[0]["r_i_docking_score"])
            if qed_score < qed_thre:
                skipped_counter += 1
                continue
            if mode == 'train':
                if test_ids:
                    if protein_id in test_ids:
                        continue
                else:
                    if (docking_score > 0):
                        print(f'skipping {protein_id} with docking score : {docking_score}')
                        continue

            pocket_x, pocket_c = extract_pocket(protein, ligand)
            ligand_x, liganc_c, ligand_a = ligand2npy(ligand)
            dataset[protein_id] = {
                "protein": [pocket_x, pocket_c],
                "ligand": [ligand_x, liganc_c, ligand_a],
                "qed": qed_score,
            }
        np.save(output_file, dataset)
        print(f'skipped_counter (qed): {skipped_counter}')


if __name__ == "__main__":
    cache = Path("cache")
    cache.mkdir(parents=True, exist_ok=True)
    
    datasets_paths = ['/Work/Datasets/PDBBind']
    test_ids_paths = ['/Work/Datasets/test']
    output_files = [cache / "pdbbind_test_hs.npy"]

    _datasets_paths = []
    _output_files = []
    _test_ids_lsts = []
    _qed_thresholds = []
    for d,o,t in zip(datasets_paths, output_files, test_ids_paths):
        _datasets_paths.append(str(d))
        _output_files.append(str(o))
        if t:
            with open(t, 'r') as f:
                ids = [id.strip() for id in f.readlines()]
                _test_ids_lsts.append(ids)
                _qed_thresholds.append(0)
        else:
            _test_ids_lsts.append([])
            _qed_thresholds.append(0.5)
    print(f'_datasets_paths: {_datasets_paths}')
    print(f'_output_files: {_output_files}')
    print(f'_test_ids_lsts: {_test_ids_lsts}')
    print(f'_qed_thresholds:{_qed_thresholds}')
                
    if len(_output_files) > 0:
        process_datasets(_datasets_paths, _output_files, _qed_thresholds, _test_ids_lsts)

    from rdkit import Chem
    pt = Chem.GetPeriodicTable()

    for o in output_files:
        data = np.load(o, allow_pickle=True)[()]
        p_sizes = []
        x = np.zeros(1000, dtype=np.int64)
        for k, v in data.items():
            protein_atoms = np.bincount(v["protein"][1][...,0])
            print(v["protein"][1][...,0])
            print(f'protein_atoms: {protein_atoms}')
            x[:len(protein_atoms)] += protein_atoms
            p_sizes.append(len(v["protein"][0]))
       
        for a in np.argsort(x)[::-1]:
            if x[a] == 0:
                break
            print(a, pt.GetElementSymbol(int(a)), x[a])
        p_sizes = np.bincount(np.array(p_sizes))
        sizes = np.arange(len(p_sizes))
        c_sizes = np.cumsum(p_sizes / p_sizes.sum())
        li = sizes[c_sizes >= 5./100.][0]
        ri = sizes[c_sizes <= 95./100.][-1]
        print(li, ri)
        print()