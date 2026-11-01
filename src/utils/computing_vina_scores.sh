#!/bin/bash

protein=$1
ligands=$2
output_path=$3
vina_prefix="autodock_vina_1_1_2_linux_x86/bin"
obabel "${protein}"  -O "${output_path}/protein.pdbqt"  -xr  -p 7.4 --partialcharge eem
obabel -isdf "${ligands}" -opdbqt -h -O "${output_path}/mols.pdbqt"

mkdir -p "${output_path}/mols_vina"

${vina_prefix}/vina_split --input "${output_path}/mols.pdbqt" --ligand "${output_path}/mols_vina/ligand"
for ligand in ${output_path}/mols_vina/*.pdbqt; do
  scorefilename="${ligand%.*}"_score.txt
  rm -f $scorefilename
  ${vina_prefix}/vina --receptor $output_path/protein.pdbqt --ligand $ligand --score_only | grep "Affinity" > $scorefilename
done
