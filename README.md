# RhNet2
This repository contains code for 
"[Neural Mulliken Analysis: Molecular Graphs from Density Matrices for QSPR on Raw Quantum-Chemical Data](https://doi.org/10.26434/chemrxiv-2024-k2k3l)"
paper.

<img src="images/intro.jpg" width="801" alt="302">

## Input Data
### Quantum Chemical Data
Prepare quantum chemical data:
```shell
python3 doeldens.py --data_csv data/sl1_train_data.csv --work_dir data/temp --conf_dir data/train_geoms --orca_path $orcadir --rot_aug 50 --pal 10 --dft_inp_path qchem/orca_inputs/pbe_def2svp_orca29 --dmt_inp_path qchem/orca_inputs/noiter_moread --basis_path qchem/basis_sets/ANOR0_Ar 
```

### Graph Data
Prepare graphs and save them in tfrecords format:
```shell
python3 do_tfrecords.py --record_name train --data_csv data/sl1_train_data.csv --orca_outs data/temp --overlap_thresh 0.035 --save_path data/tfrecords --gepol_path <gepol_path> --schema_path schema_template --scalings_csv scalings.csv --rot_aug 50 --multi_target
```

## Inference
Models trained to predict aqueous solubility under Solubility Challenge (2008)
conditions and minimal inference examples are available in a separate 
[repository](https://github.com/Shorku/SolubilityChallenge2008)

Colab Notebook with inference example is available at [Hugging Face](https://huggingface.co/Shorku/RhNet2SC1/blob/main/RhNet2SC1.ipynb)
