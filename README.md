This repository contains code for 
"[Neural Mulliken Analysis: Molecular Graphs from Density Matrices for QSPR on Raw Quantum-Chemical Data](https://doi.org/10.26434/chemrxiv-2024-k2k3l)"
paper.

Prepare quantum chemical data:
```shell
python3 doeldens.py --data_csv data/sl1_train_data.csv --work_dir data/temp --conf_dir data/train_geoms --orca_path $orcadir --rot_aug 50 --pal 10 --dft_inp_path qchem/orca_inputs/pbe_def2svp_orca29 --dmt_inp_path qchem/orca_inputs/noiter_moread_blyp
```

Prepare graphs and save them in tfrecords format:
```shell
python3 do_tfrecords.py --record_name train --data_csv data/sl1_train_data.csv --orca_outs data/temp --overlap_thresh 0.035 --save_path data/tfrecords --gepol_path <gepol_path> --schema_path schema_template --scalings_csv scalings.csv --rot_aug 50 --multi_target
```


Models trained to predict aqueous solubility under Solubility Challenge (2008)
conditions and minimal inference examples are available in a separate 
[repository](https://github.com/Shorku/SolubilityChallenge2008)