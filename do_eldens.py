import os
import argparse
import concurrent.futures

import pandas as pd

from munch import Munch

from qchem_data_utils import split_to_atoms, rotate_geom
from qchem_run_utils import orca_runner, orca_cleanup


PARSER = argparse.ArgumentParser(description="do_eldens")

PARSER.add_argument('--data_csv',
                    type=str,
                    required=True,
                    help="""Path to csv-file with compounds to be computed""")

PARSER.add_argument('--work_dir',
                    type=str,
                    default='calculated_data',
                    help="""Path to working directory with calculated data""")

PARSER.add_argument('--conf_dir',
                    type=str,
                    required=True,
                    help="""Path to a directory with optimized geometries""")

PARSER.add_argument('--orca_path',
                    type=str,
                    required=True,
                    help="""ORCA location""")

PARSER.add_argument('--basis_path',
                    type=str,
                    default='basis_sets/ANOR0_Ar',
                    help="""Basis set path""")

PARSER.add_argument('--dft_inp_path',
                    type=str,
                    default='orca_inputs/pbe_def2svp_orca29',
                    help="""DFT input path""")

PARSER.add_argument('--dmt_inp_path',
                    type=str,
                    default='orca_inputs/noiter_moread',
                    help="""Basis set projection input path""")

PARSER.add_argument('--rot_aug',
                    type=int,
                    default=1,
                    help="""The number of rotated structures""")

PARSER.add_argument('--pal',
                    type=int,
                    default=1,
                    help="""The number of CPU cores to use""")


def parse_args(flags):
    return Munch({
        'data_csv': flags.data_csv,
        'work_dir': flags.work_dir,
        'conf_dir': flags.conf_dir,
        'orca_path': flags.orca_path,
        'basis_path': flags.basis_path,
        'dft_inp_path': flags.dft_inp_path,
        'dmt_inp_path': flags.dmt_inp_path,
        'rot_aug': flags.rot_aug,
        'pal': flags.pal
    })


def setup(params=None):
    if not params:
        params = parse_args(PARSER.parse_args())
    os.makedirs(params.work_dir, exist_ok=True)
    return params


def single_single_point(base: str, aug_id: int, atoms: list, coords: list,
                        dft_template: str, dmt_template: str, basis_block: str,
                        params: Munch):
    aug_geom = rotate_geom(atoms, coords)
    aug_base = f'{base}_{aug_id}'
    dft_inp = dft_template.format(aug_geom)
    dmt_inp = dmt_template.format(f'{aug_base}_dft', basis_block, aug_geom)
    with open(os.path.join(params.work_dir, f'{aug_base}_dft.inp'), 'w') as f:
        f.write(dft_inp)
    with open(os.path.join(params.work_dir, f'{aug_base}.inp'), 'w') as f:
        f.write(dmt_inp)
    orca_runner([f'{aug_base}_dft'], params.work_dir, params.orca_path)
    orca_runner([f'{aug_base}'], params.work_dir, params.orca_path)
    orca_cleanup([f'{aug_base}_dft'], [f'{aug_base}'], params.work_dir)


def single_points(params: Munch):
    with open(params.dft_inp_path) as f:
        dft_template = f.read()
    with open(params.dmt_inp_path) as f:
        dmt_template = f.read()
    with open(params.basis_path) as f:
        basis = f.read()
    data_df = pd.read_csv(params.data_csv)
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=params.pal) as executor:
        futures = []
        for idx, isomer in data_df.iterrows():
            isomer_id = isomer['isomer_id']
            nconf = isomer['nconf']
            for conf in range(1, nconf + 1):
                base = f'{isomer_id}_{conf}'
                conf_file = os.path.join(params.conf_dir,
                                         f'{base}.xyz')
                with open(conf_file) as f:
                    conf_geom = ''.join(f.readlines()[2:]).strip()
                conf_atoms, conf_coords = split_to_atoms(conf_geom)
                conf_basis = ''.join([f'NewGTO {element}\n{basis}end\n'
                                      for element in set(conf_atoms)]).strip()
                for rot_aug in range(1, params.rot_aug + 1):
                    futures.append(executor.submit(single_single_point,
                                                   base,
                                                   rot_aug,
                                                   conf_atoms,
                                                   conf_coords,
                                                   dft_template,
                                                   dmt_template,
                                                   conf_basis,
                                                   params))
        for future in concurrent.futures.as_completed(futures):
            future.result()


def main():
    params = setup()
    single_points(params)


if __name__ == '__main__':
    main()
