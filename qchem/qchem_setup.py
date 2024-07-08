import os
import argparse

from munch import Munch

PARSER = argparse.ArgumentParser(description="qchem")

PARSER.add_argument('--data_csv',
                    type=str,
                    required=True,
                    help="""Path to csv-file with compounds to be computed""")

PARSER.add_argument('--work_dir',
                    type=str,
                    default='/calculated_data',
                    help="""Path to working directory with calculated data""")

PARSER.add_argument('--conf_thresh',
                    type=float,
                    default=0.2,
                    help="""Threshold for RDKit conformers embedding""")

PARSER.add_argument('--nconf',
                    type=int,
                    default=50,
                    help="""Max number of conformers RDKit will generate""")

PARSER.add_argument('--opt_thresh',
                    type=float,
                    default=5.0,
                    help="""Energy window to keep conformers within, kJ/mol""")

PARSER.add_argument('--enforce_chirality', dest='enforce_chirality',
                    action='store_true',
                    help="""Enforce RDKit to preserve chirality """)

PARSER.add_argument('--orca_path',
                    type=str,
                    help="""ORCA location""")

PARSER.add_argument('--basis_path',
                    type=str,
                    default='basis_sets/ANOR0_Ar',
                    help="""Basis set path""")

PARSER.add_argument('--dft_inp_path',
                    type=str,
                    default='orca_inputs/pbe_def2tzvp_orca29',
                    help="""DFT input path""")

PARSER.add_argument('--dmt_inp_path',
                    type=str,
                    default='orca_inputs/noiter_moread',
                    help="""Basis set projection input path""")


def parse_args(flags):
    return Munch({
        'data_csv': flags.data_csv,
        'work_dir': flags.work_dir,
        'conf_thresh': flags.conf_thresh,
        'nconf': flags.nconf,
        'opt_thresh': flags.opt_thresh,
        'enforce_chirality': flags.enforce_chirality,
        'orca_path': flags.orca_path,
        'basis_path': flags.basis_path,
        'dft_inp_path': flags.dft_inp_path,
        'dmt_inp_path': flags.dmt_inp_path
    })


def set_env(params):
    if params.orca_path:
        os.environ['ORCA'] = os.path.abspath(params.orca_path)
    elif 'ORCA' not in os.environ:
        raise EnvironmentError('ORCA executables path not specified. '
                               'Use --orca_path.')


def prepare_dir(params):
    os.makedirs(params.work_dir, exist_ok=True)


def setup(params=None):
    if not params:
        params = parse_args(PARSER.parse_args())
    set_env(params)
    prepare_dir(params)

    return params
