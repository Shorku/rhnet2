import os
import argparse

import pandas as pd

from munch import Munch

from qchem_data_utils import orca_out_not_ok


PARSER = argparse.ArgumentParser(description="check_orca_logs")

PARSER.add_argument('--data_csv',
                    type=str,
                    required=True,
                    help="""csv-file path with compounds to be serialized""")

PARSER.add_argument('--orca_outs',
                    type=str,
                    required=True,
                    help="""Path to a directory with calculated data""")

PARSER.add_argument('--rot_aug',
                    type=int,
                    default=1,
                    help="""The number of rotated structures""")


def parse_args(flags):
    return Munch({
        'data_csv': flags.data_csv,
        'orca_outs': flags.orca_outs,
        'rot_aug': flags.rot_aug,
    })


def main():
    params = parse_args(PARSER.parse_args())
    data_df = pd.read_csv(params.data_csv)
    for idx, isomer in data_df.iterrows():
        isomer_id = isomer['isomer_id']
        nconf = isomer['nconf']
        for conf in range(1, nconf + 1):
            base = f'{isomer_id}_{conf}'
            for rot_aug in range(1, params.rot_aug + 1):
                fname = f'{base}_{rot_aug}_dft.zip'
                if orca_out_not_ok(os.path.join(params.orca_outs, fname)):
                    with open(os.path.join(params.orca_outs,
                                           'sanity_check.log'), 'a') as f:
                        f.write(f'{fname}\n')
                else:
                    fname = f'{base}_{rot_aug}.zip'
                    if orca_out_not_ok(os.path.join(params.orca_outs, fname)):
                        with open(os.path.join(params.orca_outs,
                                               'sanity_check.log'), 'a') as f:
                            f.write(f'{fname}\n')


if __name__ == '__main__':
    main()
