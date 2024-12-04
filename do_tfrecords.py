import os
import argparse

from munch import Munch

from data_utils import serialize_from_orca


PARSER = argparse.ArgumentParser(description="do_tfrecords")

PARSER.add_argument('--record_name',
                    type=str,
                    default='new_record',
                    help="""Data ID""")

PARSER.add_argument('--data_csv',
                    type=str,
                    required=True,
                    help="""csv-file path with compounds to be serialized""")

PARSER.add_argument('--orca_outs',
                    type=str,
                    required=True,
                    help="""Path to a directory with calculated data""")

PARSER.add_argument('--overlap_thresh',
                    type=float,
                    default=0.035,
                    help="""The number of rotated structures""")

PARSER.add_argument('--save_path',
                    type=str,
                    default='converted_data',
                    help="""Directory to save the tfrecords""")

PARSER.add_argument('--gepol_path',
                    type=str,
                    default='',
                    help="""Path to a directory with GEPOL binary""")

PARSER.add_argument('--schema_path',
                    type=str,
                    required=True,
                    help="""Path to the graph schema template""")

PARSER.add_argument('--scalings_csv',
                    type=str,
                    default='',
                    help="""Path to a csv file with targets scaling factors""")

PARSER.add_argument('--rot_aug',
                    type=int,
                    default=1,
                    help="""The number of rotated structures""")

PARSER.add_argument('--multi_target',
                    dest='multi_target',
                    action='store_true',
                    help="""Write multiple targets""")

PARSER.add_argument('--monolith_record', '--monolith',
                    dest='monolith',
                    action='store_true',
                    help="""Save everyting in a single tfrecord""")

PARSER.add_argument('--aux_orca_outs',
                    type=str,
                    default=None,
                    help="""Additional calculated data""")

PARSER.add_argument('--aux_rot_aug',
                    type=int,
                    default=5,
                    help="""Rotated structures in additional data""")

PARSER.add_argument('--aux_overlap_thresh',
                    type=float,
                    default=0.19,
                    help="""The number of rotated structures""")

PARSER.add_argument('--aux_overlap_share',
                    type=float,
                    default=0.1,
                    help="""The number of rotated structures""")


def parse_args(flags):
    return Munch({
        'record_name': flags.record_name,
        'data_csv': flags.data_csv,
        'orca_outs': flags.orca_outs,
        'overlap_thresh': flags.overlap_thresh,
        'save_path': flags.save_path,
        'gepol_path': flags.gepol_path,
        'schema_path': flags.schema_path,
        'scalings_csv': flags.scalings_csv,
        'rot_aug': flags.rot_aug,
        'multi_target': flags.multi_target,
        'monolith': flags.monolith,
        'aux_orca_outs': flags.aux_orca_outs,
        'aux_rot_aug': flags.aux_rot_aug,
        'aux_overlap_thresh': flags.aux_overlap_thresh,
        'aux_overlap_share': flags.aux_overlap_share,
    })


def setup(params=None):
    if not params:
        params = parse_args(PARSER.parse_args())
    os.makedirs(params.save_path, exist_ok=True)
    return params


def main():
    params = setup()
    orca_outs = [params.orca_outs]
    rot_augs = [params.rot_aug]
    if params.aux_orca_outs:
        orca_outs += [params.aux_orca_outs]
        rot_augs += [params.aux_rot_aug]

    serialize_from_orca(csv_path=params.data_csv,
                        orca_out_paths=orca_outs,
                        overlap_thresh=params.overlap_thresh,
                        save_path=params.save_path,
                        record_name=params.record_name,
                        schema_template_path=params.schema_path,
                        scalings_csv_path=params.scalings_csv,
                        monolith=params.monolith,
                        multi_target=params.multi_target,
                        rot_augs=rot_augs,
                        gepol_path=params.gepol_path,
                        aux_overlap_thresh=params.aux_overlap_thresh,
                        aux_overlap_share=params.aux_overlap_share)


if __name__ == '__main__':
    main()
