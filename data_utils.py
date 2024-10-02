import os
import re
import functools

import pandas as pd
import tensorflow as tf
import tensorflow_gnn as tfgnn

from qchem_data_utils import blocks_from_orca
from qchem_data_utils import dipole_from_orca_out
from qchem_data_utils import surf_from_orca_out
from qchem_data_utils import vol_from_orca_out


###############################################################################
# TF-GNN data manipulation utilities
###############################################################################
def read_schema(schema_path: str):
    schema = tfgnn.read_schema(schema_path)

    return tfgnn.create_graph_spec_from_schema_pb(schema)


def save_schema(nbas: int, schema_template_path: str, schema_save_path: str):
    with open(schema_template_path) as f:
        schema_template = f.read()
    with open(schema_save_path, 'w') as f:
        f.write(schema_template.format(ndiag=nbas * (nbas + 1) // 2,
                                       noffdiag=nbas * nbas))


def context_features(schema_template_path: str):
    with open(schema_template_path) as f:
        schema_template = f.read()
    start = schema_template.find('context')
    for i in range(start, len(schema_template)):
        if schema_template[i] == '{':
            opening = i
            break
    else:
        raise Exception('Invalid schema')
    counter = 0
    for i in range(opening, len(schema_template)):
        if schema_template[i] == '{':
            counter += 1
        if schema_template[i] == '}':
            counter -= 1
        if counter == 0:
            ending = i
            break
    else:
        raise Exception('Invalid schema')
    matches = re.findall(r'features(.*?)key(.*?)[\',\"](.*?)[\',\"]',
                         schema_template[opening:ending + 1],
                         re.DOTALL)

    return [match[-1] for match in matches]


def decode_graph(record_bytes, graph_spec=None, targets=None):
    assert graph_spec is not None, "No graph specification provided"
    graph = tfgnn.parse_single_example(graph_spec, record_bytes, validate=True)
    if targets:
        return graph, {target: graph.context[target] for target in targets}
    else:
        return graph


def get_decode_fn(graph_spec, targets=None):
    return functools.partial(decode_graph,
                             graph_spec=graph_spec,
                             targets=targets)


def balanced_dataset(graph_spec, data_path,
                     shuffle_buffer=0,
                     idc_list=None,
                     compression_type=None):
    decode_fn = get_decode_fn(graph_spec, 'target')
    tfrecord_path = os.path.join(data_path, '{}')
    if idc_list:
        idc_list = [f'{tfrecord}.tfrecord' for tfrecord in idc_list
                    if f'{tfrecord}.tfrecord' in list(os.listdir(data_path))]
    else:
        idc_list = [fname for fname in os.listdir(data_path)
                    if fname.endswith('.tfrecord')]
    weights = [1 / len(idc_list) for _ in idc_list]

    dataset = tf.data.Dataset.sample_from_datasets([
        tf.data.TFRecordDataset([tfrecord_path.format(tfrecord)],
                                compression_type=compression_type)
        .map(decode_fn)
        .cache()
        .shuffle(buffer_size=shuffle_buffer,
                 reshuffle_each_iteration=True)
        .repeat()
        if shuffle_buffer else
        tf.data.TFRecordDataset([tfrecord_path.format(tfrecord)],
                                compression_type=compression_type)
        .map(decode_fn)
        .cache()
        .repeat()
        for tfrecord in idc_list],
        weights=weights,
        rerandomize_each_iteration=True if shuffle_buffer else False)

    return dataset


def init_node_state(node_set, *, node_set_name):
    if node_set_name == 'atom':
        return tf.concat([node_set["density"], node_set["nuc_charge"]],
                         axis=-1)
    else:
        return node_set["density"]


def init_edge_state(edge_set, *, edge_set_name):
    return edge_set["overlap"]


def drop_features(graph_piece, **unused_kwargs):
    return {}


###############################################################################
# ORCA to TF-GNN data conversion utilities
###############################################################################
def graph_from_orca(out_file: str,
                    overlap_thresh: float,
                    target_features: dict,
                    dummy: bool = False):

    diagonal_densities,               \
        off_diagonal_densities,       \
        off_diagonal_overlaps,        \
        adjacency_atom2link_sources,  \
        adjacency_atom2link_targets,  \
        adjacency_link2atom_sources,  \
        adjacency_link2atom_targets,  \
        atoms, natoms, nlinks, nbas = \
        blocks_from_orca(out_file, overlap_thresh, dummy=dummy)

    graph = tfgnn.GraphTensor.from_pieces(
        context=tfgnn.Context.from_fields(
            features={target_feature: tf.constant([target_value])
                      for target_feature, target_value
                      in target_features.items()}),
        node_sets={
            "atom": tfgnn.NodeSet.from_fields(
                sizes=tf.constant([natoms]),
                features={"density": tf.constant(diagonal_densities),
                          "nuc_charge": tf.constant(atoms)}),
            "link": tfgnn.NodeSet.from_fields(
                sizes=tf.constant([nlinks]),
                features={"density": tf.constant(off_diagonal_densities)})},
        edge_sets={
            "atom2link": tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([nlinks]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("atom", tf.constant(adjacency_atom2link_sources)),
                    target=("link", tf.constant(adjacency_atom2link_targets))),
                features={"overlap": tf.constant(off_diagonal_overlaps)}),
            "link2atom": tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([nlinks]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("link", tf.constant(adjacency_link2atom_sources)),
                    target=("atom", tf.constant(adjacency_link2atom_targets))),
                features={"overlap": tf.constant(off_diagonal_overlaps)})})

    return graph, nbas


def tfrecord_from_orca(data_df: pd.DataFrame,
                       orca_out_path: str,
                       overlap_thresh: float,
                       save_path: str,
                       record_name: str,
                       targets: list,
                       scalings: dict,
                       rot_aug: int = 1,
                       gepol_path: str = '',
                       dummy: bool = False):
    record_path = os.path.join(save_path, f'{record_name}.tfrecord')
    record_options = tf.io.TFRecordOptions(compression_type='GZIP')

    target_functions_dict = {'dipole': dipole_from_orca_out,
                             'vol': functools.partial(vol_from_orca_out,
                                                      gepol_path=gepol_path,
                                                      dummy=dummy),
                             'surf': functools.partial(surf_from_orca_out,
                                                       gepol_path=gepol_path,
                                                       dummy=dummy)}

    def context_features_values(row, out_file, targets, scalings):
        target_values = {target: row[target] / scalings.get(target, 1.0)
                         if target in row
                         else (target_functions_dict[target](out_file)
                               / scalings.get(target, 1.0))
                         for target in targets}
        return target_values

    with tf.io.TFRecordWriter(record_path, options=record_options) as writer:
        for idx, row in data_df.iterrows():
            isomer = row['isomer_id']
            nconf = row['nconf']
            for conf in range(1, nconf + 1):
                prop_out_file = os.path.join(orca_out_path,
                                             f'{isomer}_{conf}_1_dft.zip')
                target_features = context_features_values(row,
                                                          prop_out_file,
                                                          targets,
                                                          scalings)
                for aug in range(1, rot_aug + 1):
                    out_file = os.path.join(orca_out_path,
                                            f'{isomer}_{conf}_{aug}.zip')
                    graph, nbas = graph_from_orca(out_file,
                                                  overlap_thresh,
                                                  target_features,
                                                  dummy=dummy)
                    example = tfgnn.write_example(graph)
                    writer.write(example.SerializeToString())

    return nbas


def serialize_from_orca(csv_path: str,
                        orca_out_path: str,
                        overlap_thresh: float,
                        save_path: str,
                        record_name: str,
                        schema_template_path: str,
                        scalings_csv_path: str = '',
                        monolith: bool = True,
                        multi_target: bool = False,
                        rot_aug: int = 1,
                        gepol_path: str = '',
                        dummy: bool = False):
    data_df = pd.read_csv(csv_path)
    os.makedirs(os.path.join(save_path, record_name), exist_ok=True)
    targets = \
        context_features(schema_template_path) if multi_target else ['target']
    if scalings_csv_path:
        scalings_df = pd.read_csv(scalings_csv_path)
        scalings = {target: divisor for target, divisor
                    in zip(scalings_df['target'].to_list(),
                           scalings_df['divisor'].to_list())}
    else:
        scalings = {}
    if monolith:
        nbas = tfrecord_from_orca(data_df=data_df,
                                  orca_out_path=orca_out_path,
                                  overlap_thresh=overlap_thresh,
                                  save_path=os.path.join(save_path,
                                                         record_name),
                                  record_name=record_name,
                                  targets=targets,
                                  scalings=scalings,
                                  rot_aug=rot_aug,
                                  gepol_path=gepol_path,
                                  dummy=dummy)
    else:
        for cas in data_df['cas'].unique():
            nbas = tfrecord_from_orca(data_df=data_df[data_df['cas'] == cas],
                                      orca_out_path=orca_out_path,
                                      overlap_thresh=overlap_thresh,
                                      save_path=os.path.join(save_path,
                                                             record_name),
                                      record_name=cas,
                                      targets=targets,
                                      scalings=scalings,
                                      rot_aug=rot_aug,
                                      gepol_path=gepol_path,
                                      dummy=dummy)
    save_schema(nbas, schema_template_path,
                os.path.join(save_path, f'{record_name}.pbtxt'))
