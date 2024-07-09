import os
import functools

import pandas as pd
import tensorflow as tf
import tensorflow_gnn as tfgnn

from .qchem import qchem_parsers


def read_schema(schema_path):
    schema = tfgnn.read_schema(schema_path)
    return tfgnn.create_graph_spec_from_schema_pb(schema)


def save_schema(nbas, schema_template_path, schema_save_path):
    with open(schema_template_path) as f:
        schema_template = f.read()
    with open(schema_save_path, 'w') as f:
        f.write(schema_template.format(ndiag=nbas * (nbas + 1) // 2,
                                       noffdiag=nbas * nbas))


def graph_from_orca(out_file, overlap_thresh, target, label):

    diagonal_densities, off_diagonal_densities,  off_diagonal_overlaps, \
        adjacency_atom2link_sources, adjacency_atom2link_targets, \
        adjacency_link2atom_sources, adjacency_link2atom_targets, \
        natoms, nlinks, nbas = \
        qchem_parsers.blocks_from_orca(out_file, overlap_thresh)

    graph = tfgnn.GraphTensor.from_pieces(context=tfgnn.Context.from_fields(
            features={'target': tf.constant([target]),
                      'id':  tf.constant([label])}),
                                          node_sets={
        "atom": tfgnn.NodeSet.from_fields(
            sizes=tf.constant([natoms]),
            features={"density": tf.constant(diagonal_densities)}),
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


def monolithic_record_from_orca(csv_path: str, orca_out_path: str,
                                overlap_thresh: float, save_path: str,
                                record_name: str, schema_template_path: str):
    table = pd.read_csv(csv_path)
    with tf.io.TFRecordWriter(
            os.path.join(save_path, f'{record_name}.tfrecord')
    ) as writer:
        for idx, row in table.iterrows():
            cas = row['cas']
            target = row['target']
            nconf = row['nconf']
            for conf in range(nconf):
                out_file = os.path.join(orca_out_path, f'{cas}_{conf}.zip')
                print(out_file)
                graph, nbas = \
                    graph_from_orca(out_file, overlap_thresh, target, conf)
                example = tfgnn.write_example(graph)
                writer.write(example.SerializeToString())
    save_schema(nbas, schema_template_path,
                os.path.join(save_path, f'{record_name}.pbtxt'))


def buckets_from_orca(csv_path: str, orca_out_path: str,
                      overlap_thresh: float, save_path: str,
                      record_name: str, schema_template_path: str):
    table = pd.read_csv(csv_path)
    tfrecord_save_path = os.path.join(save_path, record_name)
    os.makedirs(tfrecord_save_path, exist_ok=True)
    for idx, row in table.iterrows():
        cas = row['cas']
        target = row['target']
        nconf = row['nconf']
        with tf.io.TFRecordWriter(
                os.path.join(tfrecord_save_path, f'{cas}.tfrecord')
        ) as writer:
            for conf in range(nconf):
                out_file = os.path.join(orca_out_path, f'{cas}_{conf}.zip')
                graph, nbas = \
                    graph_from_orca(out_file, overlap_thresh, target, conf)
                example = tfgnn.write_example(graph)
                writer.write(example.SerializeToString())
    save_schema(nbas, schema_template_path,
                os.path.join(save_path, f'{record_name}.pbtxt'))


def serialize_from_orca(csv_path: str, orca_out_path: str,
                        overlap_thresh: float, save_path: str,
                        record_name: str, schema_template_path: str,
                        monolith: bool = True):
    if monolith:
        monolithic_record_from_orca(
            csv_path, orca_out_path, overlap_thresh, save_path, record_name,
            schema_template_path)
    else:
        buckets_from_orca(
            csv_path, orca_out_path, overlap_thresh, save_path, record_name,
            schema_template_path)


def decode_graph(record_bytes, graph_spec=None, context_feature=None):
    assert graph_spec is not None, "No graph specification provided"
    assert context_feature is not None, "No context feature name provided"
    graph = tfgnn.parse_single_example(graph_spec, record_bytes, validate=True)
    return graph, graph.context[context_feature]


def get_decode_fn(graph_spec, context_feature):
    return functools.partial(decode_graph,
                             graph_spec=graph_spec,
                             context_feature=context_feature)


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
    return node_set["density"]


def init_edge_state(edge_set, *, edge_set_name):
    return edge_set["overlap"]


def drop_features(graph_piece, **unused_kwargs):
    return {}
