#!/usr/bin/python3


from argparse import ArgumentParser

import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher
from networkx.drawing.nx_agraph import graphviz_layout

import pandas


'''
Represent Dagstuhl arguments as graphs

Usage: python3 reconstruct_graphs.py -dataset

Example: python3 reconstruct_graphs.py\
 ../../../../dagstuhl-15512-argquality-corpus-v2/dagstuhl_labeled.csv
'''


def structure_arguments(df):
    '''
    Assign ids to the text units,
    represent argument structures with dictionaries
    '''

    counter = 0
    arg_ids = {}
    arg_ids_inv = {}
    arg_structure = {}

    for index, record in df.iterrows():
        arg_id, _, rel, arg1, arg2, _ = record
        if arg_id not in arg_ids:
            arg_ids[arg_id] = {}
            arg_ids_inv[arg_id] = {}
            arg_structure[arg_id] = []
        if arg1 not in arg_ids[arg_id]:
            arg_ids[arg_id][arg1] = counter
            arg_ids_inv[arg_id][counter] = arg1
            counter += 1
        if arg2 not in arg_ids[arg_id]:
            arg_ids[arg_id][arg2] = counter
            arg_ids_inv[arg_id][counter] = arg2
            counter += 1

        if rel != 'NoRel':
            record = (arg_ids[arg_id][arg1],
                      arg_ids[arg_id][arg2], {'rel': rel})
            arg_structure[arg_id].append(record)

    return arg_ids, arg_ids_inv, arg_structure


def to_graph(arg_structure):
    '''Transform dictionaries to the graph representations'''

    g = nx.DiGraph()
    g.add_edges_from(arg_structure)
    return g, num_nodes


def group_into_isomorphic_families(argument_graphs):
    '''
    Find unique argument structures and group them
    into families
    TODO: check for isomorphism taking into account edges (rel, sem_rel)
    '''

    unclassified_graphs = argument_graphs
    graph_families = []
    for unclassified_graph in unclassified_graphs:
        for graph_family in graph_families:
            family_member = graph_family[0]
            dm = DiGraphMatcher(unclassified_graph, family_member)
            if dm.is_isomorphic():
                graph_family.append(unclassified_graph)
                break
        else:
            graph_families.append([unclassified_graph])

    return graph_families


def render_graph(g, num, n_samples):

    pos = graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=False, font_weight='bold')
    edge_sem_labels = nx.get_edge_attributes(g, 'rel')
    # uncomment to print edge labels
    # nx.draw_networkx_edge_labels(G, pos, labels = edge_sem_labels)
    plt.savefig('{}_{}_sampl.png'.format(num, n_samples), format='png')
    plt.close()


if __name__ == '__main__':

    parser = ArgumentParser(
        description='Show labeled data')
    parser.add_argument(
        'dataset', help='Path to the file containing labeled dataset')
    args = parser.parse_args()

    df = pandas.read_csv(args.dataset, header=0)
    # drop rows without labels
    df = df.dropna(subset=['label_corrected'], inplace=False)
    df = df.dropna(subset=['XLNet_label'], inplace=False)

    arg_ids, arg_ids_inv, arg_structures = structure_arguments(df)

    graphs = []
    arg_ids = []
    num_nodes = []
    for unique_id, item in arg_structures.items():
        graph, num_node = to_graph(item)
        num_nodes.append(num_node)
        graphs.append(graph)
        arg_ids.append(unique_id)

    unique_graphs = group_into_isomorphic_families(graphs)

    # save graphs in png formt
    for i, family in enumerate(unique_graphs):
        n_samples = len(family)
        render_graph(family[0], i, n_samples)
