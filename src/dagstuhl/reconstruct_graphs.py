#!/usr/bin/python3

import os
from argparse import ArgumentParser
import difflib
from graphviz import Digraph
import itertools
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.algorithms.isomorphism import DiGraphMatcher
import pandas

'''
Represent Dagstuhl arguments as graphs

Usage: python3 dagstuhl_preprocessing.py -evidence_directory -claim_directory -raw_files
Example: python3 dagstuhl_preprocessing.py ../dagstuhl-15512-argquality-corpus-v2/dagstuhl-15512-argquality-corpus-v2/dagstuhl_segmenter_output_corrected/ ../dagstuhl-15512-argquality-corpus-v2/dagstuhl-15512-argquality-corpus-v2/claims/ ../dagstuhl-15512-argquality-corpus-v2/dagstuhl-15512-argquality-corpus-v2/dagstuhl-15512-argquality-corpus-unannotated/
'''

def arg_to_dict(df):
    '''
    Assign ids to the text units, 
    define argument structures
    '''

    counter = 0
    arg_ids = {}
    arg_ids_inv = {}
    arg_structure = {}

    for index, record in df.iterrows():
        arg_id = record[0]
        arg1 = record[3]
        arg2 = record[4]
        rel = record[2]
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
            record = (arg_ids[arg_id][arg1], arg_ids[arg_id][arg2], {'rel': rel})
            arg_structure[arg_id].append(record)

    return arg_ids, arg_ids_inv, arg_structure


def to_graph(arg_structure):
    '''Transform to the graph representation'''
    
    G = nx.DiGraph()
    G.add_edges_from(arg_structure) # value of the dict
    num_nodes = G.number_of_nodes()
    print(num_nodes)

    return G, num_nodes


def check_isomorphism(Arg_list):
    '''
    Find unique argument structures checking graphs 
    for isomorphism
    '''

    # TODO: check for isomorphism taking into account edges (rel, sem_rel)
    unclassified_graphs = Arg_list
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


def show_graph(G, num, n_samples):

    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=False, font_weight='bold')
    edge_sem_labels = nx.get_edge_attributes(G,'rel')
    #nx.draw_networkx_edge_labels(G, pos, labels = edge_sem_labels)
    plt.savefig('{}_{}_sampl.png'.format(num, n_samples), format='png')
    plt.close()


if __name__ == '__main__':

    parser = ArgumentParser(
        description='Show labeled data')
    parser.add_argument(
        'dataset', help='Path to the file containing the dataset')
    args = parser.parse_args()

    df = pandas.read_csv(args.dataset, header=0)
    # drop rows without labels
    df = df.dropna(subset = ['label_corrected'], inplace=False)
    df = df.dropna(subset = ['XLNet_label'], inplace=False)

    arg_ids, arg_ids_inv, arg_structures = arg_to_dict(df)

    graphs = []
    arg_ids = []
    num_nodes = []
    for unique_id, item in arg_structures.items():
        graph, num_node = to_graph(item)
        num_nodes.append(num_node)
        graphs.append(graph)
        arg_ids.append(unique_id)

    unique_graphs = check_isomorphism(graphs)

    # save graphs in png formt
    for i, family in enumerate(unique_graphs):
        n_samples = len(family)
        show_graph(family[0], i, n_samples)
        dict_repr = nx.to_dict_of_lists(family[0])