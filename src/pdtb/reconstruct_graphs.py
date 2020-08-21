#!/usr/bin/python3

import os
from argparse import ArgumentParser
from collections import defaultdict, Counter, OrderedDict
import difflib
from graphviz import Digraph
import itertools
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.algorithms.isomorphism import DiGraphMatcher
import pickle


def get_file_paths(data_dir):
    '''Get paths for all files in the given directory'''

    file_names = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(args.data):
        for file in f:
            if '.pipe' in file:
                file_names.append(os.path.join(r, file))
    return file_names


def split_data(data):

    splitted_data = data.rstrip('\n').split('|')
    return splitted_data


def check_connections(Arg_seq):

    if (len(set(Arg_seq)) < len(Arg_seq)):
        return True
    else:
        return False


def get_arg_data(arg_type, record):

    if arg_type == 'Arg1':
        sec_num = record[1]  # Section number (folder name)
        file_num = record[2]  # File number
        span = record[22]  # Arg1 span
        text = record[24]  # Arg1 text
        return (sec_num, file_num, 'Arg1', span, text), text
    else:
        rel_type = record[0]  # Relation type
        conn_span = record[3]  # Connextive/AltLex span list
        conn_exp = record[8]  # Explicit connective
        # 1st Implicit connective (can have two connectives)
        conn_imlp1 = record[9]
        conn_impl2 = record[10]  # 2nd Implicit connective
        # Implicit, Explicit and AltLex relations
        sem_class1_conn1 = record[11]
        # Implicit, Explicit and AltLex relations
        sem_class2_conn1 = record[12]
        sem_class1_conn2 = record[13]  # only Implicit relations
        sem_class2_conn2 = record[14]  # only Implicit relations
        span = record[32]
        text = record[34]

        new_record = (rel_type, conn_span, conn_exp, conn_imlp1, conn_impl2,
                      sem_class1_conn1, sem_class1_conn2, sem_class2_conn1, sem_class2_conn2,
                      span, text)

        return new_record, text


def assign_id(record, id_dict, id_dict_inv, dict_text, arg_id, id_text):
    ''' 
    Assign id to a text unit. 
    Arguments with the same span get the same id
    '''
    arg_types = ['Arg1', 'Arg2']
    span1 = record[22]  # column 23
    span2 = record[32]  # column 33
    text1 = record[24]
    text2 = record[34]
    arg_seq = []

    arg_attributes = []
    for arg_type, span in zip(arg_types, [span1, span2]):
        key = span
        key_text = span  # key for the text dictionary
        arg_attribute, text = get_arg_data(arg_type, record)

        if key not in id_dict.keys():
            new_rec = {key: arg_id}
            new_rec_text = {key_text: text}
            arg_attributes.append(arg_attribute)
            id_dict.update(new_rec)
            dict_text.update(new_rec_text)
            arg_seq.append(arg_id)
            arg_id += 1
        else:
            arg_attributes.append(arg_attribute)
            repeated_id = id_dict[key]
            arg_seq.append(repeated_id)

    key_inv = (id_dict[span1], id_dict[span2])
    new_rec_inv = {key_inv: arg_attributes}
    id_dict_inv.update(new_rec_inv)

    if id_dict[span1] not in id_text:
        id_text[id_dict[span1]] = text1

    if id_dict[span2] not in id_text:
        id_text[id_dict[span2]] = text2

    return id_dict, id_dict_inv, arg_seq, dict_text, arg_id, id_text


def get_Arg_structure(Arg_seqi, dict_inv):

    # Reconstruct an Argument
    Arg_structure = defaultdict()
    Arg_seq = iter(Arg_seqi)

    for i, arg in enumerate(Arg_seq):
        if arg not in Arg_structure.keys():
            Arg_structure[arg] = [next(Arg_seq)]
        else:
            Arg_structure[arg].append(next(Arg_seq))

    # Delete all relations except Explicit, Implicit and AltLex
    for key in list(Arg_structure):
        new_values = []
        for i, value in enumerate(Arg_structure[key]):
            rel_type = dict_inv[(key, value)][1][0]
            if (rel_type == 'Explicit' or
                    rel_type == 'Implicit' or
                    rel_type == 'AltLex'):
                new_values.append(value)
        Arg_structure[key] = new_values
        if len(Arg_structure[key]) == 0:
            del Arg_structure[key]
    
    # Clean the Argument stucture
    # (Delete one-to-one links with no connections)
    val_list = [y for x in Arg_structure.values() for y in x]
    for key in list(Arg_structure):
        if (key not in val_list and
                len(set(Arg_structure[key]).intersection(set(Arg_structure.keys()))) == 0):
            del Arg_structure[key]

    return Arg_structure


def separate_Arg(Arg_structure):
    '''
    Split an Argument dict since there can be several
    Args which are not connected to each other in one file
    '''
    graph = nx.DiGraph(Arg_structure)
    subtrees = []
    for nodes in nx.weakly_connected_components(graph):
        subgraph = graph.subgraph(nodes)
        subtree = nx.to_dict_of_lists(subgraph)
        subtrees.append(subtree)
    return subtrees


def to_graph(Arg_structure, dict_inv):

    '''Transform to the graphical representation'''

    edges = []
    for key, values in Arg_structure.items():
        for value in values:
            rel_type = dict_inv[(key, value)][1][0]
            sem_rel = dict_inv[(key, value)][1][5]
            edges.append(
                (key, value, {'sem_rel': sem_rel}))

    G = nx.DiGraph()
    G.add_edges_from(edges)

    return G


def check_isomorphism(Arg_list):

    # Find unique argument structures 
    # checking graphs for isomorphism
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

    # Display the graph
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, font_weight='bold')


    # Display edge attributes: relation and semantic information
    #  (looks ugly)
    #edge_rel_labels = nx.get_edge_attributes(G,'rel_type')
    edge_sem_labels = nx.get_edge_attributes(G,'sem_rel')
    #nx.draw_networkx_edge_labels(G, pos, labels = edge_rel_labels)
    nx.draw_networkx_edge_labels(G, pos, labels = edge_sem_labels)
    plt.savefig('{}_{}_sampl.png'.format(num, n_samples), format='png')
    plt.close()
    plt.show()


if __name__ == '__main__':

    parser = ArgumentParser(
        description='PDTB parser')
    parser.add_argument(
        'data', help='Path to the directory containing the data')
    args = parser.parse_args()

    file_paths = get_file_paths(args.data)

    id_counter = 0
    id_dict = {}
    id_dict_inv = {}
    dict_text = {}
    id_text = {}

    counter_all = 0
    counter = 0
    Arg_num = 0

    Arg_final = []

    for file in file_paths:
        with open(file, 'r', encoding='utf-8', errors='ignore') as fd:
            content = fd.readlines()

        # Here we construct an Argument from the connected text units
        Arg_seq = []
        Arg_item = []
        counter_all += 1
        for item in content:
            item = split_data(item)
            id_dict, id_dict_inv, arg_seq, dict_text, arg_id, id_text = assign_id(
                item, id_dict, id_dict_inv, dict_text, id_counter, id_text)
            id_counter = arg_id
            Arg_seq += arg_seq
            Arg_item.append(item)

        if check_connections(Arg_seq):
            counter += 1
            # Get Arguments
            Arg_nested_seq = get_Arg_structure(Arg_seq, id_dict_inv)
            splitted_Args = separate_Arg(Arg_nested_seq)
            Arg_final += splitted_Args

    print('Total number of arguments: {}'.format(len(Arg_final)))

    graphs = []
    for Arg in Arg_final:
        graph = to_graph(Arg, id_dict_inv)
        graphs.append(graph)

    unique_graphs = check_isomorphism(graphs)

    for i, family in enumerate(unique_graphs):
        n_samples = len(family)
        show_graph(family[0], i, n_samples)
        dict_repr = nx.to_dict_of_lists(family[0])

        for key, values in dict_repr.items():
            print(dict_repr)
            print(key, id_text[key])
            for value in values:
                print(value, id_text[value])
     
    print("Overall number of Arguments is {}".format(len(Arg_final)))



