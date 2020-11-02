#!/usr/bin/python3

from argparse import ArgumentParser

import dgl

import numpy as np

import pickle

import pandas

from sentence_transformers import SentenceTransformer

import torch


def store_graphs(df, graph_structures):
    """Read data from the pickled file and store in a dictionary."""
    graph_scores = {}
    ids = []
    graphs = []
    acceptability = []
    relevance = []
    sufficiency = []
    cogency = []

    for index, row in df.iterrows():
        arg_id = row['arg_id']
        if arg_id in graph_structures:
            ids.append(arg_id)
            graph = graph_structures[arg_id]
            graphs.append(graph)
            acceptability.append(row['LocalAcceptability'])
            relevance.append(row['LocalRelevance'])
            sufficiency.append(row['LocalSufficiency'])
            cogency.append(row['Cogency'])

    graph_scores['arg_ids'] = ids
    graph_scores['graphs'] = graphs
    graph_scores['acceptability_scores'] = acceptability
    graph_scores['relevance_scores'] = relevance
    graph_scores['sufficiency_scores'] = sufficiency
    graph_scores['cogency'] = cogency

    return graph_scores


def get_max_len(graphs):
    """Compute maximum number of nodes."""
    lens = []
    for g in graphs:
        lens.append(len(g))
    return max(lens)


def encode_relations(graph_structures):
    """Encode discourse relations with
    one-hot vectors.
    """
    all_rel = []
    rel_enc = {}
    for graph in graph_structures:
        for el in graph:
            all_rel.append(el[2]['rel'])

    classes = set(all_rel)
    one_hot_targets = torch.eye(len(classes))
    for i, rel in enumerate(classes):
        rel_enc[rel] = one_hot_targets[i, :]

    return rel_enc


def use_glove_features(nodes_dict, id_text):
    """Use Glove embeddings as node features"""

    graph_feat = torch.zeros((len(nodes_dict), 300))
    for k, v in nodes_dict.items():
        node_emb = id_text[k]
        graph_feat[[v]] = torch.from_numpy(node_emb)

    return graph_feat


def use_bert_features(nodes_dict, id_text):
    """Use Sentence BERT embeddings as node features"""

    graph_feat = torch.zeros((len(nodes_dict), 768))
    for k, v in nodes_dict.items():
        node_emb = id_text[k]
        graph_feat[[v]] = torch.from_numpy(node_emb)

    return graph_feat


def graph_to_dgl(graph_structures, enc_rel, node_feat='rand'):
    """Create dgl graphs from dictionary,
    add edge and node features
    """
    dgls = []
    for graph in graph_structures:
        local_dict = {}
        counter = 0
        G = dgl.DGLGraph()
        edges = []
        for el in graph:
            id_1 = el[0]
            if id_1 not in local_dict:
                local_dict[id_1] = counter
                counter += 1
            id_2 = el[1]
            if id_2 not in local_dict:
                local_dict[id_2] = counter
                counter += 1
            edges.append((local_dict[id_1], local_dict[id_2], el[2]['rel']))
        G.add_nodes(len(local_dict))
        # Use different feature types
        if node_feat == 'rand':
            G.ndata['x'] = torch.zeros((len(local_dict), 300))
            # TODO: feature assignment using embeddings
            for k, v in local_dict.items():
                G.nodes[[v]].data['x'] = torch.ones(1, 300)
        elif node_feat == 'glove':
            with open(args.glove_emb, "rb") as f:
                glove_vectors = pickle.load(f)
            feat = use_glove_features(local_dict, glove_vectors)
            G.ndata['x'] = feat
        elif node_feat == 'bert':
            with open(args.bert_emb, "rb") as f:
                bert_vectors = pickle.load(f)
            feat = use_bert_features(local_dict, bert_vectors)
            G.ndata['x'] = feat
        # Add edges
        for i, edge in enumerate(edges):
            G.add_edge(edge[0], edge[1])
        G.edata['y'] = torch.zeros(len(edges), 13)
        for i, edge in enumerate(edges):
            rel = edge[2]
            G.edata['y'][i] = enc_rel[rel]
        dgls.append(G)

    return dgls


if __name__ == '__main__':

    parser = ArgumentParser(
        description='Main ODQA script')
    parser.add_argument(
        '--graphs', default='data/graphs.pickle',
        help='Pickled graphs')
    parser.add_argument(
        '--dataset', default='data/dg_scores.csv',
        help='Dataset containing quality scores')
    parser.add_argument(
        '--node-feat-type', choices=['rand', 'glove', 'bert'],
        default='rand', help='Node feature type')
    parser.add_argument(
        '--glove_emb', default='data/nodes_glove.pickle',
        help='Path to the file with nodes encoded with GloVe embeddings')
    parser.add_argument(
        '--bert_emb', default='data/nodes_bert.pickle',
        help='Path to the file with nodes encoded with BERT embeddings')
    args = parser.parse_args()

    with open(args.graphs, 'rb') as fd:
        graph_structures = pickle.load(fd)

    df = pandas.read_csv(args.dataset)

    graph_scores = store_graphs(df, graph_structures[2])
    idx_text = graph_structures[0]  # dictionary with nodes id and corresponding text
    max_nodes = get_max_len(graph_scores['graphs'])
    enc_rel = encode_relations(graph_scores['graphs'])
    print(enc_rel)
    dgl_graphs_rand = graph_to_dgl(graph_scores['graphs'], enc_rel)
    dgl_graphs_glove = graph_to_dgl(graph_scores['graphs'], enc_rel, node_feat='glove')
    dgl_graphs_bert = graph_to_dgl(graph_scores['graphs'], enc_rel, node_feat='bert')

    graph_data = {}

    graph_data['dgl_graphs_rand'] = dgl_graphs_rand
    graph_data['dgl_graphs_glove'] = dgl_graphs_glove
    graph_data['dgl_graphs_bert'] = dgl_graphs_bert
    graph_data['acceptability_scores'] = graph_scores['acceptability_scores']
    graph_data['relevance_scores'] = graph_scores['relevance_scores']
    graph_data['sufficiency_scores'] = graph_scores['sufficiency_scores']
    graph_data['cogency'] = graph_scores['cogency']

    # Store the graphs and labels to .pkl
    with open('data/graphs_scores_dict.pickle', 'wb') as file:
        pickle.dump(graph_data, file, protocol=pickle.HIGHEST_PROTOCOL)
