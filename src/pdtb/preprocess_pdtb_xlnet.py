#!/usr/bin/python3

import os
from argparse import ArgumentParser
from collections import Counter
import random


def get_file_paths(data_dir):
    # Get paths for all files in the given directory

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


def preprocess_disc_rel(discourse_rel, l1=True, l2=False):

    discourse_rel = discourse_rel.split('.')
    if l2:
        if len(discourse_rel) == 1:
            discourse_rel = discourse_rel[0]
        else:
            discourse_rel = "{}.{}".format(discourse_rel[0], discourse_rel[1])
    else:
        discourse_rel = discourse_rel[0]

    return discourse_rel


def read_file(file, pdtb=False):

    preprocessed_lines = []
    with open(file, 'r', encoding='utf-8', errors='ignore') as fd:
        content = fd.readlines()
        for item in content:
            item = split_data(item)
            if pdtb:
                sec_num = item[1] # Section number (folder name)
                file_num = item[2] # File number
            else:
                file_name = os.path.basename(file)
                file_name = os.path.splitext(file_name)[0]
                file_name = os.path.splitext(file_name)[0]
                file_name = file_name.replace('wsj_', '')
                sec_num = file_name[:2]
                file_num = file_name[-2:]
            rel_type = item[0] # Relation type
            text1 = item[24] # Arg1 text
            text2 = item[34] # Arg2 text       

            if rel_type == "Implicit" or rel_type == "Explicit" or rel_type == "AltLex" or rel_type == "EntRel":
                #disc_rel = preprocess_disc_rel(item[11], l2=True)
                # second optional field for the Implicit, Explicit and AltLex: record[13]
                line = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(sec_num, file_num, rel_type, "Rel", text1, text2)
                preprocessed_lines.append(line)
            else:
                print('jjjjjjjjjjjjjj')
                line = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(sec_num, file_num, rel_type, "NoRel", text1, text2)
                preprocessed_lines.append(line)

    return preprocessed_lines


def split_corpus(corpus, pdtb=True):
    '''Split preprocessed corpus into the train, dev and test sets'''

    train_set = []
    #dev_set = []
    test_set = []

    for line in corpus:
        splitted_line = line.rstrip('\n').split('\t')
        section = int(splitted_line[0])
        if section in range(0, 23):
            train_set.append(line)
        elif section in range(23, 25):
            test_set.append(line)
        # No dev set so far
        #elif section in range(0, 2):
        #    dev_set.append(line)

    #return train_set, dev_set, test_set
    return train_set, test_set


def count_senses(corpus):
    
    sense_list = [line.split('\t')[3] for line in corpus]
    sense_count = Counter(sense_list)

    rel_list = [line.split('\t')[2] for line in corpus]
    rel_count = Counter(rel_list)

    return sense_count, rel_count


def delete_infrequent(corpus, sense_count, train=False):

    labels = []
    print(len(corpus))
    for line in corpus[:]:
        disc_rel = line.split('\t')[3]
        disc_rel_split = disc_rel.split('.')
        if (sense_count[disc_rel] < 50 or
            len(disc_rel_split) == 1):
            corpus.remove(line)
        else:
            labels.append(disc_rel)

    if train:
        print(Counter(labels))
    
    return corpus, set(labels)


def handle_imbalance(training_set):

    args1 = []
    args2 = []
    norel_items = []
    generated = []

    for item in training_set:
        item = item.rstrip().split('\t')
        rel = item[3]

        if rel == 'Rel':
            args1.append(item[4])
            args2.append(item[5])

        if rel == 'NoRel':
            norel_items.append(item)

    # add items to the non-rel class by randomly combining statements 
    assert len(args1) == len(args2)

    imbalance_count = len(args1) - len(norel_items)
    #args1_sampling = random.choices(args1, k=imbalance_count)
    args1_sampling_indicies = random.choices(range(len(args1)), k=int(imbalance_count * 0.006))
    for i, arg1_i in enumerate(args1_sampling_indicies):
        print('{} out of {} processed'.format(i, len(args1_sampling_indicies)))
        # assert we don't pick the corresponding pair
        arg1 = args1[arg1_i]
        #args2_exlude = args2[1:arg1_i] + args2[arg1_i+1:]
        args2_exlude = args2.copy()
        del args2_exlude[arg1_i]
        args2_random = random.choice(args2_exlude)
        line = '{}\t{}\t{}\t{}\t{}\t{}\n'.format("generated", "generated", "generated", "NoRel", arg1, args2_random)
        norel_items.append(line)
        generated.append(line)

    #assert len(args1) == len(norel_items)

    balanced = training_set + generated

    return balanced


def write_to_file(corpus, name):

    with open(name, 'w') as file:
        for line in corpus:
            file.write(line)
        file.close()


if __name__ == '__main__':

    parser = ArgumentParser(
        description='PDTB parser')
    parser.add_argument(
        'data', help='Path to the directory containing the data')
    args = parser.parse_args()

    file_paths = get_file_paths(args.data)

    preprocessed_corpus = []

    for file in file_paths:
        new_lines = read_file(file)
        preprocessed_corpus += new_lines

    #train_set, dev_set, test_set = split_corpus(preprocessed_corpus)
    #train_set, test_set = split_corpus(preprocessed_corpus)

    #sense_count, relation_count = count_senses(train_set)

    balanced_train_set = handle_imbalance(preprocessed_corpus)
    print(len(balanced_train_set))
    random.shuffle(balanced_train_set)
    print(balanced_train_set)
    #train_set, labels_train = delete_infrequent(train_set, sense_count, train=True)
    #dev_set, labels_dev = delete_infrequent(dev_set, sense_count)
    #test_set, labels_test = delete_infrequent(test_set, sense_count)

    # print(len(train_set), len(dev_set), len(test_set))
    #print(len(train_set), len(test_set))

    sense_count_tr, relation_count = count_senses(balanced_train_set)
    #sense_count_tr_bal, relation_count = count_senses(balanced_train_set)
    #sense_count_dev, relation_count = count_senses(dev_set)
    #sense_count_ts, relation_count = count_senses(test_set)

   # assert labels_train == labels_dev == labels_test

    write_to_file(balanced_train_set, 'ptb_dev.txt')
    #write_to_file(balanced_train_set, 'ptb_train_bal.txt')
    #write_to_file(dev_set, 'pdtb_dev.txt')
    #write_to_file(test_set, 'ptb_test.txt')
    #print(len(balanced_train_set))