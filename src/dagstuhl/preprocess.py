#!/usr/bin/python3

import csv
import ntpath
import re
import string
import os
import csv
import pandas
from argparse import ArgumentParser

'''
Preprocess Dagstuhl corpus for fine-tuning and classification with XLNet

Usage: python3 dagstuhl_preprocessing.py -evidence_directory -claim_directory -raw_files

Example: python3 dagstuhl_preprocessing.py ../dagstuhl-15512-argquality-corpus-v2/dagstuhl-15512-argquality-corpus-v2/dagstuhl_segmenter_output_corrected/ ../dagstuhl-15512-argquality-corpus-v2/dagstuhl-15512-argquality-corpus-v2/claims/ ../dagstuhl-15512-argquality-corpus-v2/dagstuhl-15512-argquality-corpus-v2/dagstuhl-15512-argquality-corpus-unannotated/
'''


def get_file_paths(data_dir, extension):
    # Get paths for all files in the given directory

    file_names = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(data_dir):
        for file in f:
            if file.endswith(extension):
                file_names.append(os.path.join(r, file))
    return file_names


def get_id(raw_folder, evidence_file_name):

    csv_file_name = preprocess_file_name(evidence_file_name)
    evidence_number = evidence_file_name[-5:-4]
    evidence_number = int(evidence_number)
    content = pandas.read_csv(raw_folder + csv_file_name + '.csv', header=0, sep='\t')
    arg_id = content.iloc[evidence_number, 0]

    return arg_id


def preprocess_file_name(name, claim=False):

    name = ntpath.basename(name)
    name = os.path.splitext(name)[0]
    if claim:
        name = name.replace('claim_', '') # remove claim_ from the name
    else:   
        name = name[:-2] # remove _number from the filename

    return name


def create_pairs(claim, evidence_edus):

    pairs = []

    for edu in evidence_edus: 
        if claim.rstrip() and edu.rstrip():
            claim = claim.replace('\n', '')
            edu = edu.replace('\n', '')
            pairs.append((claim.rstrip(), edu.rstrip()))

    for edu1, edu2 in zip(evidence_edus, evidence_edus[1:]):
        if edu1.rstrip() and edu2.rstrip():
            edu1 = edu1.replace('\n', '')
            edu2 = edu2.replace('\n', '')
            pairs.append((edu1.rstrip(), edu2.rstrip()))

    return pairs


def write_file(file, items):
    # the lable is a placeholder
    with open(file, 'w') as fd:
        for arg_id, label, arg1, arg2 in items:
            fd.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(arg_id, 'dummy',' dummy', 'dummy', 'dummy', label, arg1, arg2))
    

if __name__ == '__main__':

    parser = ArgumentParser(
        description='Create data for labeling (Dagstuhl)')
    parser.add_argument(
        'evidence_files', help='Path to the directory containing the evidences')
    parser.add_argument(
        'claim_files', help='Path to the directory containing the claims')
    parser.add_argument(
        'raw_files', help='Path to the directory containing raw files')
    args = parser.parse_args()

    evidence_files = get_file_paths(args.evidence_files, '.txt')
    claim_files = get_file_paths(args.claim_files, '.txt')

    all_pairs = []

    for evidence_file in evidence_files:
        evidence_file_name = preprocess_file_name(evidence_file)
        for claim_file in claim_files:
            claim_file_name = preprocess_file_name(claim_file, claim=True)
            if evidence_file_name == claim_file_name:
                with open(evidence_file, 'r', encoding='utf-8', errors='ignore') as ef:
                    evidence_content = ef.readlines()
                with open(claim_file, 'r', encoding='utf-8', errors='ignore') as cf:
                    claim_content = cf.read()

                pairs = create_pairs(claim_content, evidence_content)

                evidence_id = get_id(args.raw_files, evidence_file)

                for pair in pairs:
                    all_pairs.append((evidence_id, 'Expansion.Conjunction', pair[0], pair[1]))

    write_file('dagstuhl_xlnet.txt', all_pairs)


