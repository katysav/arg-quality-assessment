#!/usr/bin/python3

import csv
import glob
import os

from argparse import ArgumentParser
import pandas


'''
Preprocess Dagstuhl corpus for fine-tuning and classification with XLNet

Usage: python3 dagstuhl_preprocessing.py -evidence_directory\
-claim_directory -raw_files

Example: python3 preprocess.py \
../../../dagstuhl-15512-argquality-corpus-v2\
/dagstuhl-15512-argquality-corpus-v2/dagstuhl_segmenter_output_corrected/\
 ../../../dagstuhl-15512-argquality-corpus-v2/\
 dagstuhl-15512-argquality-corpus-v2/claims/\
 ../../../dagstuhl-15512-argquality-corpus-v2\
 /dagstuhl-15512-argquality-corpus-v2/dagstuhl-15512-argquality-corpus-unannotated/
'''


def get_id(raw_folder, evidence_file_name):

    csv_file_name = preprocess_file_name(evidence_file_name)
    evidence_number = int(evidence_file_name[-5:-4])
    content = pandas.read_csv(
        raw_folder + csv_file_name + '.csv', header=0, sep='\t')
    arg_id = content.iloc[evidence_number, 0]

    return arg_id


def preprocess_file_name(name, claim=False):

    name = os.path.basename(name)
    name = os.path.splitext(name)[0]
    if claim:
        name = name.replace('claim_', '')  # remove claim_ from the name
    else:
        name = name[:-2]  # remove _number from the filename

    return name


def create_pairs(claim, evidence_edus):

    pairs = []

    for edu in evidence_edus:
        claim = claim.rstrip()
        edu = edu.rstrip()
        if claim and edu:
            claim = claim.replace('\n', '')
            edu = edu.replace('\n', '')
            pairs.append((claim, edu))

    for edu1, edu2 in zip(evidence_edus, evidence_edus[1:]):
        if edu1.rstrip() and edu2.rstrip():
            edu1 = edu1.replace('\n', '')
            edu2 = edu2.replace('\n', '')
            pairs.append((edu1.rstrip(), edu2.rstrip()))

    return pairs


def write_file(file, items):
    # the lable is a placeholder
    with open(file, 'w') as fd:
        writer = csv.writer(fd, delimiter='\t', lineterminator='\n')
        for arg_id, label, arg1, arg2 in items:
            writer.writerow([arg_id, 'dummy', ' dummy', 'dummy',
                            'dummy', label, arg1, arg2])


if __name__ == '__main__':

    parser = ArgumentParser(
        description='Create data for labeling (Dagstuhl)')
    parser.add_argument(
        'evidence_files',
        help='Path to the directory containing the segmented evidences')
    parser.add_argument(
        'claim_files',
        help='Path to the directory containing the claims')
    parser.add_argument(
        'raw_files',
        help='Path to the directory containing raw files')
    args = parser.parse_args()

    evidence_files = glob.glob(
        '{}/*.txt'.format(args.evidence_files), recursive=True)

    claim_files = glob.glob(
        '{}/*.txt'.format(args.claim_files), recursive=True)

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
