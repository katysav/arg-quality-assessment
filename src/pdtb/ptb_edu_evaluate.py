#!/usr/bin/python3

import os
import re
from argparse import ArgumentParser
from collections import Counter
import string
import pandas as pd 
from sklearn.utils import resample, shuffle
import numpy as np
import random


def get_file_paths(data_dir, extension):
    # Get paths for all files in the given directory

    file_names = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(data_dir):
        for file in f:
            if file.endswith(extension):
                file_names.append(os.path.join(r, file))
    return file_names


def get_corresponding_file(gold_file, segmented_files):

    gf_name = os.path.basename(gold_file)
    gf_name = os.path.splitext(gold_file)[0]
    gf_number = gf_name.split('_')[-1]

    for pf in segmented_files: 
        pf_name = os.path.basename(pf)
        # Twice because the name is .txt.pipe
        pf_name = os.path.splitext(pf_name)[0]
        pf_name = os.path.splitext(pf_name)[0]
        pf_number = pf_name.split('_')[-1]
        if gf_number == pf_number:
            return pf



def evaluate(gold_content, segmented_content):

    exact_match = 0
    partial_match = 0
    false_negative = 0
    total = 0

    gold_segments = []
    edu_segments = []

    for line in gold_content:
        line = line.rstrip().split('|')
        text1 = line[24]
        text1 = text1.translate(str.maketrans('', '', string.punctuation))
        text2 = line[34]
        text2 = text2.translate(str.maketrans('', '', string.punctuation))
        gold_segments.append(text1)
        gold_segments.append(text2)

    for line in segmented_content:
        line = line.rstrip()
        line = line.translate(str.maketrans('', '', string.punctuation))
        line = re.sub(' +', ' ', line).rstrip()
        total += 1
        for item in set(gold_segments):
            if line == item:
                exact_match += 1
            elif line in item:
                print(line)
                print(item)
                partial_match += 1

    if partial_match + exact_match > len(set(gold_segments)):
        partial_match = len(set(gold_segments)) - exact_match

    return total, exact_match, partial_match


if __name__ == '__main__':

    parser = ArgumentParser(
        description='')
    parser.add_argument(
        'golden_files', help='')
    parser.add_argument(
        'segmented_files', help='')
    args = parser.parse_args()

    golden_paths = get_file_paths(args.golden_files, '.pipe')
    segmented_paths = get_file_paths(args.segmented_files, '.txt')

    exact_match = 0
    partial_match = 0
    total = 0

    for file in golden_paths:
        segmented_file = get_corresponding_file(file, segmented_paths)
        with open(file, 'r', encoding='utf-8', errors='ignore') as fd:
            gf_content = fd.readlines()
        with open(segmented_file, 'r', encoding='utf-8', errors='ignore') as fd:
            sf_content = fd.readlines()

        total_local, exact_match_local, partial_match_local = evaluate(gf_content, sf_content)
        total += total_local
        exact_match += exact_match_local
        partial_match += partial_match_local


    print(total)
    print(exact_match)
    print(partial_match)

    # accuracy: TP + TN / TP + TN + FP + FN
    # precision: TP / TP + FP
    # recall : TP / TP + FN
    # F1: 2 x precision * recall / (precision + recall)

    precision = (exact_match + partial_match) / total
    print(precision)
    recall = (exact_match + partial_match) / ((81200 - exact_match - partial_match) + exact_match + partial_match)
    print(recall)
    f1 = 2 * precision * recall / (precision + recall)
    print(f1)


# python3 ptb_edu_evaluate.py ../PDTB/v2/output/ ../PTB_wsj2.0/raw/segmented_edu/
