#!/usr/bin/python3

import csv
import ntpath
import re
import string
import os
import csv
import pandas
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sn
import matplotlib.pyplot as plt


def evaluate_first_level(df):
    
    true_first = []
    predicted_first = []

    for index, row in df.iterrows():
        predicted = row['XLNet_label']
        true = row['label_corrected']
        true = true.split('.')[0]

        predicted = predicted.split('.')[0]

        true_first.append(true)
        predicted_first.append(predicted)

    return true_first, predicted_first


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
    unique_labels = df['label_corrected'].unique()


    # evaluate first level
    y_true, y_predicted = evaluate_first_level(df)
    confusion_first = confusion_matrix(y_true, y_predicted, labels=['Expansion', 'NoRel', 'Contingency', 'Temporal', 'Comparison'])
    fig = plt.figure()
    sn.heatmap(confusion_first, annot=True, fmt='g')
    plt.show()
    #print(confusion_first)
    accuracy_first = accuracy_score(y_true, y_predicted)
    print(accuracy_first)

    # evaluate second level
    y_predicted_second = df['XLNet_label']
    y_true_second = df['label_corrected']
    labels = unique_labels
    confusion_second = confusion_matrix(y_true_second, y_predicted_second, labels)
    #print(confusion_second)
    accuracy_second = accuracy_score(y_true_second, y_predicted_second)
    print(accuracy_second)


    confusion_matrix2 = pandas.crosstab(y_true_second, y_predicted_second, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix2, annot=True, fmt='g')
    plt.show()
