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


'''
Preprocess Dagstuhl for XLNet fine-tuning

Usage: python3 dagstuhl_preprocessing.py -evidence_directory -claim_directory -raw_files

Example: python3 dagstuhl_preprocessing.py ../dagstuhl-15512-argquality-corpus-v2/dagstuhl-15512-argquality-corpus-v2/dagstuhl_segmenter_output_corrected/ ../dagstuhl-15512-argquality-corpus-v2/dagstuhl-15512-argquality-corpus-v2/claims/ ../dagstuhl-15512-argquality-corpus-v2/dagstuhl-15512-argquality-corpus-v2/dagstuhl-15512-argquality-corpus-unannotated/
'''

def drop_without_labels(df):
    ''' Take into account only with quality labels '''

    valid_ids = ['arg167721', 'arg168778', 'arg168795', 'arg168790', 'arg168836', 'arg580842', 'arg168803', 'arg168824', 'arg168771', 'arg168835', 'arg168834', 'arg168822', 'arg168780', 'arg168804', 'arg168774', 'arg168830', 'arg168807', 'arg168779', 'arg168818', 'arg168801', 'arg336563', 'arg334884', 'arg335047', 'arg336222', 'arg335090', 'arg335089', 'arg334924', 'arg334959', 'arg334973', 'arg335097', 'arg335095', 'arg336204', 'arg335564', 'arg335129', 'arg334934', 'arg335091', 'arg336179', 'arg334943', 'arg334968', 'arg335030', '13279', '13275', '30319', '76937', '75146', '13259', '70668', '13260', '13270', '54474', '63512', '1661', '74039', '28415', '1683', '37929', '1633', '28498', '1641', '69016', 'arg126375', 'arg145468', 'arg122478', 'arg118660', 'arg108959', 'arg118518', 'arg132199', 'arg649666', 'arg108968', 'arg596217', 'arg126374', 'arg106426', 'arg623493', 'arg132483', 'arg123380', 'arg106091', 'arg116888', 'arg260216', 'arg126378', '817', '80854', '578317615', '79918', '800', '822', '823', '794', '814', '12580', '70855', '12589', '28068', '71559', '12567', '12592', 'arg33141', 'arg33099', 'arg33135', 'arg33129', 'arg33162', 'arg33060', 'arg33082', 'arg33119', 'arg33089', 'arg33075', 'arg33070', 'arg33127', 'arg33115', 'arg33118', 'arg33143', 'arg33086', 'arg33069', 'arg33126', 'arg33105', 'arg33123', '390', '458', '460', '477', '468', '415', '389', '76359', '33757', '405', '33187', '12371', '12367', '65191', '12365', '12388', '12383', '12380', '65125', '12414', '12431', '12466', '69708', '31527', '42624', '12421', '12402', '76796', '12430', '530', '73625', '30529', '76758', '80510', '46238', '33506', '67160', '63084', 'arg561428', 'arg399268', 'arg585674', 'arg376774', 'arg470033', 'arg399267', 'arg479199', 'arg399270', 'arg363603', 'arg135702', 'arg585714', 'arg213555', 'arg159445', 'arg497712', 'arg135637', 'arg317750', 'arg213296', 'arg223675', 'arg135648', 'arg33284', 'arg33282', 'arg33280', 'arg33293', 'arg33342', 'arg33339', 'arg33289', 'arg33272', 'arg33261', 'arg33323', 'arg33298', 'arg33288', 'arg33319', 'arg33226', 'arg33233', 'arg33270', 'arg33285', 'arg33341', 'arg33243', 'arg33279', 'arg260899', 'arg37974', 'arg54258', 'arg345997', 'arg107445', 'arg7751', 'arg107456', 'arg148559', 'arg213068', 'arg230691', 'arg216634', 'arg312577', 'arg54267', 'arg485419', 'arg39274', 'arg271353', 'arg203471', 'arg245105', 'arg202607', 'arg110967', 'arg561672', 'arg198376', 'arg234411', 'arg200133', 'arg203869', 'arg198417', 'arg203922', 'arg660921', 'arg212151', 'arg542561', 'arg231770', 'arg238471', 'arg198954', 'arg336277', 'arg238468', 'arg251309', 'arg200706', 'arg238473', 'arg199549', 'arg199159', 'arg35584', 'arg35726', 'arg35650', 'arg35698', 'arg35700', 'arg35676', 'arg35635', 'arg35615', 'arg35619', 'arg35720', 'arg35718', 'arg35758', 'arg35705', 'arg35622', 'arg35748', 'arg35591', 'arg35669', 'arg35706', 'arg35632', 'arg35616', 'arg636360', 'arg289787', 'arg229317', 'arg644073', 'arg244440', 'arg240623', 'arg439197', 'arg231491', 'arg240625', 'arg229484', 'arg229241', 'arg246334', 'arg575745', 'arg270902', 'arg231620', '2137504198', '29393', '1191878965', '71943', '39342', '27868', '71821', '29278', '54187', '79734', '13993', '14000', '54763', '14007', '13989', '14003', '73596', '14002', '13998', '37675', 'arg219220', 'arg219261', 'arg219227', 'arg219254', 'arg219204', 'arg219233', 'arg219213', 'arg219206', 'arg219225', 'arg219245', 'arg219250', 'arg219244', 'arg219268', 'arg219258', 'arg219293', 'arg219259']

    for index, row in df.iterrows():
        if row['arg_id'] not in valid_ids:
            df = df.drop(index)

    return df


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

    df['true_first_level'] = true_first
    df['predicted_first_level'] = predicted_first

    return df


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
    print(set(unique_labels))

    drop_without_labels(df)

    # evaluate first level
    df = evaluate_first_level(df)
    y_true_first = df['true_first_level']
    y_predicted_first = df['predicted_first_level']
    confusion_matrix1 = pandas.crosstab(y_true_first, y_predicted_first, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix1, annot=True, fmt='g')
    plt.show()
    #print(confusion_first)
    accuracy_first = accuracy_score(y_true_first, y_predicted_first)
    print(accuracy_first)

    # evaluate second level
    y_predicted_second = df['XLNet_label']
    y_true_second = df['label_corrected']
    labels = unique_labels
    print(y_true_second)
    confusion_second = confusion_matrix(y_true_second, y_predicted_second, labels)
    accuracy_second = accuracy_score(y_true_second, y_predicted_second)
    print(accuracy_second)


    confusion_matrix2 = pandas.crosstab(y_true_second, y_predicted_second, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix2, annot=True, fmt='g')
    plt.show()
