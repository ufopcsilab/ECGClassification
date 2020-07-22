#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: out-10 of 2019
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras

from qrs.data_handler import ECGDataHandler
from qrs.database_details.mit_database_details import MITDetails
from qrs.database_details.cybhi_database_details import CYBHiDetails
from qrs.evaluation_qrs import EvaluationQRS
from qrs.qrs_net import QRSNet


def train_pipeline():
    epochs = 2
    protocols = [1, 2]
    net_type = 'cnn'
    database = 'mit'
    details = {'shift_wave_in_ms': 30}
    data_augmentation = None
    if database == 'mit':
        database_details, input_size = MITDetails, 300
    elif database == 'cybhi':
        database_details, input_size = CYBHiDetails, 833
    else:
        raise AttributeError('The code is settled only for MIT and CYBHi databases.')
    output_size = 300
    model_name, model_plot_name, model_history_name = './qrs-1-{}-net-cnn.h5'.format(database), './qrs-{}-net-cnn.png'.format(database), './qrs-{}-net-cnn.json'.format(database)
    save_errors = False
    output_path = ''

    print('### Setting data handler ...................')
    data_handler = ECGDataHandler(input_size, output_size, database_details)

    if os.path.exists(model_name):
        print('### Loading network ........................')
        # qrs_net = keras.models.load_model(model_name)
        qrs_net = QRSNet.load_model(model_name)
    else:

        print('### Loading training data ..................')
        [train_data, train_labels, train_infos, validation_data, validation_labels, _] = data_handler.load_train_data(details=details, data_augmentation=data_augmentation)

        print('\t# Total of samples for each class: QRS ({}) and No QRS ({})'.format(sum(1 if x == data_handler.qrs_label else 0 for x in train_labels),
                                                                                     sum(1 if x == data_handler.no_qrs_label else 0 for x in train_labels)))

        for i in range(len(train_data)):
            if len(train_data[i]) != output_size:
                raise Exception('Data does not have the specific size: {} - {}'.format(len(train_data[i]), train_infos[i]))
        print('### Building network .......................')
        qrs_net = QRSNet.build(net_type)

        print('### Training network .......................')
        qrs_net, history = QRSNet.train(qrs_net, train_data, train_labels, validation_data, validation_labels, number_of_classes=2, epochs=epochs)

        print('### Saving network .........................')
        # qrs_net.save(model_name)
        QRSNet.save_model(qrs_net, model_name)

        print('### Saving network training history ........')
        with open(model_history_name, 'w') as f:
            json.dump(history.history, f)

        print('### Saving network training graph ..........')
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(np.arange(0, epochs), history.history['loss'], label='train_loss')
        plt.plot(np.arange(0, epochs), history.history['val_loss'], label='val_loss')
        plt.plot(np.arange(0, epochs), history.history['acc'], label='train_acc')
        plt.plot(np.arange(0, epochs), history.history['val_acc'], label='val_acc')
        plt.title('Training loss and Accuracy on Dataset')
        plt.xlabel('Epoch #')
        plt.ylabel('Loss Accuracy')
        plt.legend(loc='upper right')
        plt.savefig(model_plot_name)

    evaluation = EvaluationQRS(centralize_wave_during_preprocess=False, data_handler=data_handler)
    if 1 in protocols:
        print('### [Protocol 1] Loading data test .........')
        [test_data, test_labels, _] = data_handler.load_test_data(details=details)

        print('### [Protocol 1] Evaluating model trained ..')
        confusion_matrix, accuracy = evaluation.protocol_classification(model=qrs_net, data=test_data, labels=test_labels)
        print(confusion_matrix)
        print('\t# Accuracy: {:.6f}'.format(accuracy))

    if 2 in protocols:
        print('### [Protocol 2] Loading data test .........')
        [signals, r_peaks, type_r_peaks, records] = data_handler.load_raw_signals(records_type='test')

        print('### [Protocol 2] Evaluate model trained ....')
        response = evaluation.protocol_second_judge(model=qrs_net,
                                                    signals=signals,
                                                    r_peaks=r_peaks,
                                                    type_r_peaks=type_r_peaks,
                                                    records=records,
                                                    save_errors=save_errors,
                                                    output_path=output_path)

        print('\t\t      True-positive: {:9d}\t | {:9d}'.format(response['algorithm']['tp'], response['model']['tp']))
        print('\t\t     False-positive: {:9d}\t | {:9d}'.format(response['algorithm']['fp'], response['model']['fp']))
        print('\t\t     False-negative: {:9d}\t | {:9d}'.format(response['algorithm']['fn'], response['model']['fn']))
        print('\t\t      True-negative: {:9d}\t | {:9d}'.format(response['algorithm']['tn'], response['model']['tn']))
        print('\t\tPositive Prediction: {:9.5f}\t | {:9.5f}'.format(response['algorithm']['positive_prediction'], response['model']['positive_prediction']))
        print('\t\t        Sensitivity: {:9.5f}\t | {:9.5f}'.format(response['algorithm']['sensitivity'], response['model']['sensitivity']))
        print('\t\t            F-Score: {:9.5f}\t | {:9.5f}'.format(response['algorithm']['f-score'], response['model']['f-score']))


if __name__ == '__main__':
    train_pipeline()

# https://github.com/topics/ecg-qrs-detection
# https://github.com/c-labpl/qrs_detector
# https://sotabench.com/benchmarks/image-classification-on-imagenet#leaderboard
