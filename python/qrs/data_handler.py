#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: out-15 of 2019
"""
import copy

import numpy as np

from scipy import interpolate as interp

from qrs.utils import read_csv, fill_empty_dict_fields_with_reference_data


class ECGDataHandler(object):
    """
    Class responsible to load ECG data.
    """

    def __init__(self, input_beat_size, output_beat_size, database_details):
        """
        Constructor of the data handler.
        :param int input_beat_size: amount of samples get from a signal.
        :param int output_beat_size: amount of samples that will be returned.
        :param BaseDatabaseDetails database_details: class with the database details, that inherits from BaseDatabaseDetails class.
        """
        self.qrs_label = 1
        self.no_qrs_label = 0
        self.__input_beat_size = input_beat_size
        self.__input_half_beat_size = input_beat_size // 2
        self.__output_beat_size = output_beat_size
        self.__database_details = database_details
        self.__data_frequency = database_details.get_data_frequency()
        self.__proportion_frequency_and_ms = self.__data_frequency / 1000  # relation between frequency and a second (1000ms)

        self.__data_augmentation = {'p-wave':  # p-wave data augmentation details
                                        {'ms': 375,  # how many milliseconds will be attenuated (from the beginning of the signal)
                                         'attenuation': 0.3},  # how much of the p-wave will be attenuated
                                    't-wave':  # t-wave data augmentation details
                                        {'ms': 375,  # how many milliseconds will be attenuated (from the beginning of the signal)
                                         'attenuation': 0.3},  # how much of the t-wave will be attenuated
                                    'gain': [.6, .8]  # the gains that will be applied to the segment
                                   }

        self.__details = {'shift_wave_in_ms': 14,  # how many milliseconds will be shifted to get another sample (only for negative samples)
                          'shift_to_avoid_r_peak': 139,  # how many milliseconds need to be shifted to avoid the r-peak
                          'train-portion': 0.70}  # the proportion that must be used to train the model and the proportion for validation

        def interpolate_signal(big_signal):
            array_interp = interp.interp1d(range(len(big_signal)), big_signal)
            return array_interp(np.linspace(0, len(big_signal) - 1, output_beat_size))

        self.__preprocess_function = lambda a: a if input_beat_size == output_beat_size else interpolate_signal(a)

    def get_preprocess_function(self):
        return self.__preprocess_function

    def get_data_frequency(self):
        return self.__data_frequency

    def get_input_beat_size(self):
        return self.__input_beat_size

    def get_input_half_beat_size(self):
        return self.__input_half_beat_size

    def get_output_beat_size(self):
        return self.__output_beat_size

    def get_database_details(self):
        return self.__database_details

    def _get_positive_samples(self, signal, r_peak, data_augmentation, details):
        """
        Get positive samples around a specific r-peak.
        :param list signal: ECG signal.
        :param int r_peak: the r-peak used to get the positive samples.
        :param dict data_augmentation: dictionary with the details about the data augmentation that will be applied.
        :param dict details: dictionary with the details around the data.
        :return: four lists: one with the waves, one with labels, the waves info and the last one is the records name.
        """
        waves, labels, infos = [], [], []
        for i in range(-details['shift_wave_in_ms'] * 3, details['shift_wave_in_ms'] * 3 + 1, details['shift_wave_in_ms']):
            try:
                wave = self.get_wave(signal, r_peak + i)
                if len(wave) == self.__output_beat_size:
                    waves.append(wave)
                    labels.append(self.qrs_label)
                    infos.append('QRS - Shifted {} ({})'.format(i, r_peak - i))
            except:
                pass
        if data_augmentation:
            wave = self.get_wave(signal, r_peak)
            if len(wave) == self.__input_beat_size:
                if 'p-wave' in data_augmentation:
                    attenuate = round(self.__proportion_frequency_and_ms * data_augmentation['p-wave']['ms'])
                    wave = copy.deepcopy(wave)
                    wave[1:attenuate] = [i * data_augmentation['p-wave']['attenuation'] for i in wave[1:attenuate]]
                    waves.append(wave)
                    labels.append(self.qrs_label)
                    infos.append('QRS - P-wave attenuated ({})'.format(r_peak))
                if 't-wave' in data_augmentation:
                    attenuate = round(self.__proportion_frequency_and_ms * data_augmentation['t-wave']['ms'])
                    wave = copy.deepcopy(wave)
                    wave[self.__input_beat_size - attenuate:] = [i * data_augmentation['t-wave']['attenuation'] for i in wave[self.__input_beat_size - attenuate:]]
                    waves.append(wave)
                    labels.append(self.qrs_label)
                    infos.append('QRS - T-wave attenuated ({})'.format(r_peak))
                if 'gain' in data_augmentation:
                    for gain in data_augmentation['gain']:
                        wave = [i * gain for i in copy.deepcopy(wave)]
                        waves.append(wave)
                        labels.append(self.qrs_label)
                        infos.append('QRS - Gain of {}% ({})'.format(gain, r_peak))
        return waves, labels, infos

    def _get_negative_samples(self, signal, first_r_peak, second_r_peak, details):
        """
        Get negative samples between two specific r-peaks.
        :param list signal: ECG signal.
        :param int first_r_peak: the r-peak from which the negative samples acquisition starts.
        :param int second_r_peak: the r-peak from which the negative samples acquisition stops.
        :param dict details: dictionary with the details around the data.
        :return: four lists: one with the waves, one with labels, the waves info and the last one is the records name.
        """
        waves, labels, infos = [], [], []
        samples_to_avoid = details['shift_to_avoid_r_peak']  # 140ms
        first_r_peak += samples_to_avoid
        second_r_peak -= samples_to_avoid
        for i in range(first_r_peak, second_r_peak + 1, details['shift_wave_in_ms']):
            wave = self.get_wave(signal, i)
            if len(wave) == self.__output_beat_size:
                waves.append(wave)
                labels.append(self.no_qrs_label)
                infos.append('Without QRS ({})'.format(i + self.__input_half_beat_size))
        return waves, labels, infos

    def _get_waves(self, signal, r_peaks, type_r_peaks, data_augmentation, details):
        """
        Get positive and negative samples around all r-peaks available.
        :param list signal: ECG signal.
        :param list r_peaks: all the r-peaks related to the signal.
        :param type_r_peaks: the type of each r-peak (in MIT database only the normal signals are used).
        :param dict data_augmentation: dictionary with the details about the data augmentation that will be applied.
        :param dict details: dictionary with the details around the data.
        :return: four lists: one with the waves, one with labels, the waves info and the last one is the records name.
        """
        waves, labels, infos = [], [], []
        for i in range(len(r_peaks) - 1):
            if r_peaks[i] > self.__input_half_beat_size and type_r_peaks[i] == 'N':
                [r_waves, r_labels, r_infos] = self._get_positive_samples(signal, r_peaks[i], data_augmentation, details)
                waves += r_waves
                infos += r_infos
                labels += r_labels

                [r_waves, r_labels, r_infos] = self._get_negative_samples(signal, r_peaks[i], r_peaks[i + 1], details)
                waves += r_waves
                infos += r_infos
                labels += r_labels

        return waves, labels, infos

    def get_wave(self, signal, center):
        """
        Get a window of a signal centered in a specific point.
        :param list signal: ECG signal.
        :param int center: the center of the window.
        :return list: a list with part of the signal.
        """
        wave = signal[center - self.__input_half_beat_size:center + self.__input_half_beat_size]
        return self.__preprocess_function(wave)

    def load_raw_signals(self, records_type='train'):
        """
        Load all data of a database without any pre-processing.
        :param str records_type: 'train' for the training data and 'test' for the testing data.
        :return: four lists: one with the waves, one with labels, the waves info and the last one is the records name.
        """
        if records_type == 'train':
            records = self.__database_details.get_train_records()
        elif records_type == 'test':
            records = self.__database_details.get_test_records()
        else:
            raise AttributeError('The only valid records type are "train" and "test".')
        signals, r_peaks, type_r_signals = [], [], []
        print('\t# Loading records.......................')
        for record in records:
            signals.append([float(i[0]) for i in read_csv('{}{}-signal.txt'.format(self.__database_details.get_base_data_path(), record))])
            r_peaks.append([int(i[0]) for i in read_csv('{}{}-rpeaks.txt'.format(self.__database_details.get_base_data_path(), record))])
            type_r_signals.append([i[0] for i in read_csv('{}{}-type.txt'.format(self.__database_details.get_base_data_path(), record), quote_char='[')])
        return signals, r_peaks, type_r_signals, records

    def load_train_data(self, details=None, data_augmentation=None):
        """
        Load all training data of a database and return it for training.
        :param dict details: dictionary with the details around the data.
        :param dict data_augmentation: dictionary with the details about the data augmentation that will be applied.
        :return: four lists: one with the waves, one with labels, the waves info and the last one is the records name.
        """
        data_augmentation = fill_empty_dict_fields_with_reference_data(self.__data_augmentation, data_augmentation)
        details = fill_empty_dict_fields_with_reference_data(self.__details, details)

        details['shift_wave_in_ms'] = int(details['shift_wave_in_ms'] * self.__proportion_frequency_and_ms)
        details['shift_to_avoid_r_peak'] = int(details['shift_to_avoid_r_peak'] * self.__proportion_frequency_and_ms)
        train_waves, train_infos, train_labels, validation_waves, validation_infos, validation_labels = [], [], [], [], [], []

        (signals, all_r_peaks, all_type_r_peaks, records) = self.load_raw_signals(records_type='train')
        for signal, r_peaks, type_r_peaks, record in zip(signals, all_r_peaks, all_type_r_peaks, records):
            print('\t# Record: {}'.format(record))
            train_portion = details['train-portion'] * len(signal)
            train_r_peaks = [x for x in r_peaks if x <= train_portion]
            train_type_r_peaks = [type_r_peaks[e] for e in range(len(type_r_peaks)) if r_peaks[e] <= train_portion]
            [r_waves, r_labels, r_infos] = self._get_waves(signal=signal,
                                                           r_peaks=train_r_peaks,
                                                           type_r_peaks=train_type_r_peaks,
                                                           data_augmentation=data_augmentation,
                                                           details=details)
            train_waves += r_waves
            train_infos += r_infos
            train_labels += r_labels

            validation_r_peaks = [train_r_peaks[-1]] + [x for x in r_peaks if x > train_portion] + [len(signal)]
            validation_type_r_peaks = [train_type_r_peaks[-1]] + [type_r_peaks[e] for e in range(len(type_r_peaks)) if r_peaks[e] > train_portion]
            [r_waves, r_labels, r_infos] = self._get_waves(signal=signal,
                                                           r_peaks=validation_r_peaks,
                                                           type_r_peaks=validation_type_r_peaks,
                                                           data_augmentation=data_augmentation,
                                                           details=details)
            validation_waves += r_waves
            validation_infos += r_infos
            validation_labels += r_labels

        return train_waves, train_labels, train_infos, validation_waves, validation_labels, validation_infos

    def load_test_data(self, details=None):
        """
        Load all testing data of a database and return it for training.
        :param dict details: dictionary with the details around the data.
        :return: four lists: one with the waves, one with labels, the waves info and the last one is the records name.
        """
        fill_empty_dict_fields_with_reference_data(self.__details, details)

        details['shift_wave_in_ms'] = int(details['shift_wave_in_ms'] * self.__proportion_frequency_and_ms)
        details['shift_to_avoid_r_peak'] = int(details['shift_to_avoid_r_peak'] * self.__proportion_frequency_and_ms)
        waves, infos, labels = [], [], []
        signals, all_r_peaks, all_type_r_peaks, records = self.load_raw_signals(records_type='test')

        for (signal, r_peaks, type_r_peaks, record) in zip(signals, all_r_peaks, all_type_r_peaks, records):

            print('\t# Record: {}'.format(record))
            [r_waves, r_labels, r_infos] = self._get_waves(signal=signal,
                                                           r_peaks=r_peaks,
                                                           type_r_peaks=type_r_peaks,
                                                           data_augmentation={},
                                                           details=details)
            waves += r_waves
            infos += r_infos
            labels += r_labels

        return waves, labels, infos
