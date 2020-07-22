#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: out-15 of 2019
"""

import matplotlib.pyplot as plt
import numpy as np
import operator

from progressbar import Bar
from progressbar import ETA
from progressbar import ProgressBar
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from qrs.ecg_detectors_algorithms.ecg_detectors import Detectors


class EvaluationQRS(object):
    """
    Class responsible to conduct the evaluation process.
    """

    def __init__(self, centralize_wave_during_preprocess, data_handler):
        """
        Constructor of the evaluation class.
        :param bool centralize_wave_during_preprocess (True): True if the peak search should be with a centralized window,
                                                              or search in a window that the end is the r-peak returned if False (default is True).
        :param DataHandler data_handler: object of type DataHandler with the details of the data.
        """
        self.__data_handler = data_handler
        self.__input_shape = (data_handler.get_output_beat_size(), 1)
        self.__centralize_wave_during_preprocess = centralize_wave_during_preprocess
        self.__smallest_distance_from_r_peaks = 0.200  # A r-peak must be at least 200 ms from each other

    def preprocess_response(self, signal, r_peaks):
        """
        Function to make all r-peaks to be the greatest point within a window.
        :param list signal: the ECG signal.
        :param list r_peaks: all the r-peaks related to the signal.
        :return list: the r-peaks processed.
        """
        size_window = int(self.__data_handler.get_data_frequency() * self.__smallest_distance_from_r_peaks)
        processed_r_peaks = []
        if self.__centralize_wave_during_preprocess:
            (begin_window, end_window) = (-size_window // 2, size_window // 2)
        else:
            (begin_window, end_window) = (-int(size_window), 0)

        for i in r_peaks:
            diff = abs(min([0, i + begin_window]))
            wave = signal[i + begin_window + diff:i + end_window + diff]
            max_index, _ = max(enumerate(wave), key=operator.itemgetter(1))
            processed_r_peaks.append(i + begin_window + max_index + diff)
        return processed_r_peaks

    @classmethod
    def filter_data_to_get_only_normal_beats(cls, r_peaks, detected_r_peaks, type_r_peaks):
        """
        Filter the data to use only the good beats. The bad beats are discarded from the ground-truth and the detected ones.
        :param list r_peaks: all the ground-truth r-peaks related to the signal.
        :param list detected_r_peaks: all the detected r-peaks by an algorithm related to the signal.
        :param type_r_peaks: description if the signal is a good one or not.
        :return list, list: the r-peaks and detected r-peaks filtered.
        """
        tolerance = 3
        ids_ground_truth = [True] * len(r_peaks)
        ids_detected_r_peaks = [True] * len(detected_r_peaks)
        count = 0
        for i in range(len(detected_r_peaks)):
            ini = detected_r_peaks[i]
            for j in range(count, len(r_peaks)):
                print('\t{:6d}\t-\t{:6d} {:6d} - {}\t'.format(i, ini, r_peaks[j], type_r_peaks[j], end=''))
                # If the r-peak is not a normal beat (N), we discard this as ground truth
                if type_r_peaks[j] != 'N':
                    ids_ground_truth[j] = False
                # True Positive - the rPeak is somewhere between the acceptable window and only accept the normal signals
                if ini - tolerance < r_peaks[j] < ini + tolerance:
                    if type_r_peaks[j] != 'N':
                        print('Eliminating', end='')
                        ids_detected_r_peaks[i] = False
                    count += 1
                    break
        print('')
        response_ground_truth = [r_peaks[x] for x in range(len(ids_ground_truth)) if ids_ground_truth[x]]
        return response_ground_truth

    def get_metrics(self, signal, r_peaks, detected_r_peaks, record, save_errors, output_path):
        """
        Return the basic metrics (True-positives/tp, True-negatives/tn, False-positives/fp and False-negatives/fn) besides
            the right and false-positive and false-negatives waves miss detected.
        :param list signal: the ECG signal.
        :param list r_peaks: all the r-peaks related to the signal.
        :param list detected_r_peaks: all the detected r-peaks by an algorithm related to the signal.
        :param str record: string with the identification of the respective signal.
        :param bool save_errors: True if save the erros, False, otherwise.
        :param str output_path: path where the figures should be saved.
        :return: the basic metrics (True-positives/tp, True-negatives/tn, False-positives/fp and False-negatives/fn) and
                    the right, false-positive and false-negatives waves.
        """
        (ii, jj, tp, tolerance) = (0, 0, 0, 3)
        (right_beats, wrong_fn_beats, wrong_fp_beats) = ([], [], [])
        id_r_peaks = [True] * len(r_peaks)
        id_detected_r_peaks = [True] * len(detected_r_peaks)

        for i in range(len(detected_r_peaks)):
            ini = detected_r_peaks[i]
            flag = False
            j = 0
            for j in range(jj, len(r_peaks)):
                r_peak = r_peaks[j]
                # True Positive - the rPeak is somewhere between the acceptable window
                if ini - tolerance < r_peak < ini + tolerance:
                    flag = True
                    right_beats.append(signal[ini - self.__data_handler.get_input_half_beat_size():ini + self.__data_handler.get_input_half_beat_size()])
                    ii = ii + 1
                    break
            if flag:  # TP
                tp = tp + 1
                id_detected_r_peaks[i] = False
                id_r_peaks[j] = False
                jj = j + 1
        fp = sum([1 if x > 0 else 0 for x in id_detected_r_peaks])
        fn = sum([1 if x else 0 for x in id_r_peaks])
        tn = len(signal) - tp + fp

        a = [x for e, x in enumerate(detected_r_peaks) if id_detected_r_peaks[e]]
        for i in range(len(a)):
            try:
                wave = signal[a[i] - self.__data_handler.get_input_half_beat_size():a[i] + self.__data_handler.get_input_half_beat_size()]
                wrong_fp_beats.append(wave)
                if save_errors:
                    plt.plot(range(self.__data_handler.get_input_size_beats), wave)
                    plt.title('IN ({}) - FP ({})'.format(record, a[i]))
                    plt.axis([0, self.__data_handler.get_input_size_beats, -1, 1])
                    plt.savefig('{}{}_FP({}).png'.format(output_path, record, a[i]))
            except:
                pass

        a = [x for x in r_peaks if x > 0]
        for i in range(len(a)):
            try:
                wave = signal[a[i] - self.__data_handler.get_input_half_beat_size():a[i] + self.__data_handler.get_input_half_beat_size()]
                wrong_fn_beats.append(wave)
                if save_errors:
                    plt.plot(range(self.__data_handler.get_input_size_beats), wave)
                    plt.title('IN ({}) - FN ({})'.format(record, a[i]))
                    plt.axis([0, self.__data_handler.get_input_size_beats, -1, 1])
                    plt.savefig('{}{}_FN({}).png'.format(output_path, record, a[i]))
            except:
                pass
        return tp, fp, fn, tn, right_beats, wrong_fn_beats, wrong_fp_beats

    @classmethod
    def _predict_class(cls, model, data):
        """
        Function to feed-forward the data over a model.
        :param keras.Sequential model: a trained model.
        :param list data: the ECG segment that will be evaluated.
        :return: the class reported by the model
        """
        predictions = model.predict(data)
        return np.argmax(predictions, axis=1)[0]

    def protocol_classification(self, model, data, labels):
        """
        Normal classification protocol.
        :param keras.Sequential model: a trained model.
        :param list data: a list with ECG segments that will be evaluated.
        :param list labels: the respective label for each ECG segment.
        :return matrix, float: the confusion matrix and accuracy.
        """

        data = np.asarray(data).astype(np.float32)
        data = data.reshape(-1, self.__input_shape[0], self.__input_shape[1])  # Reshape for the model -  should work!!
        # data = data / np.amax(data)

        predictions = []

        widgets = [Bar('>'), ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, maxval=len(data)).start()
        for i, (a, b) in enumerate(zip(data, labels)):
            a = a.reshape(1, self.__input_shape[0], self.__input_shape[1])
            predictions.append(self._predict_class(model, a))
            pbar.update(i + 1)

        pbar.finish()

        return confusion_matrix(labels, predictions), accuracy_score(labels, predictions)

    def _evaluate_qrs_detected_over_a_signal(self, model, signal, r_peaks_detected):
        """
        For each QRS complex detected, the model is applied, if it agree, the r-peak is saved.
        :param keras.Sequential model: a trained model.
        :param list signal: the ECG signal.
        :param list r_peaks_detected: all the r-peaks detected over the signal.
        :return list: all the r-peaks that the model agrees it is a r-peak.
        """
        r_peaks_confirmed = []
        preprocess_function = self.__data_handler.get_preprocess_function()
        for i in r_peaks_detected:
            wave = preprocess_function(signal[i - self.__data_handler.get_input_half_beat_size():i + self.__data_handler.get_input_half_beat_size()])
            if len(wave) != self.__data_handler.get_output_beat_size():
                input('waiting.....')
            wave = np.asarray(wave)
            wave = wave.reshape(1, self.__input_shape[0], self.__input_shape[1])
            prediction = self._predict_class(model, wave)
            if prediction == self.__data_handler.qrs_label:
                r_peaks_confirmed.append(i)
        return r_peaks_confirmed

    def protocol_second_judge(self, model, signals, r_peaks, type_r_peaks, records, save_errors=False, output_path=''):
        """
        :param keras.Sequential model: a trained model.
        :param list signals: a list with ECG signals that will be evaluated.
        :param list r_peaks: the respective labels for each ECG signal.
        :param type_r_peaks: the respective label types for each r-peaks for each signal.
        :param list records: strings with the identification of each signal.
        :param bool save_errors: True if save the erros, False, otherwise.
        :param str output_path: path where the figures should be saved.
        :return dict: True-positives/tp, True-negatives/tn, False-positives/fp, False-negatives/fn,
                     positive-prediction, sensitivity and f-score of the model and qrs_jetson detector.
        """
        detector = Detectors(self.__data_handler.get_data_frequency())
        (tps_algorithm, fps_algorithm, fns_algorithm, tns_algorithm) = [0, 0, 0, 0]
        (tps_model, fps_model, fns_model, tns_model, right_beats, wrong_fn_beats, wrong_fp_beats) = [0, 0, 0, 0, [], [], []]

        for signal, r_peak, type_r_peak, record in zip(signals, r_peaks, type_r_peaks, records):
            # type_r_peak = [type_r_peak[e] for e in range(len(r_peak)) if r_peak[e] > data_frequency]
            r_peak = [x for x in r_peak if self.__data_handler.get_data_frequency() < x < len(signal) - self.__data_handler.get_data_frequency()]
            detected_peaks = self.preprocess_response(signal=signal,
                                                      r_peaks=[x for x in detector.pan_tompkins_detector(signal)
                                                               if (self.__data_handler.get_data_frequency() < x < len(signal) - self.__data_handler.get_data_frequency())])
            tp1, fp1, fn1, tn1, _, _, _ = self.get_metrics(signal=signal,
                                                           r_peaks=r_peak,
                                                           detected_r_peaks=detected_peaks,
                                                           record=record,
                                                           save_errors=save_errors,
                                                           output_path=output_path)
            tps_algorithm += tp1
            fps_algorithm += fp1
            fns_algorithm += fn1
            tns_algorithm += tn1

            r_peaks_evaluated = self._evaluate_qrs_detected_over_a_signal(model=model,
                                                                          signal=signal,
                                                                          r_peaks_detected=detected_peaks)
            tp2, fp2, fn2, tn2, right_beats, wrong_fn_beats, wrong_fp_beats = self.get_metrics(signal=signal,
                                                                                               r_peaks=r_peak,
                                                                                               detected_r_peaks=r_peaks_evaluated,
                                                                                               record=record,
                                                                                               save_errors=save_errors,
                                                                                               output_path=output_path)
            tps_model += tp2
            fps_model += fp2
            fns_model += fn2
            tns_model += tn2

            print('\t# Record: {}'.format(record))
            print('\t\t      True-positive: {:9d}\t | {:9d}'.format(tp1, tp2))
            print('\t\t     False-positive: {:9d}\t | {:9d}'.format(fp1, fp2))
            print('\t\t     False-negative: {:9d}\t | {:9d}'.format(fn1, fn2))
            print('\t\t      True-negative: {:9d}\t | {:9d}'.format(tn1, tn2))
            print('\t\tPositive Prediction: {:9.5f}\t | {:9.5f}'.format(tp1 / max([1, (tp1 + fp1)]), tp2 / max([1, (tp2 + fp2)])))
            print('\t\t        Sensitivity: {:9.5f}\t | {:9.5f}'.format(tp1 / max([1, (tp1 + fn1)]), tp2 / max([1, (tp2 + fn2)])))

        return {'algorithm': {'fp': fps_algorithm,
                              'fn': fns_algorithm,
                              'tp': tps_algorithm,
                              'tn': tns_algorithm,
                              'positive_prediction': tps_algorithm / max([1, (tps_algorithm + fps_algorithm)]),
                              'sensitivity': tps_algorithm / max([1, (tps_algorithm + fns_algorithm)]),
                              'f-score': 2 * (tps_algorithm / max([1, (tps_algorithm + fps_algorithm)]) * tps_algorithm / max([1, (tps_algorithm + fns_algorithm)])) /
                                             (tps_algorithm / max([1, (tps_algorithm + fps_algorithm)]) + tps_algorithm / max([1, (tps_algorithm + fns_algorithm)]))},
                'model': {'fp': fps_model,
                          'fn': fns_model,
                          'tp': tps_model,
                          'tn': tns_model,
                          'positive_prediction': tps_model / max([1, (tps_model + fps_model)]),
                          'sensitivity': tps_model / max([1, (tps_model + fns_model)]),
                          'f-score': 2 * (tps_model / max([1, (tps_model + fps_model)]) * tps_model / max([1, (tps_model + fns_model)])) /
                                     max([1, (tps_model / max([1, (tps_model + fps_model)]) + tps_model / max([1, (tps_model + fns_model)]))])}
                }
