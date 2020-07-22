#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: out-16 of 2019
"""

import csv
import cv2
import socket


def get_host_name():
    """
    Return the host name.
    :return str: host name.
    """
    return socket.gethostname()


def read_csv(file_name, return_the_header=False, delimiter=',', quote_char='|'):
    """
    Read a CSV file and return a list of list str.
    :param str file_name: file name which will be loaded.
    :param bool return_the_header: if should return the header (first line) (True) or not (False) (default is True).
    :param str delimiter: delimiter used to separate the rows.
    :param str quote_char: the quote char used in csv.
    :return list: the CSV content with a list (rows) of lists (columns).
    """
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar=quote_char)
        if return_the_header:
            next(reader)  # Skip header row
        data = []
        for row in reader:
            data.append(row)
    return data


def read_gray_scale_image(data_path):
    """
    Load a image as a gray-scale image.
    :param str data_path: file name which will be loaded.
    :return: the gray-scale image.
    """
    return cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)


def read_colored_image(data_path):
    """
    Load a image as a rgb image.
    :param str data_path: file name which will be loaded.
    :return: the gray-scale image.
    """
    return cv2.imread(data_path, cv2.IMREAD_COLOR)


def fill_empty_dict_fields_with_reference_data(dict_reference, dict_data):
    if dict_data is None:
        dict_data = {}
    for r in dict_reference:
        if r not in dict_data:
            if type(dict_reference[r]) is dict:
                dict_data[r] = fill_empty_dict_fields_with_reference_data(dict_reference[r], {})
            else:
                dict_data[r] = dict_reference[r]
    return dict_data