#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: out-15 of 2019
"""

from .base_database_details import BaseDatabaseDetails
from ..paths import DataPaths


class CYBHiDetails(BaseDatabaseDetails):

    @classmethod
    def get_base_data_path(cls):
        return DataPaths.get_cybhi_base_data_path()

    @classmethod
    def get_test_records(cls):
        return ['20120416-AL-A0-35.txt.mat',  '20120416-FO-A0-35.txt.mat',  '20120416-JP-A0-35.txt.mat', '20120416-JS-A0-35.txt.mat',
                '20120416-RAA-A0-35.txt.mat', '20120416-RL-A0-35.txt.mat',  '20120419-AC-A0-35.txt.mat', '20120419-CB-A0-35.txt.mat',
                '20120419-FP-A0-35.txt.mat',  '20120419-JCA-A0-35.txt.mat', '20120419-MQ-A0-35.txt.mat', '20120419-RR-A0-35.txt.mat',
                '20120423-AFS-A0-35.txt.mat', '20120423-JSA-A0-35.txt.mat', '20120423-RD-A0-35.txt.mat', '20120423-SR-A0-35.txt.mat',
                '20120424-AG-A0-35.txt.mat',  '20120424-AR-A0-35.txt.mat',  '20120424-MGA-A0-35.txt.mat', '20120427-MMJ-A0-35.txt.mat',
                '20120430-ACA-A0-35.txt.mat', '20120430-ARL-A0-35.txt.mat', '20120430-CSR-A0-35.txt.mat', '20120430-DS-A0-35.txt.mat',
                '20120430-FM-A0-35.txt.mat',  '20120430-IC-A0-35.txt.mat',  '20120430-JB-A0-35.txt.mat', '20120430-JCC-A0-35.txt.mat',
                '20120430-JM-A0-35.txt.mat',  '20120430-JPA-A0-35.txt.mat', '20120430-MC-A0-35.txt.mat', '20120430-MJR-A0-35.txt.mat',
                '20120430-MP-A0-35.txt.mat',  '20120430-PES-A0-35.txt.mat', '20120430-RA-A0-35.txt.mat', '20120430-RRA-A0-35.txt.mat',
                '20120430-TC-A0-35.txt.mat',  '20120430-TV-A0-35.txt.mat',  '20120502-ABD-A0-35.txt.mat', '20120502-ARA-A0-35.txt.mat',
                '20120502-DB-A0-35.txt.mat',  '20120502-DC-A0-35.txt.mat',  '20120502-GF-A0-35.txt.mat', '20120502-HF-A0-35.txt.mat',
                '20120502-IB-A0-35.txt.mat',  '20120502-JL-A0-35.txt.mat',  '20120502-JN-A0-35.txt.mat', '20120502-JV-A0-35.txt.mat',
                '20120502-MA-A0-35.txt.mat',  '20120502-MB-A0-35.txt.mat',  '20120502-MBA-A0-35.txt.mat', '20120502-PMA-A0-35.txt.mat',
                '20120502-SF-A0-35.txt.mat',  '20120502-TF-A0-35.txt.mat',  '20120502-VM-A0-35.txt.mat', '20120502-VO-A0-35.txt.mat',
                '20120504-CF-A0-35.txt.mat']

    @classmethod
    def get_train_records(cls):
        return ['20120106-AA-A0-35.txt.mat',  '20120106-AL-A0-35.txt.mat',  '20120106-FO-A0-35.txt.mat',  '20120106-FP-A0-35.txt.mat',
                '20120106-JP-A0-35.txt.mat',  '20120106-JS-A0-35.txt.mat',  '20120106-MB-A0-35.txt.mat',  '20120106-MQ-A0-35.txt.mat',
                '20120106-RAA-A0-35.txt.mat', '20120106-RF-A0-35.txt.mat',  '20120111-AG-A0-35.txt.mat',  '20120111-DC-A0-35.txt.mat',
                '20120111-DS-A0-35.txt.mat',  '20120111-FM-A0-35.txt.mat',  '20120111-JCC-A0-35.txt.mat', '20120111-JPA-A0-35.txt.mat',
                '20120111-MGA-A0-35.txt.mat', '20120111-MP-A0-35.txt.mat',  '20120111-RA-A0-35.txt.mat',  '20120111-RD-A0-35.txt.mat',
                '20120111-TC-A0-35.txt.mat',  '20120111-TV-A0-35.txt.mat',  '20120112-ABD-A0-35.txt.mat', '20120112-ARL-A0-35.txt.mat',
                '20120112-CSR-A0-35.txt.mat', '20120112-IC-A0-35.txt.mat',  '20120112-JB-A0-35.txt.mat',  '20120112-MC-A0-35.txt.mat',
                '20120112-MJR-A0-35.txt.mat', '20120112-RR-A0-35.txt.mat',  '20120112-RRA-A0-35.txt.mat', '20120113-AR-A0-35.txt.mat',
                '20120113-JA-A0-35.txt.mat',  '20120113-MMJ-A0-35.txt.mat', '20120113-PM-A0-35.txt.mat',  '20120113-PMA-A0-35.txt.mat',
                '20120113-SF-A0-35.txt.mat',  '20120113-VO-A0-35.txt.mat',  '20120118-AC-A0-35.txt.mat',  '20120118-ACA-A0-35.txt.mat',
                '20120118-AFS-A0-35.txt.mat', '20120118-CB-A0-35.txt.mat',  '20120118-JCA-A0-35.txt.mat', '20120118-JL-A0-35.txt.mat',
                '20120118-JM-A0-35.txt.mat',  '20120118-JN-A0-35.txt.mat',  '20120118-JSA-A0-35.txt.mat', '20120118-MA-A0-35.txt.mat',
                '20120120-ARA-A0-35.txt.mat', '20120120-DB-A0-35.txt.mat',  '20120120-GF-A0-35.txt.mat',  '20120120-HF-A0-35.txt.mat',
                '20120120-MBA-A0-35.txt.mat', '20120120-SR-A0-35.txt.mat',  '20120120-TF-A0-35.txt.mat',  '20120120-VM-A0-35.txt.mat',
                '20120416-AA-A0-35.txt.mat']
        # return ['20120106-RF-A0-35.txt.mat']

    @classmethod
    def get_data_frequency(cls):
        return 1000
