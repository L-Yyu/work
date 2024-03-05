import numpy as np 
import pandas as pd
import queue
import os
from tqdm import tqdm

class IMUData(object):
    def __init__(self, time, imu_angle_increment, imu_vel_increment) -> None:
        self.imu_time = time
        self.imu_angle_increment = imu_angle_increment
        self.imu_vel_increment = imu_vel_increment

    @staticmethod
    def read_imu_data(data_file)->queue.Queue:
        imu_data_queue = queue.Queue()
        with open(os.path.join(data_file), mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for iter, line in enumerate(tqdm(lines, desc="reading imu data")):
                numbers = [float(num) for num in line.split()]
                time = numbers[0]
                imu_angle_increment = np.array(numbers[1:4]).reshape(3, 1)
                imu_vel_increment = np.array(numbers[4:7]).reshape(3, 1)
                imu_data = IMUData(time, imu_angle_increment, imu_vel_increment)
                imu_data_queue.put(imu_data)
        print('read imu data total: ',imu_data_queue.qsize())
        return imu_data_queue
    
            

class GNSSData(object):
    def __init__(self, time, position_lla, gnss_std) -> None:
        self.gnss_time = time
        self.position_lla = position_lla
        self.gnss_std = gnss_std
    @staticmethod
    def read_gnss_data(data_file)->queue.Queue:
        gnss_data_queue = queue.Queue()
        with open(os.path.join(data_file), mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for iter, line in enumerate(tqdm(lines, desc="reading gnss data")):
                numbers = [float(num) for num in line.split()]
                time = numbers[0]
                pos_lla = np.array(numbers[1:4]).reshape(3, 1)
                gnss_std = np.array(numbers[4:7]).reshape(3, 1)
                gnss_data = GNSSData(time, pos_lla, gnss_std)
                gnss_data_queue.put(gnss_data)
        print('read gnss data total: ', gnss_data_queue.qsize())
        return gnss_data_queue
    
