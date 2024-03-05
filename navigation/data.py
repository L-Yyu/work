import numpy as np 
import pandas as pd
import queue
import os

from tqdm import tqdm

class IMUData(object):
    def __init__(self, time, linear_acceleration, angle_vel, true_linear_acceleration, true_angle_vel,gt_quat,gt_pos_lla) -> None:
        self.imu_time = time
        self.imu_linear_acceleration = linear_acceleration
        self.imu_angle_vel =  angle_vel
        self.true_imu_linear_acceleration = true_linear_acceleration
        self.true_imu_angle_vel = true_angle_vel

        self.gt_quat = gt_quat
        self.gt_pos_lla = gt_pos_lla

    @staticmethod
    def read_imu_data(data_path)->queue.Queue:
        imu_data_queue = queue.Queue()
        imu_time_df = pd.read_csv(os.path.join(data_path,'time.csv'))
        imu_linear_acceleration_df = pd.read_csv(os.path.join(data_path,'accel-0.csv'))
        imu_angle_vel_df = pd.read_csv(os.path.join(data_path,'gyro-0.csv'))
        true_imu_linear_acceleration_df = pd.read_csv(os.path.join(data_path,'ref_accel.csv'))
        true_imu_angle_vel_df = pd.read_csv(os.path.join(data_path,'ref_gyro.csv'))

        gt_quat_df = pd.read_csv(os.path.join(data_path,'ref_att_quat.csv'))
        gt_pos_lla_df = pd.read_csv(os.path.join(data_path,'ref_pos.csv'))
        for i in tqdm(range(len(imu_time_df)), desc="reading imu data"):
            imu_data = IMUData(imu_time_df.loc[i,'time (sec)'],
                               np.array([imu_linear_acceleration_df.loc[i,'accel_x (m/s^2)'],
                                         imu_linear_acceleration_df.loc[i,'accel_y (m/s^2)'],
                                         imu_linear_acceleration_df.loc[i,'accel_z (m/s^2)']]).reshape(3,1),
                               np.array(np.deg2rad([imu_angle_vel_df.loc[i,'gyro_x (deg/s)'],
                                         imu_angle_vel_df.loc[i,'gyro_y (deg/s)'],
                                         imu_angle_vel_df.loc[i,'gyro_z (deg/s)']])).reshape(3,1),
                               np.array([true_imu_linear_acceleration_df.loc[i,'ref_accel_x (m/s^2)'],
                                         true_imu_linear_acceleration_df.loc[i,'ref_accel_y (m/s^2)'],
                                         true_imu_linear_acceleration_df.loc[i,'ref_accel_z (m/s^2)']]).reshape(3,1),
                               np.array([true_imu_angle_vel_df.loc[i,'ref_gyro_x (deg/s)'],
                                         true_imu_angle_vel_df.loc[i,'ref_gyro_y (deg/s)'],
                                         true_imu_angle_vel_df.loc[i,'ref_gyro_z (deg/s)']]).reshape(3,1),
                               np.array([gt_quat_df.loc[i,'q0 ()'],
                                        gt_quat_df.loc[i,'q1'],
                                        gt_quat_df.loc[i,'q2'],
                                        gt_quat_df.loc[i,'q3']]),
                               np.array([gt_pos_lla_df.loc[i,'ref_pos_lon (deg)'],
                                         gt_pos_lla_df.loc[i,'ref_pos_lat (deg)'],
                                         gt_pos_lla_df.loc[i,'ref_pos_alt (m)']]).reshape(3,1)
                                         )
            imu_data_queue.put(imu_data)
        print('read imu data total: ',imu_data_queue.qsize())
        return imu_data_queue
    
            

class GNSSData(object):
    def __init__(self, time, position_lla, gnss_velocity, true_position_lla, true_velocity) -> None:
        self.gnss_time = time
        self.position_lla = position_lla
        self.gnss_velocity = gnss_velocity
        self.true_position_lla = true_position_lla
        self.true_velocity = true_velocity

    @staticmethod
    def read_gnss_data(data_path)->queue.Queue:
        gnss_data_queue = queue.Queue()
        gnss_time_df = pd.read_csv(os.path.join(data_path,'gps_time.csv'))
        gnss_data_df = pd.read_csv(os.path.join(data_path,'gps-0.csv'))
        true_gnss_data_df = pd.read_csv(os.path.join(data_path,'ref_gps.csv'))
        for i in tqdm(range(len(gnss_time_df)), desc="reading gnss data"):
            gnss_data = GNSSData(gnss_time_df.loc[i,'gps_time (sec)'],
                               np.array([gnss_data_df.loc[i,'gps_lon (deg)'],
                                         gnss_data_df.loc[i,'gps_lat (deg)'],
                                         gnss_data_df.loc[i,'gps_alt (m)']]).reshape(3,1),
                               np.array([gnss_data_df.loc[i,'gps_vN (m/s)'],
                                         gnss_data_df.loc[i,'gps_vE (m/s)'],
                                         gnss_data_df.loc[i,'gps_vD (m/s)']]).reshape(3,1),
                               np.array([true_gnss_data_df.loc[i,'ref_gps_lon (deg)'],
                                            true_gnss_data_df.loc[i,'ref_gps_lat (deg)'],
                                            true_gnss_data_df.loc[i,'ref_gps_alt (m)']]).reshape(3,1),
                               np.array([true_gnss_data_df.loc[i,'ref_gps_vN (m/s)'],
                                            true_gnss_data_df.loc[i,'ref_gps_vE (m/s)'],
                                            true_gnss_data_df.loc[i,'ref_gps_vD (m/s)']]).reshape(3,1))
            gnss_data_queue.put(gnss_data)
        print('read gnss data total: ',gnss_data_queue.qsize())
        return gnss_data_queue
    
