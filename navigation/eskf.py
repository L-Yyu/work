import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import queue
import os
import yaml
from data import IMUData, GNSSData
from tools import *

DIM_STATE = 15
DIM_STATE_NOISE = 6
DIM_MEASUREMENT = 3
DIM_MEASUREMENT_NOISE = 3

INDEX_STATE_POSI = 0
INDEX_STATE_VEL = 3
INDEX_STATE_ORI = 6
INDEX_STATE_GYRO_BIAS = 9
INDEX_STATE_ACC_BIAS = 12
INDEX_MEASUREMENT_POSI = 0

D2R = np.pi/180.0

class ESKF(object):

    def __init__(self, config) -> None:

        self.X = np.zeros((DIM_STATE, 1))
        self.Y = np.zeros((DIM_MEASUREMENT, 1))
        self.F = np.zeros((DIM_STATE, DIM_STATE))
        self.B = np.zeros((DIM_STATE, DIM_STATE_NOISE))
        self.Q = np.zeros((DIM_STATE_NOISE, DIM_STATE_NOISE))
        self.P = np.zeros((DIM_STATE, DIM_STATE))
        self.K = np.zeros((DIM_STATE, DIM_MEASUREMENT))
        self.C = np.identity(DIM_MEASUREMENT_NOISE)
        self.G = np.zeros((DIM_MEASUREMENT, DIM_STATE))
        self.G[INDEX_MEASUREMENT_POSI:INDEX_MEASUREMENT_POSI+3, INDEX_MEASUREMENT_POSI:INDEX_MEASUREMENT_POSI+3] = np.eye(3)
        self.R = np.zeros((DIM_MEASUREMENT, DIM_MEASUREMENT))

        self.Ft = np.zeros((DIM_STATE, DIM_STATE))

        self.velocity_ = np.zeros((3, 1))
        self.quat_ = np.zeros((4, 1))
        self.rotation_matrix_ = np.zeros((3, 3))
        self.pos_ = np.zeros((3, 1))
        self.ref_pos_lla = np.array([config['ref_longitude'],config['ref_latitude'],config['ref_altitude']]).reshape(3,1)

        self.gyro_bias_ = np.zeros((3, 1))
        self.accel_bias_ = np.zeros((3, 1))
        self.g_ = np.array([[0], [0], [-config['earth_gravity']]]).reshape((3,1))

        self.curr_gnss_data:GNSSData
        self.last_imu_data:IMUData
        self.curr_imu_data:IMUData
        self.imu_data_queue = queue.Queue()
        self.gnss_data_queue = queue.Queue()
        # init P
        self.SetP(config['init_position_std'], config['init_velocity_std'], 
                  config['init_rotation_std'], config['init_gyro_bias_std'], config['init_accelerometer_bias_std'])
        # 输入(imu bias)的方差
        self.SetQ(config['gyro_noise_std'], config['accelerometer_noise_std'])
        # 观测(gnss)的方差
        self.SetR(config['gps_position_x_std'], config['gps_position_y_std'], config['gps_position_z_std'])

    def SetQ(self, gyro_noise, accel_noise):
        self.Q = np.zeros((DIM_STATE_NOISE, DIM_STATE_NOISE))
        self.Q[0:3, 0:3] = np.eye(3) * gyro_noise * gyro_noise
        self.Q[3:6, 3:6] = np.eye(3) * accel_noise * accel_noise

    def SetR(self, position_x_std, position_y_std, position_z_std):
        self.R = np.zeros((DIM_MEASUREMENT_NOISE, DIM_MEASUREMENT_NOISE))
        self.R[0, 0] =  position_x_std * position_x_std
        self.R[1, 1] =  position_y_std * position_y_std
        self.R[2, 2] =  position_z_std * position_z_std

    def SetP(self, posi_noise, velocity_noise, ori_noise, gyro_noise, accel_noise):
        self.P = np.zeros((DIM_STATE, DIM_STATE))
        self.P[INDEX_STATE_POSI:INDEX_STATE_POSI+3, INDEX_STATE_POSI:INDEX_STATE_POSI+3] = np.eye(3) * np.array(posi_noise) * np.array(posi_noise)
        self.P[INDEX_STATE_VEL:INDEX_STATE_VEL+3, INDEX_STATE_VEL:INDEX_STATE_VEL+3] = np.eye(3) * np.array(velocity_noise) * np.array(velocity_noise)
        self.P[INDEX_STATE_ORI:INDEX_STATE_ORI+3, INDEX_STATE_ORI:INDEX_STATE_ORI+3] = np.eye(3) * np.array(ori_noise) * np.array(ori_noise)
        self.P[INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3, INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3] = np.eye(3) * np.array(gyro_noise) * np.array(gyro_noise)
        self.P[INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3, INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3] = np.eye(3) * np.array(accel_noise) * np.array(accel_noise)


    def Init(self, data_path:str):
        self.imu_data_queue = IMUData.read_imu_data(data_path)
        self.gnss_data_queue = GNSSData.read_gnss_data(data_path)

        self.last_imu_data = self.imu_data_queue.get()
        self.curr_gnss_data = self.gnss_data_queue.get()

        # self.velocity_= np.array([[0], [0], [0]])
        self.velocity_ = self.curr_gnss_data.true_velocity

        self.quat_ = euler2quaternion(np.array([0, 0, 0])).reshape(4, 1)
        self.rotation_matrix_ = euler2rot(np.array([0, 0, 0]))

        # self.pos_ = np.array([0,0,0]).reshape(3,1)
        self.pos_ = lla2ned(self.curr_gnss_data.position_lla, self.ref_pos_lla)

    def Predict(self):
        self.curr_imu_data = self.imu_data_queue.get()
        curr_imu_data = self.curr_imu_data
        last_imu_data = self.last_imu_data
        delta_t = curr_imu_data.imu_time - last_imu_data.imu_time
        curr_accel = self.rotation_matrix_ @ curr_imu_data.imu_linear_acceleration
        # 姿态更新
        unbias_gyro_0 = last_imu_data.imu_angle_vel - self.gyro_bias_
        unbias_gyro_1 = curr_imu_data.imu_angle_vel - self.gyro_bias_
        delta_theta = 0.5 * (unbias_gyro_0 + unbias_gyro_1) * delta_t
        rotation_vector = delta_theta
        # 基于旋转矩阵的更新算法
        last_rotation_matrix = self.rotation_matrix_
        delta_rotation_matrix = rv2rot(rotation_vector)
        curr_rotation_matrix = last_rotation_matrix @ delta_rotation_matrix
        self.rotation_matrix_ = curr_rotation_matrix
        self.quat_ = rot2quaternion(curr_rotation_matrix)
    
        # 基于四元数的更新算法
        # delta_quat = rv2quaternion(delta_theta)
        # last_quat = quat_
        # curr_quat = last_quat

        # 速度更新
        unbias_accel_0 = last_rotation_matrix @ (last_imu_data.imu_linear_acceleration - self.accel_bias_)-self.g_
        unbias_accel_1 = curr_rotation_matrix @ (curr_imu_data.imu_linear_acceleration - self.accel_bias_)-self.g_
        last_vel = self.velocity_
        curr_vel = last_vel + delta_t * 0.5 * (unbias_accel_0 + unbias_accel_1)
        self.velocity_ = curr_vel

        # 位置更新
        # self.pos_ = self.pos_ + 0.5 * delta_t * (curr_vel + last_vel) + \
        #     0.25 * (curr_imu_data.imu_linear_acceleration+last_imu_data.imu_linear_acceleration) * delta_t * delta_t
        self.pos_ = self.pos_ + 0.5 * delta_t * (curr_vel + last_vel) + 0.25 * (unbias_accel_0 + unbias_accel_1) * delta_t * delta_t
        self.last_imu_data = curr_imu_data

        self.UpdateErrorState(delta_t, curr_accel)

    
    def UpdateErrorState(self, delta_t, curr_accel_n):
        F_23 = BuildSkewSymmetricMatrix(curr_accel_n)
        F_33 = -BuildSkewSymmetricMatrix(np.array([0,0,0]).reshape(3,1)) # w_in_n

        self.F[INDEX_STATE_POSI:INDEX_STATE_VEL, INDEX_STATE_VEL:INDEX_STATE_ORI] = np.eye(3)
        self.F[INDEX_STATE_VEL:INDEX_STATE_ORI, INDEX_STATE_ORI:INDEX_STATE_GYRO_BIAS] = F_23
        self.F[INDEX_STATE_ORI:INDEX_STATE_GYRO_BIAS, INDEX_STATE_ORI:INDEX_STATE_GYRO_BIAS] = F_33
        self.F[INDEX_STATE_VEL:INDEX_STATE_ORI, INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3] = self.rotation_matrix_
        self.F[INDEX_STATE_ORI:INDEX_STATE_GYRO_BIAS, INDEX_STATE_GYRO_BIAS:INDEX_STATE_ACC_BIAS] = -self.rotation_matrix_

        self.B[INDEX_STATE_VEL:INDEX_STATE_ORI, 3:6] = self.rotation_matrix_
        self.B[INDEX_STATE_ORI:INDEX_STATE_GYRO_BIAS, 0:3] = -self.rotation_matrix_

        Fk = np.eye(DIM_STATE) + self.F * delta_t
        Bk = self.B * delta_t

        self.X = Fk @ self.X
        self.P = Fk @ self.P @ Fk.T + Bk @ self.Q @ Bk.T

    def Correct(self):
        if self.curr_gnss_data.gnss_time > self.last_imu_data.imu_time:
            return
        else:
            self.Y = lla2ned(self.curr_gnss_data.position_lla,self.ref_pos_lla) - self.pos_
            self.K = self.P @ self.G.T @ np.linalg.inv(self.G @ self.P @ self.G.T + self.C @ self.R @ self.C.T)
            self.P = (np.eye(DIM_STATE) - self.K @ self.G) @ self.P
            self.X = self.X + self.K @ (self.Y - self.G @ self.X)
            self.EliminateError()
            self.ResetState()
            if not self.gnss_data_queue.empty():
                self.curr_gnss_data = self.gnss_data_queue.get()
            else:
                print('GNSS data is empty')

    def EliminateError(self):
        self.pos_ = self.pos_ + self.X[INDEX_STATE_POSI:INDEX_STATE_VEL, :]
        self.velocity_ = self.velocity_ + self.X[INDEX_STATE_VEL:INDEX_STATE_ORI, :]
        self.rotation_matrix_ = rv2rot(-self.X[INDEX_STATE_ORI:INDEX_STATE_GYRO_BIAS, :]) @ self.rotation_matrix_
        self.quat_ = rot2quaternion(self.rotation_matrix_)
        self.gyro_bias_ = self.gyro_bias_ + self.X[INDEX_STATE_GYRO_BIAS:INDEX_STATE_ACC_BIAS, :]
        self.accel_bias_ = self.accel_bias_ + self.X[INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3, :]

    def ResetState(self):
        self.X = np.zeros((DIM_STATE, 1))

    def SaveData(self, data_path:str):
        with open(data_path, 'a') as f:
            f.write(str(eskf.last_imu_data.imu_time)+' ')
            for i in range(3):
                f.write(str(eskf.pos_[i][0])+' ')
            for i in range(4):
                if i < 3:
                    f.write(str(eskf.quat_[i])+' ')
                else:
                    f.write(str(eskf.quat_[i])+'\n')

    def SaveGnssData(self, data_path:str):
        curr_position_ned = lla2ned(self.curr_gnss_data.position_lla,self.ref_pos_lla)
        with open(data_path, 'a') as f:
            f.write(str(eskf.curr_gnss_data.gnss_time)+' ')
            for i in range(3):
                f.write(str(curr_position_ned[i][0])+' ')
            f.write('0 0 0 1\n')

    def SaveGTData(self, data_path:str):
        gt_pos_ned = lla2ned(self.last_imu_data.gt_pos_lla, self.ref_pos_lla)
        with open(data_path, 'a') as f:
            f.write(str(eskf.last_imu_data.imu_time)+' ')
            for i in range(3):
                f.write(str(gt_pos_ned[i][0])+' ')
            for i in range(4):
                if i < 3:
                    f.write(str(eskf.quat_[i])+' ')
                else:
                    f.write(str(eskf.quat_[i])+'\n')
        
if __name__ == "__main__":
    data_path = './data/raw_data'

    # load configuration
    config_path = os.path.join(data_path,'config.yaml')
    with open(config_path,encoding='utf-8') as config_file:
        config = yaml.load(config_file,Loader=yaml.FullLoader)
        print(config)

    fuse_file = os.path.join(data_path,'fused.txt')
    gps_file = os.path.join(data_path,'gps_measurement.txt')
    gt_file = os.path.join(data_path,'gt.txt')
    ins_file = os.path.join(data_path,'ins.txt')
    if os.path.exists(fuse_file):
        os.remove(fuse_file)
    if os.path.exists(gps_file):
        os.remove(gps_file)
    if os.path.exists(gt_file):
        os.remove(gt_file)
    if os.path.exists(ins_file):
        os.remove(ins_file)
    
    # imu only
    """
    eskf = ESKF(config)
    eskf.Init(data_path)
    while((not eskf.imu_data_queue.empty()) and (not eskf.gnss_data_queue.empty())):
        eskf.Predict()
        eskf.SaveData(ins_file)  
    """
    # imu gps fuse
    eskf = ESKF(config)
    eskf.Init(data_path)
    while((not eskf.imu_data_queue.empty()) and (not eskf.gnss_data_queue.empty())):
        eskf.Predict()
        if eskf.curr_gnss_data.gnss_time <= eskf.last_imu_data.imu_time:
            eskf.Correct()
            eskf.SaveGnssData(gps_file)
            eskf.SaveData(fuse_file)
            eskf.SaveGTData(gt_file)   

    # display
    display = True
    if display:

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('compare path')
        if os.path.exists(fuse_file):
            fuse_data = np.loadtxt(fuse_file)
            ax.plot3D(fuse_data[:, 1], fuse_data[:, 2], fuse_data[:, 3], color='r', label='fuse_gps_imu')
        if os.path.exists(gps_file):
            gps_data = np.loadtxt(gps_file)
            ax.plot3D(gps_data[:, 1], gps_data[:, 2], gps_data[:, 3], color='g', alpha=0.5, label='gps')
        if os.path.exists(gt_file):
            gt_data = np.loadtxt(gt_file)
            ax.plot3D(gt_data[:, 1], gt_data[:, 2], gt_data[:, 3], color='b', label='ground_truth')
        if os.path.exists(ins_file):
            ins_data = np.loadtxt(ins_file)
            # ax.plot3D(ins_data[:, 1], ins_data[:, 2], ins_data[:, 3], color='y', label='ins')
        plt.legend(loc='best')
        plt.show()