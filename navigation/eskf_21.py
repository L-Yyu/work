import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import queue
import os
import yaml
from data import IMUData, GNSSData
from tools import *

# dim_state = 21
# dim_state_noise = 18
# DIM_MEASUREMENT = 3
# DIM_MEASUREMENT_NOISE = 3

INDEX_STATE_POSI = 0
INDEX_STATE_VEL = 3
INDEX_STATE_ORI = 6
INDEX_STATE_GYRO_BIAS = 9
INDEX_STATE_ACC_BIAS = 12
INDEX_STATE_GYRO_SCALE = 15
INDEX_STATE_ACC_SCALE = 18

INDEX_MEASUREMENT_POSI = 0

D2R = np.pi/180

class ESKF(object):

    def __init__(self, config, states_rank, noise_rank) -> None:
        self.dim_state = 15         #   15:p v phi bg ba    21:p v phi bg ba sg sa
        self.dim_state_noise = 6   #   6:w_phi w_v     12: w_phi w_v bg ba     18: w_phi w_v bg ba sg sa
        self.dim_measurement = 3
        self.dim_measurement_noise = 3
        # dX = FX + BW(Q)  Y = GX + CN(R)   Cov(X) = P
        self.X = np.zeros((self.dim_state, 1))
        self.Y = np.zeros((self.dim_measurement, 1))
        self.F = np.zeros((self.dim_state, self.dim_state))
        self.B = np.zeros((self.dim_state, self.dim_state_noise))
        self.Q = np.zeros((self.dim_state_noise, self.dim_state_noise))
        self.P = np.zeros((self.dim_state, self.dim_state))
        self.K = np.zeros((self.dim_state, self.dim_measurement))
        self.C = np.identity(self.dim_measurement_noise)
        self.G = np.zeros((self.dim_measurement, self.dim_state))
        self.R = np.zeros((self.dim_measurement, self.dim_measurement))
        self.Ft = np.zeros((self.dim_state, self.dim_state))

        self.velocity_ = np.zeros((3, 1))
        self.quat_ = np.zeros((4, 1))
        self.rotation_matrix_ = np.zeros((3, 3))
        self.pos_ = np.zeros((3, 1))
        self.ref_pos_lla = np.array(config['init_position_lla']).reshape(3,1)

        self.gyro_bias_ = np.zeros((3, 1))
        self.accel_bias_ = np.zeros((3, 1))
        self.g_ = np.array([0, 0, -GetGravity(self.ref_pos_lla)]).reshape((3,1))

        self.curr_gnss_data:GNSSData
        self.last_imu_data:IMUData
        self.curr_imu_data:IMUData
        self.imu_data_queue = queue.Queue()
        self.gnss_data_queue = queue.Queue()
        # init P
        self.SetP(config['init_position_std'], config['init_velocity_std'], config['init_rotation_std'],
                  config['init_bg_std'], config['init_ba_std'], config['init_sg_std'], config['init_sa_std'])
        # 输入(imu bias)的方差
        self.SetQ(config['arw'], config['vrw'], config['bg_std'], config['ba_std'], config['sg_std'], config['sa_std'], config['corr_time'])
        # 观测(gnss)的方差
        self.SetR(config['gps_position_x_std'], config['gps_position_y_std'], config['gps_position_z_std'])
        self.SetG()

        self.corr_time_ = config['corr_time']

    def SetQ(self, arw, vrw, bg_std, ba_std, sg_std, sa_std, corr_time):
        # 设置imu误差
        # 转换为标准单位
        arw = np.array(arw) * D2R/60.0   # deg/sqrt(h) -> rad/sqrt(s)
        vrw = np.array(vrw) / 60.0  # m/s/sqrt(h) -> m/s/sqrt(s)
        bg_std = np.array(bg_std) * D2R/ 3600.0  # deg/h -> rad/s
        ba_std = np.array(ba_std) * 1e-5    # mGal -> m/s^2
        sg_std = np.array(sg_std) * 1e-6    # ppm -> 1
        sa_std = np.array(sa_std) * 1e-6    # ppm -> 1
        corr_time = np.array(corr_time) * 3600.0  # h -> s
        # self.Q = np.zeros((self.dim_state_noise, self.dim_state_noise))
        if self.dim_state == 15 and self.dim_state_noise == 6:
            self.Q[0:3, 0:3] = np.eye(3) * arw * arw
            self.Q[3:6, 3:6] = np.eye(3) * vrw * vrw
        elif self.dim_state==15 and self.dim_state_noise == 12:
            self.Q[0:3, 0:3] = np.eye(3) * arw * arw
            self.Q[3:6, 3:6] = np.eye(3) * vrw * vrw
            self.Q[6:9, 6:9] = np.eye(3) * bg_std * bg_std* 2 / corr_time
            self.Q[9:12, 9:12] = np.eye(3) * ba_std * ba_std* 2 / corr_time
        elif self.dim_state==21 and self.dim_state_noise == 18:
            self.Q[0:3, 0:3] = np.eye(3) * arw * arw
            self.Q[3:6, 3:6] = np.eye(3) * vrw * vrw
            self.Q[6:9, 6:9] = np.eye(3) * bg_std * bg_std* 2 / corr_time
            self.Q[9:12, 9:12] = np.eye(3) * ba_std * ba_std* 2 / corr_time
            self.Q[12:15, 12:15] = np.eye(3) * sg_std * sg_std* 2 / corr_time
            self.Q[15:18, 15:18] = np.eye(3) * sa_std * sa_std* 2 / corr_time
        else:
            print('SetQ failed')
            return

    def SetR(self, position_x_std, position_y_std, position_z_std):
        # self.R = np.zeros((self.dim_measurement, self.dim_measurement))
        self.R[0, 0] =  position_x_std * position_x_std
        self.R[1, 1] =  position_y_std * position_y_std
        self.R[2, 2] =  position_z_std * position_z_std

    def SetP(self, init_posi_std, init_vel_std, init_ori_std, init_bg_std, init_ba_std, init_sg_std, init_sa_std):
        # 设置初始状态协方差矩阵
        init_posi_std = np.array(init_posi_std)  # m
        init_vel_std = np.array(init_vel_std)   # m/s
        init_ori_std = np.array(init_ori_std) * D2R   # deg -> rad
        init_bg_std = np.array(init_bg_std) * D2R / 3600.0    # deg/h -> rad/s
        init_ba_std = np.array(init_ba_std) * 1e-5  # mGal -> m/s^2
        init_sg_std = np.array(init_sg_std) * 1e-6  # ppm -> 1
        init_sa_std = np.array(init_sa_std) * 1e-6  # ppm -> 1
        # self.P = np.zeros((self.dim_state, self.dim_state))
        if self.dim_state == 15:
            self.P[INDEX_STATE_POSI:INDEX_STATE_POSI+3, INDEX_STATE_POSI:INDEX_STATE_POSI+3] = np.eye(3) * init_posi_std * init_posi_std
            self.P[INDEX_STATE_VEL:INDEX_STATE_VEL+3, INDEX_STATE_VEL:INDEX_STATE_VEL+3] = np.eye(3) * init_vel_std * init_vel_std
            self.P[INDEX_STATE_ORI:INDEX_STATE_ORI+3, INDEX_STATE_ORI:INDEX_STATE_ORI+3] = np.eye(3) * init_ori_std * init_ori_std
            self.P[INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3, INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3] = np.eye(3) * init_bg_std * init_bg_std
            self.P[INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3, INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3] = np.eye(3) * init_ba_std * init_ba_std    
        elif self.dim_state == 21:
            self.P[INDEX_STATE_POSI:INDEX_STATE_POSI+3, INDEX_STATE_POSI:INDEX_STATE_POSI+3] = np.eye(3) * init_posi_std * init_posi_std
            self.P[INDEX_STATE_VEL:INDEX_STATE_VEL+3, INDEX_STATE_VEL:INDEX_STATE_VEL+3] = np.eye(3) * init_vel_std * init_vel_std
            self.P[INDEX_STATE_ORI:INDEX_STATE_ORI+3, INDEX_STATE_ORI:INDEX_STATE_ORI+3] = np.eye(3) * init_ori_std * init_ori_std
            self.P[INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3, INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3] = np.eye(3) * init_bg_std * init_bg_std
            self.P[INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3, INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3] = np.eye(3) * init_ba_std * init_ba_std  
            self.P[INDEX_STATE_GYRO_SCALE:INDEX_STATE_GYRO_SCALE+3, INDEX_STATE_GYRO_SCALE:INDEX_STATE_GYRO_SCALE+3] = np.eye(3) * init_sg_std * init_sg_std
            self.P[INDEX_STATE_ACC_SCALE:INDEX_STATE_ACC_SCALE+3, INDEX_STATE_ACC_SCALE:INDEX_STATE_ACC_SCALE+3] = np.eye(3) * init_sa_std * init_sa_std
        else:
            print('SetP failed')
            return
    def SetG(self):
        # 设置观测矩阵
        # self.G = np.zeros((self.dim_measurement, self.dim_state))
        self.G[INDEX_MEASUREMENT_POSI:INDEX_MEASUREMENT_POSI+3, INDEX_MEASUREMENT_POSI:INDEX_MEASUREMENT_POSI+3] = np.eye(3)
    def InitState(self, data_path:str):
        self.imu_data_queue = IMUData.read_imu_data(data_path)
        self.gnss_data_queue = GNSSData.read_gnss_data(data_path)

        self.last_imu_data = self.imu_data_queue.get()
        self.curr_gnss_data = self.gnss_data_queue.get()

        self.pos_ = np.array([0, 0, 0]).reshape(3,1)
        self.pos_lla_ = ned2lla(self.pos_, self.ref_pos_lla)
        self.velocity_= np.array([[0], [0], [0]])
        self.velocity_ = self.curr_gnss_data.true_velocity

        self.quat_ = euler2quaternion(np.array([0, 0, 0])).reshape(4, 1)
        self.rotation_matrix_ = euler2rot(np.array([0, 0, 0]))

        self.gyro_bias_ = np.zeros((3, 1))
        self.accel_bias_ = np.zeros((3, 1))
        
        if self.dim_state == 21:
            self.gyro_scale_ = np.ones((3, 1))
            self.accel_scale_ = np.ones((3, 1))

    def Predict(self):
        self.curr_imu_data = self.imu_data_queue.get()
        curr_imu_data = self.curr_imu_data
        last_imu_data = self.last_imu_data
        delta_t = curr_imu_data.imu_time - last_imu_data.imu_time
        # 根据上一时刻状态，计算状态转移矩阵F_k-1, B_k-1
        self.UpdateErrorState(delta_t, last_imu_data)
        # 姿态更新
        unbias_gyro_0 = last_imu_data.imu_angle_vel - self.gyro_bias_   #[deg/s]
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
        unbias_accel_n_0 = last_rotation_matrix @ (last_imu_data.imu_linear_acceleration - self.accel_bias_)-self.g_
        unbias_accel_n_1 = curr_rotation_matrix @ (curr_imu_data.imu_linear_acceleration - self.accel_bias_)-self.g_
        last_vel_n = self.velocity_
        curr_vel_n = last_vel_n + delta_t * 0.5 * (unbias_accel_n_0 + unbias_accel_n_1)
        self.velocity_ = curr_vel_n

        # 位置更新
        self.pos_ = self.pos_ + 0.5 * delta_t * (curr_vel_n + last_vel_n) + 0.25 * (unbias_accel_n_0 + unbias_accel_n_1) * delta_t * delta_t
        # self.pos_lla_ = NED2lla(self.pos_, self.ref_pos_lla)
        # self.g_ = np.array([0, 0, -GetGravity(self.pos_lla_)]).reshape((3,1))
        
        self.last_imu_data = curr_imu_data

    
    def UpdateErrorState(self, delta_t, last_imu_data):
        accel_b = last_imu_data.imu_linear_acceleration 
        accel_n = self.rotation_matrix_ @ accel_b
        omega_ib_b = last_imu_data.imu_angle_vel
        omega_ib_n = self.rotation_matrix_ @ omega_ib_b
        
        # 状态转移矩阵
        F_23 = BuildSkewSymmetricMatrix(accel_n)
        F_33 = -BuildSkewSymmetricMatrix(np.array([0,0,0]).reshape(3,1)) # w_in_n or w_ie_n

        if self.dim_state == 15 and self.dim_state_noise == 6:
            self.F[INDEX_STATE_POSI:INDEX_STATE_POSI+3, INDEX_STATE_VEL:INDEX_STATE_VEL+3] = np.eye(3)
            self.F[INDEX_STATE_VEL:INDEX_STATE_VEL+3, INDEX_STATE_ORI:INDEX_STATE_ORI+3] = F_23
            self.F[INDEX_STATE_ORI:INDEX_STATE_ORI+3, INDEX_STATE_ORI:INDEX_STATE_ORI+3] = F_33
            self.F[INDEX_STATE_VEL:INDEX_STATE_VEL+3, INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3] = self.rotation_matrix_
            self.F[INDEX_STATE_ORI:INDEX_STATE_ORI+3, INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3] = -self.rotation_matrix_
        elif self.dim_state == 21 and self.dim_state_noise == 6:
            self.F[INDEX_STATE_POSI:INDEX_STATE_POSI+3, INDEX_STATE_VEL:INDEX_STATE_VEL+3] = np.eye(3)
            self.F[INDEX_STATE_VEL:INDEX_STATE_VEL+3, INDEX_STATE_ORI:INDEX_STATE_ORI+3] = F_23
            self.F[INDEX_STATE_ORI:INDEX_STATE_ORI+3, INDEX_STATE_ORI:INDEX_STATE_ORI+3] = F_33
            self.F[INDEX_STATE_VEL:INDEX_STATE_VEL+3, INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3] = self.rotation_matrix_
            self.F[INDEX_STATE_ORI:INDEX_STATE_ORI+3, INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3] = -self.rotation_matrix_
            self.F[INDEX_STATE_VEL:INDEX_STATE_VEL+3, INDEX_STATE_ACC_SCALE:INDEX_STATE_ACC_SCALE+3] = self.rotation_matrix_ @ np.diag(accel_b.reshape(3))
            self.F[INDEX_STATE_ORI:INDEX_STATE_ORI+3, INDEX_STATE_GYRO_SCALE:INDEX_STATE_GYRO_SCALE+3] = -self.rotation_matrix_ @ np.diag(omega_ib_b.reshape(3))
        elif self.dim_state == 15 and self.dim_state_noise == 12:
            self.F[INDEX_STATE_POSI:INDEX_STATE_POSI+3, INDEX_STATE_VEL:INDEX_STATE_VEL+3] = np.eye(3)
            self.F[INDEX_STATE_VEL:INDEX_STATE_VEL+3, INDEX_STATE_ORI:INDEX_STATE_ORI+3] = F_23
            self.F[INDEX_STATE_ORI:INDEX_STATE_ORI+3, INDEX_STATE_ORI:INDEX_STATE_ORI+3] = F_33
            self.F[INDEX_STATE_VEL:INDEX_STATE_VEL+3, INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3] = self.rotation_matrix_
            self.F[INDEX_STATE_ORI:INDEX_STATE_ORI+3, INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3] = -self.rotation_matrix_
            self.F[INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3, INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3] = (-1/self.corr_time_)*np.eye(3)
            self.F[INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3, INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3] = (-1/self.corr_time_)*np.eye(3)
        elif self.dim_state == 21 and self.dim_state_noise == 18:
            self.F[INDEX_STATE_POSI:INDEX_STATE_POSI+3, INDEX_STATE_VEL:INDEX_STATE_VEL+3] = np.eye(3)
            self.F[INDEX_STATE_VEL:INDEX_STATE_VEL+3, INDEX_STATE_ORI:INDEX_STATE_ORI+3] = F_23
            self.F[INDEX_STATE_ORI:INDEX_STATE_ORI+3, INDEX_STATE_ORI:INDEX_STATE_ORI+3] = F_33
            self.F[INDEX_STATE_VEL:INDEX_STATE_VEL+3, INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3] = self.rotation_matrix_
            self.F[INDEX_STATE_ORI:INDEX_STATE_ORI+3, INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3] = -self.rotation_matrix_
            self.F[INDEX_STATE_VEL:INDEX_STATE_VEL+3, INDEX_STATE_ACC_SCALE:INDEX_STATE_ACC_SCALE+3] = self.rotation_matrix_ @ np.diag(accel_b.reshape(3))
            self.F[INDEX_STATE_ORI:INDEX_STATE_ORI+3, INDEX_STATE_GYRO_SCALE:INDEX_STATE_GYRO_SCALE+3] = -self.rotation_matrix_ @ np.diag(omega_ib_b.reshape(3))
            self.F[INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3, INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3] = (-1/self.corr_time_)*np.eye(3)
            self.F[INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3, INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3] = (-1/self.corr_time_)*np.eye(3)
            self.F[INDEX_STATE_GYRO_SCALE:INDEX_STATE_GYRO_SCALE+3, INDEX_STATE_GYRO_SCALE:INDEX_STATE_GYRO_SCALE+3] = (-1/self.corr_time_)*np.eye(3)
            self.F[INDEX_STATE_ACC_SCALE:INDEX_STATE_ACC_SCALE+3, INDEX_STATE_ACC_SCALE:INDEX_STATE_ACC_SCALE+3] = (-1/self.corr_time_)*np.eye(3)
        else:
            print('wrong dim_state or dim_state_noise')

        # 噪声驱动矩阵
        if self.dim_state_noise == 6:
            self.B[INDEX_STATE_VEL:INDEX_STATE_VEL+3, 3:6] = self.rotation_matrix_
            self.B[INDEX_STATE_ORI:INDEX_STATE_ORI+3, 0:3] = -self.rotation_matrix_
        elif self.dim_state_noise == 12:
            self.B[INDEX_STATE_VEL:INDEX_STATE_VEL+3, 3:6] = self.rotation_matrix_
            self.B[INDEX_STATE_ORI:INDEX_STATE_ORI+3, 0:3] = -self.rotation_matrix_
            self.B[INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3, 6:9] = np.eye(3)
            self.B[INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3, 9:12] = np.eye(3)
        elif self.dim_state_noise == 18:
            self.B[INDEX_STATE_VEL:INDEX_STATE_VEL+3, 3:6] = self.rotation_matrix_
            self.B[INDEX_STATE_ORI:INDEX_STATE_ORI+3, 0:3] = -self.rotation_matrix_
            self.B[INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3, 6:9] = np.eye(3)
            self.B[INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3, 9:12] = np.eye(3)
            self.B[INDEX_STATE_GYRO_SCALE:INDEX_STATE_GYRO_SCALE+3, 12:15] = np.eye(3)
            self.B[INDEX_STATE_ACC_SCALE:INDEX_STATE_ACC_SCALE+3, 15:18] = np.eye(3)

        # 离散化1
        Fk = np.eye(self.dim_state) + self.F * delta_t
        Bk = self.B * delta_t
        Qd = Bk @ self.Q @ Bk.T
        
        # 离散化2
        # Fk = np.eye(self.dim_state) + self.F * delta_t
        # Qd = self.B @ self.Q @ self.B.T
        # Qd = (Fk @ Qd @ Fk.T + Qd) * delta_t / 2

        # KF prediction
        self.X = Fk @ self.X
        self.P = Fk @ self.P @ Fk.T + Qd


    def Correct(self):
        if self.curr_gnss_data.gnss_time > self.last_imu_data.imu_time:
            return
        else:
            self.Y = lla2ned(self.curr_gnss_data.position_lla,self.ref_pos_lla) - self.pos_
            self.K = self.P @ self.G.T @ np.linalg.inv(self.G @ self.P @ self.G.T + self.C @ self.R @ self.C.T)
            self.P = (np.eye(self.dim_state) - self.K @ self.G) @ self.P
            self.X = self.X + self.K @ (self.Y - self.G @ self.X)
            self.EliminateError()
            self.ResetState()
            if not self.gnss_data_queue.empty():
                self.curr_gnss_data = self.gnss_data_queue.get()
            else:
                print('GNSS data is empty')

    def EliminateError(self):
        self.pos_ = self.pos_ + self.X[INDEX_STATE_POSI:INDEX_STATE_POSI+3, :]
        self.velocity_ = self.velocity_ + self.X[INDEX_STATE_VEL:INDEX_STATE_VEL+3, :]
        self.rotation_matrix_ = rv2rot(-self.X[INDEX_STATE_ORI:INDEX_STATE_ORI+3, :]) @ self.rotation_matrix_
        self.quat_ = rot2quaternion(self.rotation_matrix_)
        self.gyro_bias_ = self.gyro_bias_ + self.X[INDEX_STATE_GYRO_BIAS:INDEX_STATE_GYRO_BIAS+3, :]
        self.accel_bias_ = self.accel_bias_ + self.X[INDEX_STATE_ACC_BIAS:INDEX_STATE_ACC_BIAS+3, :]
        if self.dim_state == 21:
            self.gyro_scale_ = self.gyro_scale_ + self.X[INDEX_STATE_GYRO_SCALE:INDEX_STATE_GYRO_SCALE+3, :]
            self.accel_scale_ = self.accel_scale_ + self.X[INDEX_STATE_ACC_SCALE:INDEX_STATE_ACC_SCALE+3, :]

        # self.pos_lla_ = NED2lla(self.pos_, self.ref_pos_lla)
        # self.g_ = np.array([0, 0, -GetGravity(self.pos_lla_)]).reshape((3,1))

    def ResetState(self):
        self.X = np.zeros((self.dim_state, 1))

    def SaveData(self, file):
        file.write(str(eskf.last_imu_data.imu_time)+' ')
        for i in range(3):
            file.write(str(eskf.pos_[i][0])+' ')
        for i in range(4):
            if i < 3:
                file.write(str(eskf.quat_[i])+' ')
            else:
                file.write(str(eskf.quat_[i])+'\n')

    def SaveGnssData(self, file):
        curr_position_ned = lla2ned(self.curr_gnss_data.position_lla,self.ref_pos_lla)
        file.write(str(eskf.curr_gnss_data.gnss_time)+' ')
        for i in range(3):
            file.write(str(curr_position_ned[i][0])+' ')
        file.write('0 0 0 1\n')

    def SaveGTData(self, file):
        gt_pos_ned = lla2ned(self.last_imu_data.gt_pos_lla, self.ref_pos_lla)
        file.write(str(eskf.last_imu_data.imu_time)+' ')
        for i in range(3):
            file.write(str(gt_pos_ned[i][0])+' ')
        for i in range(4):
            if i < 3:
                file.write(str(eskf.quat_[i])+' ')
            else:
                file.write(str(eskf.quat_[i])+'\n')
        
if __name__ == "__main__":
    data_path = './data/sim_2'
    states_rank = 15
    noise_rank = 6

    # load configuration
    config_path = os.path.join(data_path,'config.yaml')
    with open(config_path,encoding='utf-8') as config_file:
        config = yaml.load(config_file,Loader=yaml.FullLoader)
        print(config)

    fuse_file_name = os.path.join(data_path,'fused.txt')
    gps_file_name = os.path.join(data_path,'gps_measurement.txt')
    gt_file_name = os.path.join(data_path,'gt.txt')
    ins_file_name = os.path.join(data_path,'ins.txt')
    
    fuse_file = open(fuse_file_name,'w')
    gps_file = open(gps_file_name,'w')
    gt_file = open(gt_file_name,'w')
    ins_file = open(ins_file_name,'w')

    # imu only
    '''
    eskf = ESKF(config,states_rank,noise_rank)
    eskf.Init(data_path)
    while((not eskf.imu_data_queue.empty()) and (not eskf.gnss_data_queue.empty())):
        eskf.Predict()
        eskf.SaveData(ins_file)  
    '''
    
    # imu gps fuse
    eskf = ESKF(config, states_rank, noise_rank)
    eskf.InitState(data_path)

    while((not eskf.imu_data_queue.empty()) and (not eskf.gnss_data_queue.empty())):
        eskf.Predict()
        if eskf.curr_gnss_data.gnss_time <= eskf.last_imu_data.imu_time:
            eskf.Correct()
            eskf.SaveGnssData(gps_file)
            eskf.SaveData(fuse_file)
            eskf.SaveGTData(gt_file)   
    fuse_file.close()
    gps_file.close()
    gt_file.close()
    ins_file.close()
    
    # display
    display = True
    if display:

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('compare path')
        if os.path.exists(fuse_file_name):
            fuse_data = np.loadtxt(fuse_file_name)
            ax.plot3D(fuse_data[:, 1], fuse_data[:, 2], fuse_data[:, 3], color='r', label='fuse_gps_imu')
        if os.path.exists(gps_file_name):
            gps_data = np.loadtxt(gps_file_name)
            ax.plot3D(gps_data[:, 1], gps_data[:, 2], gps_data[:, 3], color='g', alpha=0.5, label='gps')
        if os.path.exists(gt_file_name):
            gt_data = np.loadtxt(gt_file_name)
            ax.plot3D(gt_data[:, 1], gt_data[:, 2], gt_data[:, 3], color='b', label='ground_truth')
        if os.path.exists(ins_file_name):
            ins_data = np.loadtxt(ins_file_name)
            # ax.plot3D(ins_data[:, 1], ins_data[:, 2], ins_data[:, 3], color='y', label='ins')
        plt.legend(loc='best')
        plt.show()