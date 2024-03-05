# -*- coding: utf-8 -*-
# Filename: demo_no_algo.py

"""
The simplest demo of Sim.
Only generate reference trajectory (pos, vel, sensor output). No algorithm.
Created on 2018-01-23
@author: dongxiaoguang
"""

import numpy as np
import os
import math
from gnss_ins_sim.sim import imu_model
from gnss_ins_sim.sim import ins_sim

# globals
D2R = math.pi/180

fs = 100.0          # IMU sample frequency
fs_gps = 10.0       # GPS sample frequency
fs_mag = fs         # magnetometer sample frequency, not used for now

def test_path_gen():
    '''
    test only path generation in Sim.
    '''
    # IMU model, typical for IMU381 
    """
    imu_err = {
            # gyro bias, deg/hr
            'gyro_b': np.array([0.0, 0.0, 0.0]),
            # gyro angle random walk, deg/rt-hr
            'gyro_arw': np.array([0.2, 0.2, 0.2]),
            # gyro bias instability, deg/hr
            'gyro_b_stability': np.array([200, 200, 200]),
            # gyro bias instability correlation, sec.
            # set this to 'inf' to use a random walk model
            # set this to a positive real number to use a first-order Gauss-Markkov model
            'gyro_b_corr': np.array([100.0, 100.0, 100.0]),
            # accelerometer bias, m/s^2
            'accel_b': np.array([0.0, 0.0, 0.0]),
            # accelerometer velocity random walk, m/s/rt-hr
            'accel_vrw': np.array([0.2, 0.2, 0.2]),
            # accelerometer bias instability, m/s^2
            'accel_b_stability': np.array([1.0e3, 1.0e3, 1.0e3]),
            # accelerometer bias instability correlation, sec. Similar to gyro_b_corr
            'accel_b_corr': np.array([3.6e3, 3.6e3, 3.6e3]),
            # magnetometer noise std, uT
            'mag_std': np.array([0.2, 0.2, 0.2])
          }
    """
    data_name = 'sim_2'
    
    imu_err = {
            # gyro bias, deg/hr
            'gyro_b': np.array([0.0, 0.0, 0.0]),
            # gyro angle random walk, deg/rt-hr
            'gyro_arw': np.array([0.2, 0.2, 0.2]),
            # gyro bias instability, deg/hr
            'gyro_b_stability': np.array([15, 15, 15]),
            # gyro bias instability correlation, sec.
            # set this to 'inf' to use a random walk model
            # set this to a positive real number to use a first-order Gauss-Markkov model
            'gyro_b_corr': np.array([3.6e3, 3.6e3, 3.6e3]),
            # accelerometer bias, m/s^2
            'accel_b': np.array([0.0, 0.0, 0.0]),
            # accelerometer velocity random walk, m/s/rt-hr
            'accel_vrw': np.array([0.2, 0.2, 0.2]),
            # accelerometer bias instability, m/s^2
            'accel_b_stability': np.array([150, 150, 150])*1e-5,
            # accelerometer bias instability correlation, sec. Similar to gyro_b_corr
            'accel_b_corr': np.array([3.6e3, 3.6e3, 3.6e3]),
            # magnetometer noise std, uT
            'mag_std': np.array([0.2, 0.2, 0.2])
          }
    imu_err = 'mid-accuracy'
    
    # generate GPS and magnetometer data
    imu = imu_model.IMU(accuracy=imu_err, axis=6, gps=True, odo=True)

    #### start simulation
    sim = ins_sim.Sim([fs, fs_gps, fs_mag],
                      './motion_def_files'+'/'+data_name+'.csv',
                      ref_frame=0, # 0: NED
                      imu=imu,
                      mode=None,
                      env=None,
                      algorithm=None)
    sim.run(1)
    # save simulation data to files
    sim.results('./data'+'/'+data_name)
    # plot data, 3d plot of reference positoin, 2d plots of gyro and accel
    sim.plot(['ref_pos','accel', 'gyro', 'gps_visibility'], opt={'ref_pos': '3d'})

if __name__ == '__main__':
    test_path_gen()
