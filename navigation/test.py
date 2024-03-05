# txt to csv
import pandas as pd
from tqdm import tqdm
import csv
import os

from tools import *
data_path = 'F:/work/KF-GINS/dataset'
# 读取 txt 文件到 DataFrame

# 如果文件中没有标题行，可以设置 header=None，并通过 names 参数指定列名
# df = pd.read_csv(data_path, delimiter='\t', header=None, names=['Column1', 'Column2', ...])
#  names=['week', 'seconds', 'lat', 'lon', 'h', 'vn', 've', 'vd', 'roll','pitch','yaw']
# 查看 DataFrame 的第一个元素
#print(df.loc[0]['Column1'])


# process truth.nav

ref_pos_lla = np.array([[0.0], [0.0], [0.0]])

with open(os.path.join(data_path,'truth.nav'), mode='r', encoding='utf-8') as f:
    lines = f.readlines()

with open(os.path.join(data_path, 'gt.txt'), 'a') as save_file:
    for iter, line in enumerate(tqdm(lines, desc="Processing lines")):
        numbers = [float(num) for num in line.split()]
        time = numbers[1]
        pos_lla = np.array(numbers[2:5]).reshape(3, 1)
        if iter == 0:
            ref_pos_lla = pos_lla
        pos_ned = lla2ned(pos_lla, ref_pos_lla)
        euler = np.array(numbers[8:11])
        quat = euler2quaternion(euler)
        
        # 写入结果到文件
        save_file.write(str(time) + ' ')
        for i in range(3):
            save_file.write(str(pos_ned[i][0]) + ' ')
        for i in range(4):
            if i < 3:
                save_file.write(str(quat[i]) + ' ')
            else:
                save_file.write(str(quat[i]) + '\n')

# process GNSS
with open(os.path.join(data_path,'GNSS-RTK.txt'), mode='r', encoding='utf-8') as f:
    lines = f.readlines()

with open(os.path.join(data_path, 'gps-0.txt'), 'a') as save_file:
    for iter, line in enumerate(tqdm(lines, desc="Processing lines")):
        numbers = [float(num) for num in line.split()]
        time = numbers[1]
        pos_lla = np.array(numbers[2:5]).reshape(3, 1)
        if iter == 0:
            ref_pos_lla = pos_lla
        pos_ned = lla2ned(pos_lla, ref_pos_lla)
        euler = np.array(numbers[8:11])
        quat = euler2quaternion(euler)
        
        # 写入结果到文件
        save_file.write(str(time) + ' ')
        for i in range(3):
            save_file.write(str(pos_ned[i][0]) + ' ')
        for i in range(4):
            if i < 3:
                save_file.write(str(quat[i]) + ' ')
            else:
                save_file.write(str(quat[i]) + '\n')
