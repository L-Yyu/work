from scipy.spatial.transform import Rotation as R
import numpy as np
 
# euler q
def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler
 
def euler2quaternion(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion

# euler R
def euler2rot(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix

def rot2euler(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    euler = r.as_euler('xyz', degrees=True)
    return euler

# q R
def quaternion2rot(quaternion:np.array):
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    return rotation_matrix

def rot2quaternion(rotation_matrix:np.array):
    quaternion = R.from_matrix(rotation_matrix).as_quat()
    return quaternion

# rv q
def rv2quaternion(rotation_vector:np.array):
    rotation_vector_norm = np.linalg.norm(rotation_vector)
    rotation_vector_unit = rotation_vector/rotation_vector_norm
    quaternion = np.array([np.cos(rotation_vector_norm/2), np.sin(rotation_vector_norm/2)*rotation_vector_unit])
    return quaternion

def quaternion2rv(quaternion:np.array):
    quaternion_norm = np.linalg.norm(quaternion)
    quaternion_unit = quaternion/quaternion_norm
    rotation_vector = 2*np.arccos(quaternion_unit[0])*quaternion_unit[1:]
    return rotation_vector

# rv R  tested ok
def rv2rot(rotation_vector:np.array):
    rotation_vector_norm = np.linalg.norm(rotation_vector)
    if rotation_vector_norm == 0:
        return np.identity(3)
    else:
        rotation_vector_unit = rotation_vector/rotation_vector_norm
        rotation_matrix = np.identity(3) + BuildSkewSymmetricMatrix(rotation_vector_unit)*np.sin(rotation_vector_norm) + BuildSkewSymmetricMatrix(rotation_vector_unit)**2*(1-np.cos(rotation_vector_norm))
        return rotation_matrix

def BuildSkewSymmetricMatrix(vector:np.array):
    skew_symmetric_matrix = np.array([[0, -vector[2][0], vector[1][0]],
                                      [vector[2][0], 0, -vector[0][0]],
                                      [-vector[1][0], vector[0][0], 0]])
    return skew_symmetric_matrix

def lla2ned(lla, ref_lla):
    lon = lla[0][0] # 经度
    lat = lla[1][0] # 纬度
    h = lla[2][0]
    lon_ref = ref_lla[0][0]
    lat_ref = ref_lla[1][0]
    h_ref = ref_lla[2][0]

    # 经纬度转换为弧度
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat_ref_rad = np.deg2rad(lat_ref)
    lon_ref_rad = np.deg2rad(lon_ref)

    # WGS84椭球体参数
    a = 6378137.0  # WGS84椭球体的半长轴（单位：米）
    e_sq = 0.0066943799901413156  # 第一偏心率的平方

    # lla to ecef
    N_ref = a / np.sqrt(1 - e_sq * np.sin(lat_ref_rad) ** 2)    # 卯酉圈半径
    x_ref = (N_ref + h_ref) * np.cos(lat_ref_rad) * np.cos(lon_ref_rad)
    y_ref = (N_ref + h_ref) * np.cos(lat_ref_rad) * np.sin(lon_ref_rad)
    z_ref = ((N_ref * (1 - e_sq)) + h_ref) * np.sin(lat_ref_rad)

    N = a / np.sqrt(1 - e_sq * np.sin(lat_rad) ** 2)
    x = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + h) * np.cos(lat_rad) *np.sin(lon_rad)
    z = ((N * (1 - e_sq)) + h) * np.sin(lat_rad)

    # ecef to ned
    M = np.array([[-np.sin(lon_ref_rad), np.cos(lon_ref_rad), 0],
                    [-np.sin(lat_ref_rad) * np.cos(lon_ref_rad), -np.sin(lat_ref_rad) * np.sin(lon_ref_rad), np.cos(lat_ref_rad)],
                    [np.cos(lat_ref_rad) * np.cos(lon_ref_rad), np.cos(lat_ref_rad) * np.sin(lon_ref_rad), np.sin(lat_ref_rad)]])
    enu = M @ np.array([[x - x_ref], [y - y_ref], [z - z_ref]])
    ned = np.array([[enu[1][0]], [enu[0][0]], [-enu[2][0]]])
    return ned

def ned2lla(ned, ref_lla):
    lon_ref = ref_lla[0][0]
    lat_ref = ref_lla[1][0]
    h_ref = ref_lla[2][0]

    # 经纬度转换为弧度
    lat_ref_rad = np.deg2rad(lat_ref)
    lon_ref_rad = np.deg2rad(lon_ref)

    # WGS84椭球体参数
    a = 6378137.0  # WGS84椭球体的半长轴（单位：米）
    e_sq = 0.0066943799901413156  # 第一偏心率的平方

    # 参考点在ecef下的坐标
    N_ref = a / np.sqrt(1 - e_sq * np.sin(lat_ref_rad) ** 2)
    x_ref = (N_ref + h_ref) * np.cos(lat_ref_rad) * np.cos(lon_ref_rad)
    y_ref = (N_ref + h_ref) * np.cos(lat_ref_rad) * np.sin(lon_ref_rad)
    z_ref = ((N_ref * (1 - e_sq)) + h_ref) * np.sin(lat_ref_rad)

    # ned to ecef
    enu = np.array([[ned[1][0]], [ned[0][0]], [-ned[2][0]]])
    M = np.array([[-np.sin(lon_ref_rad), np.cos(lon_ref_rad), 0],
                    [-np.sin(lat_ref_rad) * np.cos(lon_ref_rad), -np.sin(lat_ref_rad) * np.sin(lon_ref_rad), np.cos(lat_ref_rad)],
                    [np.cos(lat_ref_rad) * np.cos(lon_ref_rad), np.cos(lat_ref_rad) * np.sin(lon_ref_rad), np.sin(lat_ref_rad)]])
    delta_ecef = M.T @ enu
    x = x_ref + delta_ecef[0][0]
    y = y_ref + delta_ecef[1][0]
    z = z_ref + delta_ecef[2][0]

    # ecef to lla
    lon = np.arctan2(y, x)
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1 - e_sq))
    N = a / np.sqrt(1 - e_sq * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N
    lat = np.rad2deg(lat)
    lon = np.rad2deg(lon)
    return np.array([[lon], [lat], [h]])

    

# 计算重力加速度
def GetGravity(lla):
    lat = lla[1][0] # 纬度 deg
    h = lla[2][0]
    lat_rad = np.deg2rad(lat)
    sin_lat_sq = np.sin(lat_rad) ** 2
    return 9.7803267715 * (1 + 0.0052790414 * sin_lat_sq + 0.0000232718 * sin_lat_sq ** 2) + \
           h * (0.0000000043977311 * sin_lat_sq - 0.0000030876910891) + 0.0000000000007211 * h ** 2