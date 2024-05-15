import numpy as np
import cv2
from mediapipe.framework.formats import landmark_pb2
from PyQt5 import QtCore, QtGui, QtWidgets

def affinemap(pfrom, pto):
    bsubmat=np.eye(3)
    ptnum=len(pfrom)
    amat=np.zeros((ptnum*3,9))
    for i in range(ptnum):
        amat[((i+1)*3-3):((i+1)*3),:]=np.kron(bsubmat,pfrom[i,:])
    amat=np.hstack((amat,np.tile(bsubmat,(ptnum,1))))
    
    bvec=np.reshape(pto,(ptnum*3,1))
    x=np.linalg.lstsq(amat, bvec, rcond=None)[0]
    A=np.reshape(x[0:9],(3,3))
    b=x[9:12]
    return [A, b]

def reg1020(Amat, bvec, pts):
    newpt = np.matmul(Amat, (np.array(pts)).T)
    newpt = newpt + np.tile(bvec, (1, len(pts)))
    newpt = newpt.T
    return newpt

def landmark2numpy(landmarks):
    pts = []
    for p in landmarks.landmark:
        pts.append([p.x, p.y, p.z])
    return np.array(pts)

def numpy2landmark(pts):
    landmarks = landmark_pb2.NormalizedLandmarkList(landmark=[])
    for p in pts:
#          print(p)
         lm =  landmark_pb2.NormalizedLandmark(x=p[0], y=p[1], z=p[2])
         landmarks.landmark.append(lm)
    return landmarks

def my_reg1020(predictor, x_pa):
    bsubmat = np.eye(3);
    amat = np.zeros((3,x_pa.shape[0]-3));
    ptnum = 1;
    for i in range(ptnum):
        amat[((i+1)*3-3):((i+1)*3),:]=np.kron(bsubmat,predictor[i,:])
    amat=np.hstack((amat,np.tile(bsubmat,(ptnum,1))))
    predicted_p = np.dot(amat, x_pa)
    return predicted_p

def convert_cv_qt(cv_img):
    # https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(800,500, QtCore.Qt.KeepAspectRatio)
    return QtGui.QPixmap.fromImage(p)


def interpolate_datasets_dict(data1, data2, scale):
    if set(data1.keys()) != set(data2.keys()):
        raise ValueError("Both datasets must have the same keys")

    interpolated_data = {}
    for key in data1:
        points1 = data1[key]
        points2 = data2[key]

        if isinstance(points1[0], list):  # Check if the points are lists (for nested lists)
            if len(points1) != len(points2):
                raise ValueError(f"Key '{key}' must have the same number of points in both datasets")
            interpolated_points = [[(1 - scale) * p1 + scale * p2 for p1, p2 in zip(point1, point2)] for point1, point2 in zip(points1, points2)]
        else:
            if len(points1) != len(points2):
                raise ValueError(f"Key '{key}' must have the same number of points in both datasets")
            interpolated_points = [(1 - scale) * p1 + scale * p2 for p1, p2 in zip(points1, points2)]

        interpolated_data[key] = interpolated_points

    return interpolated_data

def map_arrays_to_dict(array_list):
    """
    Maps a list of arrays back into a dictionary given the keys and lengths of each array.
    
    :param array_list: List of arrays to be mapped back to a dictionary.
    :param keys: List of keys corresponding to each array.
    :param lengths: List of lengths for each array.
    :return: Dictionary with keys mapped to their corresponding arrays.
    """

    keys = ['aal', 'aar', 'apl', 'apr', 'sm', 'cm', 
            'cal_1', 'car_1', 'cal_2', 'car_2', 'cal_3', 'car_3', 'cal_4', 'car_4', 'cal_5', 'car_5', 'cal_6', 'car_6', 'cal_7', 'car_7', 
            'cpl_1', 'cpr_1', 'cpl_2', 'cpr_2', 'cpl_3', 'cpr_3', 'cpl_4', 'cpr_4', 'cpl_5', 'cpr_5', 'cpl_6', 'cpr_6', 'cpl_7', 'cpr_7',]

    lengths = [9,9,9,9,17,17,
              7,7,7,7,7,7,7,7,7,7,7,7,7,7,
              7,7,7,7,7,7,7,7,7,7,7,7,7,7,]
    reversed_dict = {}
    start_index = 0
    
    for key, length in zip(keys, lengths):
        reversed_dict[key] = array_list[start_index:start_index + length]
        start_index += length
    
    return reversed_dict
