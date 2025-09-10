import numpy as np
import cv2
from mediapipe.framework.formats import landmark_pb2
from PyQt5 import QtCore, QtGui


def affinemap(pfrom, pto):
    bsubmat = np.eye(3)
    ptnum = len(pfrom)
    amat = np.zeros((ptnum * 3, 9))
    for i in range(ptnum):
        amat[((i + 1) * 3 - 3) : ((i + 1) * 3), :] = np.kron(bsubmat, pfrom[i, :])
    amat = np.hstack((amat, np.tile(bsubmat, (ptnum, 1))))

    bvec = np.reshape(pto, (ptnum * 3, 1))
    x = np.linalg.lstsq(amat, bvec, rcond=None)[0]
    A = np.reshape(x[0:9], (3, 3))
    b = x[9:12]
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
        lm = landmark_pb2.NormalizedLandmark(x=p[0], y=p[1], z=p[2])
        landmarks.landmark.append(lm)
    return landmarks


def my_reg1020(predictor, x_pa):
    bsubmat = np.eye(3)
    amat = np.zeros((3, x_pa.shape[0] - 3))
    ptnum = 1
    for i in range(ptnum):
        amat[((i + 1) * 3 - 3) : ((i + 1) * 3), :] = np.kron(bsubmat, predictor[i, :])
    amat = np.hstack((amat, np.tile(bsubmat, (ptnum, 1))))
    predicted_p = np.dot(amat, x_pa)
    return predicted_p


def convert_cv_qt(cv_img):
    # https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(
        rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
    )
    p = convert_to_Qt_format.scaled(800, 500, QtCore.Qt.KeepAspectRatio)
    return QtGui.QPixmap.fromImage(p)


def interpolate_datasets_dict(data1, data2, scale):
    if set(data1.keys()) != set(data2.keys()):
        raise ValueError("Both datasets must have the same keys")

    interpolated_data = {}
    for key in data1:
        points1 = data1[key]
        points2 = data2[key]

        if isinstance(
            points1[0], list
        ):  # Check if the points are lists (for nested lists)
            if len(points1) != len(points2):
                raise ValueError(
                    f"Key '{key}' must have the same number of points in both datasets"
                )
            interpolated_points = [
                [(1 - scale) * p1 + scale * p2 for p1, p2 in zip(point1, point2)]
                for point1, point2 in zip(points1, points2)
            ]
        else:
            if len(points1) != len(points2):
                raise ValueError(
                    f"Key '{key}' must have the same number of points in both datasets"
                )
            interpolated_points = [
                (1 - scale) * p1 + scale * p2 for p1, p2 in zip(points1, points2)
            ]

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

    keys = [
        "aal",
        "aar",
        "apl",
        "apr",
        "sm",
        "cm",
        "cal_1",
        "car_1",
        "cal_2",
        "car_2",
        "cal_3",
        "car_3",
        "cal_4",
        "car_4",
        "cal_5",
        "car_5",
        "cal_6",
        "car_6",
        "cal_7",
        "car_7",
        "cpl_1",
        "cpr_1",
        "cpl_2",
        "cpr_2",
        "cpl_3",
        "cpr_3",
        "cpl_4",
        "cpr_4",
        "cpl_5",
        "cpr_5",
        "cpl_6",
        "cpr_6",
        "cpl_7",
        "cpr_7",
    ]

    lengths = [
        9,
        9,
        9,
        9,
        17,
        17,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
    ]
    reversed_dict = {}
    start_index = 0

    for key, length in zip(keys, lengths):
        reversed_dict[key] = array_list[start_index : start_index + length]
        start_index += length

    return reversed_dict


def resize_frame_keep_aspect(frame, label_width, label_height):
    height, width = frame.shape[:2]
    aspect_ratio = width / height

    if label_width / label_height > aspect_ratio:
        new_height = label_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = label_width
        new_height = int(new_width / aspect_ratio)

    new_width += new_width % 2
    new_height += new_height % 2

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame


def apply_registration_to_atlas_points(Amat, bvec, atlas10_5):
    """Apply registration to all atlas points at once"""
    point_groups = [
        "aal",
        "aar",
        "apl",
        "apr",
        "cm",
        "sm",
        "cal_1",
        "car_1",
        "cal_2",
        "car_2",
        "cal_3",
        "car_3",
        "cal_4",
        "car_4",
        "cal_5",
        "car_5",
        "cal_6",
        "car_6",
        "cal_7",
        "car_7",
        "cpl_1",
        "cpr_1",
        "cpl_2",
        "cpr_2",
        "cpl_3",
        "cpr_3",
        "cpl_4",
        "cpr_4",
        "cpl_5",
        "cpr_5",
        "cpl_6",
        "cpr_6",
        "cpl_7",
        "cpr_7",
    ]

    brain10_5p = {}
    for group in point_groups:
        if group in atlas10_5:
            brain10_5p[group] = reg1020(Amat, bvec, atlas10_5[group])

    return brain10_5p


ATLAS_MAPPING = {
    "Atlas (Colin27)": (
        "1020atlas/1020atlas_Colin27.json",
        "1020atlas/1020atlas_Colin27_5points.json",
    ),
    "Atlas (Age 20-24)": (
        "1020atlas/1020atlas_20-24Years.json",
        "1020atlas/1020atlas_20-24Years_5points.json",
    ),
    "Atlas (Age 25-29)": (
        "1020atlas/1020atlas_25-29Years.json",
        "1020atlas/1020atlas_25-29Years_5points.json",
    ),
    "Atlas (Age 30-34)": (
        "1020atlas/1020atlas_30-34Years.json",
        "1020atlas/1020atlas_30-34Years_5points.json",
    ),
    "Atlas (Age 35-39)": (
        "1020atlas/1020atlas_35-39Years.json",
        "1020atlas/1020atlas_35-39Years_5points.json",
    ),
    "Atlas (Age 40-44)": (
        "1020atlas/1020atlas_40-44Years.json",
        "1020atlas/1020atlas_40-44Years_5points.json",
    ),
    "Atlas (Age 45-49)": (
        "1020atlas/1020atlas_45-49Years.json",
        "1020atlas/1020atlas_45-49Years_5points.json",
    ),
    "Atlas (Age 50-54)": (
        "1020atlas/1020atlas_50-54Years.json",
        "1020atlas/1020atlas_50-54Years_5points.json",
    ),
    "Atlas (Age 55-59)": (
        "1020atlas/1020atlas_55-59Years.json",
        "1020atlas/1020atlas_55-59Years_5points.json",
    ),
    "Atlas (Age 60-64)": (
        "1020atlas/1020atlas_60-64Years.json",
        "1020atlas/1020atlas_60-64Years_5points.json",
    ),
    "Atlas (Age 65-69)": (
        "1020atlas/1020atlas_65-69Years.json",
        "1020atlas/1020atlas_65-69Years_5points.json",
    ),
    "Atlas (Age 70-74)": (
        "1020atlas/1020atlas_70-74Years.json",
        "1020atlas/1020atlas_70-74Years_5points.json",
    ),
    "Atlas (Age 75-79)": (
        "1020atlas/1020atlas_75-79Years.json",
        "1020atlas/1020atlas_75-79Years_5points.json",
    ),
    "Atlas (Age 80-84)": (
        "1020atlas/1020atlas_80-84Years.json",
        "1020atlas/1020atlas_80-84Years_5points.json",
    ),
}


# utils/brain_processing.py
def extract_eeg_system_points(brain10_5p):
    """Extract and organize points for different EEG systems"""
    # Indices configuration
    indices = {
        "1010": [0, 2, 4, 6, 8, 10, 12, 14, 16],
        "1020": [0, 4, 8, 12, 16],
        "aal_aar_1010": [1, 3, 5, 7],
        "aal_aar_1020": [3, 7],
        "cal_car": [1, 3, 5],
        "cal_car_6": [3],
    }

    # Extract points
    systems = {}

    # 1010 system
    cm_1010 = brain10_5p["cm"][indices["1010"]]
    sm_1010 = brain10_5p["sm"][indices["1010"]]

    systems["1010"] = {
        "cm": cm_1010,
        "sm": sm_1010,
        "front_cm": cm_1010[: len(cm_1010) // 2 + 1],
        "back_cm": cm_1010[len(cm_1010) // 2 + 1 :],
        "aal": brain10_5p["aal"][indices["aal_aar_1010"]],
        "aar": brain10_5p["aar"][indices["aal_aar_1010"]],
        "apl": brain10_5p["apl"][indices["aal_aar_1010"]],
        "apr": brain10_5p["apr"][indices["aal_aar_1010"]],
        "cal_2": brain10_5p["cal_2"][indices["cal_car"]],
        "car_2": brain10_5p["car_2"][indices["cal_car"]],
        "cal_4": brain10_5p["cal_4"][indices["cal_car"]],
        "car_4": brain10_5p["car_4"][indices["cal_car"]],
        "cal_6": brain10_5p["cal_6"][indices["cal_car_6"]],
        "car_6": brain10_5p["car_6"][indices["cal_car_6"]],
        "cpl_2": brain10_5p["cpl_2"][indices["cal_car"]],
        "cpr_2": brain10_5p["cpr_2"][indices["cal_car"]],
        "cpl_4": brain10_5p["cpl_4"][indices["cal_car"]],
        "cpr_4": brain10_5p["cpr_4"][indices["cal_car"]],
        "cpl_6": brain10_5p["cpl_6"][indices["cal_car_6"]],
        "cpr_6": brain10_5p["cpr_6"][indices["cal_car_6"]],
    }

    # 1020 system
    cm_1020 = brain10_5p["cm"][indices["1020"]]
    sm_1020 = brain10_5p["sm"][indices["1020"]]

    systems["1020"] = {
        "cm": cm_1020,
        "sm": sm_1020,
        "front_cm": cm_1020[: len(cm_1020) // 2 + 1],
        "back_cm": cm_1020[len(cm_1020) // 2 + 1 :],
        "aal": brain10_5p["aal"][indices["aal_aar_1020"]],
        "aar": brain10_5p["aar"][indices["aal_aar_1020"]],
        "apl": brain10_5p["apl"][indices["aal_aar_1020"]],
        "apr": brain10_5p["apr"][indices["aal_aar_1020"]],
    }

    return systems


def get_drawing_point_groups(brain10_5p):
    """Return all drawing point groups as individual variables"""
    systems = extract_eeg_system_points(brain10_5p)

    # 10-5 system points
    front_105_points = [
        brain10_5p["aal"],
        brain10_5p["aar"],
        brain10_5p["sm"],
        brain10_5p["cm"][0:9],
        brain10_5p["cal_1"],
        brain10_5p["car_1"],
        brain10_5p["cal_2"],
        brain10_5p["car_2"],
        brain10_5p["cal_3"],
        brain10_5p["car_3"],
        brain10_5p["cal_4"],
        brain10_5p["car_4"],
        brain10_5p["cal_5"],
        brain10_5p["car_5"],
        brain10_5p["cal_6"],
        brain10_5p["car_6"],
        brain10_5p["cal_7"],
        brain10_5p["car_7"],
    ]

    back_105_points = [
        brain10_5p["apl"],
        brain10_5p["apr"],
        brain10_5p["cm"][9:],
        brain10_5p["cpl_1"],
        brain10_5p["cpr_1"],
        brain10_5p["cpl_2"],
        brain10_5p["cpr_2"],
        brain10_5p["cpl_3"],
        brain10_5p["cpr_3"],
        brain10_5p["cpl_4"],
        brain10_5p["cpr_4"],
        brain10_5p["cpl_5"],
        brain10_5p["cpr_5"],
        brain10_5p["cpl_6"],
        brain10_5p["cpr_6"],
        brain10_5p["cpl_7"],
        brain10_5p["cpr_7"],
    ]

    # 10-10 system points
    front_1010_points = [
        systems["1010"]["sm"],
        systems["1010"]["front_cm"],
        systems["1010"]["aal"],
        systems["1010"]["aar"],
        systems["1010"]["cal_2"],
        systems["1010"]["car_2"],
        systems["1010"]["cal_4"],
        systems["1010"]["car_4"],
        systems["1010"]["cal_6"],
        systems["1010"]["car_6"],
    ]

    back_1010_points = [
        systems["1010"]["apl"],
        systems["1010"]["back_cm"],
        systems["1010"]["apr"],
        systems["1010"]["cpl_2"],
        systems["1010"]["cpr_2"],
        systems["1010"]["cpl_4"],
        systems["1010"]["cpr_4"],
        systems["1010"]["cpl_6"],
        systems["1010"]["cpr_6"],
    ]

    # 10-20 system points
    front_1020_points = [
        systems["1020"]["sm"],
        systems["1020"]["aal"],
        systems["1020"]["aar"],
        systems["1020"]["front_cm"],
    ]

    back_1020_points = [
        systems["1020"]["apl"],
        systems["1020"]["apr"],
        systems["1020"]["back_cm"],
    ]

    return {
        "front_105": front_105_points,
        "back_105": back_105_points,
        "front_1010": front_1010_points,
        "back_1010": back_1010_points,
        "front_1020": front_1020_points,
        "back_1020": back_1020_points,
    }
