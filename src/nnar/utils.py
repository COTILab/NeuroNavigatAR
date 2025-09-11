import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from PyQt5 import QtCore, QtGui

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers

# =============================================================================
# Configuration and Constants
# =============================================================================

ATLAS_MAPPING = {
    "Atlas (Colin27)": (
        "data/atlases/1020atlas_Colin27.json",
        "data/atlases/1020atlas_Colin27_5points.json",
    ),
    "Atlas (Age 20-24)": (
        "data/atlases/1020atlas_20-24Years.json",
        "data/atlases/1020atlas_20-24Years_5points.json",
    ),
    "Atlas (Age 25-29)": (
        "data/atlases/1020atlas_25-29Years.json",
        "data/atlases/1020atlas_25-29Years_5points.json",
    ),
    "Atlas (Age 30-34)": (
        "data/atlases/1020atlas_30-34Years.json",
        "data/atlases/1020atlas_30-34Years_5points.json",
    ),
    "Atlas (Age 35-39)": (
        "data/atlases/1020atlas_35-39Years.json",
        "data/atlases/1020atlas_35-39Years_5points.json",
    ),
    "Atlas (Age 40-44)": (
        "data/atlases/1020atlas_40-44Years.json",
        "data/atlases/1020atlas_40-44Years_5points.json",
    ),
    "Atlas (Age 45-49)": (
        "data/atlases/1020atlas_45-49Years.json",
        "data/atlases/1020atlas_45-49Years_5points.json",
    ),
    "Atlas (Age 50-54)": (
        "data/atlases/1020atlas_50-54Years.json",
        "data/atlases/1020atlas_50-54Years_5points.json",
    ),
    "Atlas (Age 55-59)": (
        "data/atlases/1020atlas_55-59Years.json",
        "data/atlases/1020atlas_55-59Years_5points.json",
    ),
    "Atlas (Age 60-64)": (
        "data/atlases/1020atlas_60-64Years.json",
        "data/atlases/1020atlas_60-64Years_5points.json",
    ),
    "Atlas (Age 65-69)": (
        "data/atlases/1020atlas_65-69Years.json",
        "data/atlases/1020atlas_65-69Years_5points.json",
    ),
    "Atlas (Age 70-74)": (
        "data/atlases/1020atlas_70-74Years.json",
        "data/atlases/1020atlas_70-74Years_5points.json",
    ),
    "Atlas (Age 75-79)": (
        "data/atlases/1020atlas_75-79Years.json",
        "data/atlases/1020atlas_75-79Years_5points.json",
    ),
    "Atlas (Age 80-84)": (
        "data/atlases/1020atlas_80-84Years.json",
        "data/atlases/1020atlas_80-84Years_5points.json",
    ),
}

# =============================================================================
# Mathematical and Registration Functions
# =============================================================================


def affinemap(pfrom, pto):
    """
    Calculate affine transformation mapping from source to target points.

    Args:
        pfrom (np.array): Source points array
        pto (np.array): Target points array

    Returns:
        list: [A, b] where A is transformation matrix (3x3) and b is translation vector (3x1)
    """
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
    """
    Apply 10-20 registration transformation to points.

    Args:
        Amat (np.array): Transformation matrix (3x3)
        bvec (np.array): Translation vector
        pts (list): List of points to transform

    Returns:
        np.array: Transformed points
    """
    newpt = np.matmul(Amat, (np.array(pts)).T)
    newpt = newpt + np.tile(bvec, (1, len(pts)))
    newpt = newpt.T
    return newpt


def my_reg1020(predictor, x_pa):
    """
    Custom 10-20 registration using predictor and parameter array.

    Args:
        predictor (np.array): Predictor array
        x_pa (np.array): Parameter array

    Returns:
        np.array: Predicted point coordinates
    """
    bsubmat = np.eye(3)
    amat = np.zeros((3, x_pa.shape[0] - 3))
    ptnum = 1
    for i in range(ptnum):
        amat[((i + 1) * 3 - 3) : ((i + 1) * 3), :] = np.kron(bsubmat, predictor[i, :])
    amat = np.hstack((amat, np.tile(bsubmat, (ptnum, 1))))
    predicted_p = np.dot(amat, x_pa)
    return predicted_p


# =============================================================================
# Data Conversion Utilities
# =============================================================================


def landmark2numpy(landmarks):
    """
    Convert MediaPipe landmarks to numpy array.

    Args:
        landmarks: MediaPipe landmark object

    Returns:
        np.array: Array of [x, y, z] coordinates
    """
    pts = []
    for p in landmarks.landmark:
        pts.append([p.x, p.y, p.z])
    return np.array(pts)


def numpy2landmark(pts):
    """
    Convert numpy array to MediaPipe landmarks format.

    Args:
        pts (np.array): Array of [x, y, z] coordinates

    Returns:
        landmark_pb2.NormalizedLandmarkList: MediaPipe landmark object
    """
    landmarks = landmark_pb2.NormalizedLandmarkList(landmark=[])
    for p in pts:
        lm = landmark_pb2.NormalizedLandmark(x=p[0], y=p[1], z=p[2])
        landmarks.landmark.append(lm)
    return landmarks


def convert_cv_qt(cv_img):
    """
    Convert OpenCV image to QPixmap for PyQt display.

    Args:
        cv_img: OpenCV image (BGR format)

    Returns:
        QtGui.QPixmap: Qt-compatible pixmap scaled to 800x500
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(
        rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
    )
    p = convert_to_Qt_format.scaled(800, 500, QtCore.Qt.KeepAspectRatio)
    return QtGui.QPixmap.fromImage(p)


def map_arrays_to_dict(array_list):
    """
    Maps a list of arrays back into a dictionary with predefined keys and lengths.

    Args:
        array_list (list): List of arrays to be mapped back to dictionary

    Returns:
        dict: Dictionary with electrode keys mapped to their corresponding arrays
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
        17,  # Basic electrode groups
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
        7,  # CAL/CAR groups
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
        7,  # CPL/CPR groups
    ]

    reversed_dict = {}
    start_index = 0

    for key, length in zip(keys, lengths):
        reversed_dict[key] = array_list[start_index : start_index + length]
        start_index += length

    return reversed_dict


# =============================================================================
# Image Processing Utilities
# =============================================================================


def resize_frame_keep_aspect(frame, label_width, label_height):
    """
    Resize frame while maintaining aspect ratio to fit within label dimensions.

    Args:
        frame: Input image frame
        label_width (int): Target label width
        label_height (int): Target label height

    Returns:
        np.array: Resized frame maintaining aspect ratio
    """
    height, width = frame.shape[:2]
    aspect_ratio = width / height

    # Calculate new dimensions maintaining aspect ratio
    if label_width / label_height > aspect_ratio:
        new_height = label_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = label_width
        new_height = int(new_width / aspect_ratio)

    # Ensure even dimensions for video compatibility
    new_width += new_width % 2
    new_height += new_height % 2

    # Resize the frame
    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame


# =============================================================================
# Data Processing and Interpolation
# =============================================================================


def interpolate_datasets_dict(data1, data2, scale):
    """
    Interpolate between two datasets with matching keys.

    Args:
        data1 (dict): First dataset
        data2 (dict): Second dataset
        scale (float): Interpolation factor (0.0 = data1, 1.0 = data2)

    Returns:
        dict: Interpolated dataset

    Raises:
        ValueError: If datasets don't have matching keys or point counts
    """
    if set(data1.keys()) != set(data2.keys()):
        raise ValueError("Both datasets must have the same keys")

    interpolated_data = {}
    for key in data1:
        points1 = data1[key]
        points2 = data2[key]

        if isinstance(points1[0], list):  # Handle nested lists
            if len(points1) != len(points2):
                raise ValueError(
                    f"Key '{key}' must have the same number of points in both datasets"
                )
            interpolated_points = [
                [(1 - scale) * p1 + scale * p2 for p1, p2 in zip(point1, point2)]
                for point1, point2 in zip(points1, points2)
            ]
        else:  # Handle flat lists
            if len(points1) != len(points2):
                raise ValueError(
                    f"Key '{key}' must have the same number of points in both datasets"
                )
            interpolated_points = [
                (1 - scale) * p1 + scale * p2 for p1, p2 in zip(points1, points2)
            ]

        interpolated_data[key] = interpolated_points

    return interpolated_data


# =============================================================================
# Atlas and Brain Processing
# =============================================================================


def apply_registration_to_atlas_points(Amat, bvec, atlas10_5):
    """
    Apply registration transformation to all atlas points.

    Args:
        Amat (np.array): Transformation matrix
        bvec (np.array): Translation vector
        atlas10_5 (dict): Atlas data with electrode positions

    Returns:
        dict: Transformed atlas points for all electrode groups
    """
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


def extract_eeg_system_points(brain10_5p):
    """
    Extract and organize points for different EEG systems (10-10 and 10-20).

    Args:
        brain10_5p (dict): Processed brain atlas points

    Returns:
        dict: Organized points for different EEG systems
    """
    # Indices configuration for different systems
    indices = {
        "1010": [0, 2, 4, 6, 8, 10, 12, 14, 16],
        "1020": [0, 4, 8, 12, 16],
        "aal_aar_1010": [1, 3, 5, 7],
        "aal_aar_1020": [3, 7],
        "cal_car": [1, 3, 5],
        "cal_car_6": [3],
    }

    systems = {}

    # 10-10 system points
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

    # 10-20 system points
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
    """
    Return all drawing point groups organized by EEG system and view (front/back).

    Args:
        brain10_5p (dict): Processed brain atlas points

    Returns:
        dict: Dictionary containing point groups for different systems and views
    """
    systems = extract_eeg_system_points(brain10_5p)

    # 10-5 system points (full resolution)
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

    # 10-10 system points (intermediate resolution)
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

    # 10-20 system points (standard clinical resolution)
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


def update_moving_average(data_list, new_data, window_size):
    """Update moving average list and return averaged result if ready"""
    data_list.append(new_data)

    if len(data_list) > window_size:
        data_list = data_list[-window_size:]

    if len(data_list) == window_size:
        averaged_data = np.mean(np.stack(data_list, axis=0), axis=0)
        return map_arrays_to_dict(averaged_data), data_list

    return None, data_list


def apply_alternative_method(results, atlas10_5):
    """Apply alternative transformation method"""
    pts = results.face_landmarks.landmark
    landmark_face_subset = landmark_pb2.NormalizedLandmarkList(
        landmark=[
            pts[168],
            pts[10],
            {
                "x": 2 * pts[234].x - pts[227].x,
                "y": 2 * pts[234].y - pts[227].y,
                "z": 2 * pts[234].z - pts[227].z,
            },
            {
                "x": 2 * pts[454].x - pts[447].x,
                "y": 2 * pts[454].y - pts[447].y,
                "z": 2 * pts[454].z - pts[447].z,
            },
        ]
    )

    return affinemap(
        np.array(
            [
                atlas10_5["nz"],
                atlas10_5["sm"][1],
                atlas10_5["rpa"],
                atlas10_5["lpa"],
            ]
        ),
        landmark2numpy(landmark_face_subset),
    )


def extract_face_landmarks(results, landmark_indices):
    """Extract specific face landmarks"""
    pts = results.face_landmarks.landmark
    landmark_face_predictor = landmark_pb2.NormalizedLandmarkList(
        landmark=[pts[i] for i in landmark_indices]
    )
    predictor = landmark2numpy(landmark_face_predictor)
    return np.reshape(predictor, (1, -1))


def predict_cranial_points(predictor, model_data):
    """Predict cranial points using models"""
    return {
        "lpa": np.transpose(my_reg1020(predictor, model_data["x_lpa"])),
        "rpa": np.transpose(my_reg1020(predictor, model_data["x_rpa"])),
        "iz": np.transpose(my_reg1020(predictor, model_data["x_iz"])),
        "cz": np.transpose(my_reg1020(predictor, model_data["x_cz"])),
    }


def apply_ml_prediction_method(results, atlas10_5_3points, model_data):
    """
    Apply ML-based prediction method to calculate transformation matrix.

    Args:
        results: MediaPipe results object
        atlas10_5_3points: 3-point atlas data
        model_data: Dictionary containing ML models (x_lpa, x_rpa, x_iz, x_cz)

    Returns:
        tuple: (Amat, bvec) transformation matrix and translation vector
    """
    # Define face landmark indices for prediction
    FACE_LANDMARK_INDICES = [
        33,
        133,
        168,
        362,
        263,
        4,
        61,
        291,
        10,
        332,
        389,
        323,
        397,
        378,
        152,
        149,
        172,
        93,
        162,
        103,
    ]

    # Extract face landmarks for prediction
    predictor = extract_face_landmarks(results, FACE_LANDMARK_INDICES)

    # Predict cranial points using ML models
    predictions = predict_cranial_points(predictor, model_data)

    # Create nasion landmark for transformation
    nz = landmark_pb2.NormalizedLandmarkList(
        landmark=[results.face_landmarks.landmark[168]]
    )

    # Calculate affine transformation
    source_points = np.array(
        [
            atlas10_5_3points["nz"],
            atlas10_5_3points["lpa"],
            atlas10_5_3points["rpa"],
            atlas10_5_3points["iz"],
            atlas10_5_3points["cz"],
        ]
    )

    target_points = np.array(
        [
            landmark2numpy(nz)[0],
            predictions["lpa"][0],
            predictions["rpa"][0],
            predictions["iz"][0],
            predictions["cz"][0],
        ]
    )

    return affinemap(source_points, target_points)


def draw_landmarks_with_specs(
    image, points_list, front_color, back_color=None, thickness=None, radius=None
):
    """Draw landmarks with specified colors and styles"""
    back_color = back_color or front_color

    if points_list and len(points_list) > 0:
        valid_points = [p for p in points_list if p is not None and len(p) > 0]
        if valid_points:
            combined_points = np.vstack(valid_points)
            mp_drawing.draw_landmarks(
                image,
                numpy2landmark(combined_points),
                None,
                mp_drawing.DrawingSpec(
                    color=front_color, thickness=thickness, circle_radius=radius
                ),
                mp_drawing.DrawingSpec(
                    color=back_color, thickness=thickness, circle_radius=radius
                ),
            )


def draw_brain_landmarks(
    image, brain10_5p, results, checkbox_states, opsize, opthick, show_back
):
    """Draw brain landmarks based on checkbox states"""
    if results.face_landmarks is None:
        return

    points = get_drawing_point_groups(brain10_5p)

    # Configuration for different systems
    systems_config = {
        "105": {
            "checkbox": checkbox_states.get("105", False),
            "front_points": points["front_105"],
            "back_points": points["back_105"],
            "front_color": (255, 255, 0),
            "back_color": (0, 255, 0),
        },
        "1010": {
            "checkbox": checkbox_states.get("1010", False),
            "front_points": points["front_1010"],
            "back_points": points["back_1010"],
            "front_color": (0, 255, 0),
            "back_color": (0, 255, 0),
        },
        "1020": {
            "checkbox": checkbox_states.get("1020", False),
            "front_points": points["front_1020"],
            "back_points": points["back_1020"],
            "front_color": (0, 125, 255),
            "back_color": (0, 255, 0),
        },
    }

    # Draw each system if enabled
    for system_name, config in systems_config.items():
        if config["checkbox"]:
            # Draw front points
            draw_landmarks_with_specs(
                image,
                config["front_points"],
                config["front_color"],
                config["back_color"],
                opthick,
                opsize,
            )
            # Draw back points if enabled
            if show_back:
                draw_landmarks_with_specs(
                    image,
                    config["back_points"],
                    config["front_color"],
                    config["back_color"],
                    opthick,
                    opsize,
                )
