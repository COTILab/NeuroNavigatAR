"""
NeuroNavigatAR Utilities

Core functionality for fNIRS/EEG electrode positioning using computer vision and machine learning.
Provides utilities for 10-5, 10-10, and 10-20 electrode systems with both geometric and ML-based
positioning methods.

This module includes:
- Mathematical transformations and registration algorithms
- MediaPipe landmark processing
- Atlas data manipulation and interpolation  
- Real-time video processing utilities
- Visualization and drawing functions

Author: Fan-Yu Yen <yen.f at northeastern.edu>
"""

__all__ = [
    # Constants
    "ATLAS_MAPPING",
    # Mathematical Functions
    "calculate_affine_transform",
    "calculate_transformed_points",
    "calculate_predicted_position",
    # Data Conversion
    "landmark2numpy",
    "numpy2landmark",
    "convert_opencv_to_pixmap",
    "convert_arrays_to_electrode_dict",
    # Atlas Processing
    "apply_registration_to_atlas_points",
    "get_drawing_point_groups",
    # Face Landmark Processing
    "extract_face_landmarks",
    "predict_cranial_points",
    # Transformation Methods
    "apply_alternative_method",
    "apply_ml_prediction_method",
    # Moving Average & Adjustments
    "update_mov_avg_buffer",
    "apply_slider_adjustment",
    # Visualization
    "render_electrode_markers",
    "render_electrode_overlay",
    # Video Processing
    "process_video_stream",
]

# =============================================================================
# Dependent Libraries
# =============================================================================

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from PyQt5 import QtCore, QtGui, QtWidgets
import copy

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers

# =============================================================================
# Configuration and Constants
# =============================================================================

def get_atlas_path(filename):
    from pathlib import Path
    return str(Path(__file__).parent.parent.parent / "data" / "atlases" / filename)


# Atlas file paths
ATLAS_MAPPING = {
    "Atlas (Colin27)": (
        get_atlas_path("1020atlas_Colin27.json"),
        get_atlas_path("1020atlas_Colin27_5points.json"),
    ),
    "Atlas (Age 20-24)": (
        get_atlas_path("1020atlas_20-24Years.json"),
        get_atlas_path("1020atlas_20-24Years_5points.json"),
    ),
    "Atlas (Age 25-29)": (
        get_atlas_path("1020atlas_25-29Years.json"),
        get_atlas_path("1020atlas_25-29Years_5points.json"),
    ),
    "Atlas (Age 30-34)": (
        get_atlas_path("1020atlas_30-34Years.json"),
        get_atlas_path("1020atlas_30-34Years_5points.json"),
    ),
    "Atlas (Age 35-39)": (
        get_atlas_path("1020atlas_35-39Years.json"),
        get_atlas_path("1020atlas_35-39Years_5points.json"),
    ),
    "Atlas (Age 40-44)": (
        get_atlas_path("1020atlas_40-44Years.json"),
        get_atlas_path("1020atlas_40-44Years_5points.json"),
    ),
    "Atlas (Age 45-49)": (
        get_atlas_path("1020atlas_45-49Years.json"),
        get_atlas_path("1020atlas_45-49Years_5points.json"),
    ),
    "Atlas (Age 50-54)": (
        get_atlas_path("1020atlas_50-54Years.json"),
        get_atlas_path("1020atlas_50-54Years_5points.json"),
    ),
    "Atlas (Age 55-59)": (
        get_atlas_path("1020atlas_55-59Years.json"),
        get_atlas_path("1020atlas_55-59Years_5points.json"),
    ),
    "Atlas (Age 60-64)": (
        get_atlas_path("1020atlas_60-64Years.json"),
        get_atlas_path("1020atlas_60-64Years_5points.json"),
    ),
    "Atlas (Age 65-69)": (
        get_atlas_path("1020atlas_65-69Years.json"),
        get_atlas_path("1020atlas_65-69Years_5points.json"),
    ),
    "Atlas (Age 70-74)": (
        get_atlas_path("1020atlas_70-74Years.json"),
        get_atlas_path("1020atlas_70-74Years_5points.json"),
    ),
    "Atlas (Age 75-79)": (
        get_atlas_path("1020atlas_75-79Years.json"),
        get_atlas_path("1020atlas_75-79Years_5points.json"),
    ),
    "Atlas (Age 80-84)": (
        get_atlas_path("1020atlas_80-84Years.json"),
        get_atlas_path("1020atlas_80-84Years_5points.json"),
    ),
}

# EEG System Configurations
EEG_SYSTEM_INDICES = {
    "1010": [0, 2, 4, 6, 8, 10, 12, 14, 16],
    "1020": [0, 4, 8, 12, 16],
    "aal_aar_1010": [1, 3, 5, 7],
    "aal_aar_1020": [3, 7],
    "cal_car": [1, 3, 5],
    "cal_car_6": [3],
}

# Face landmark indices for ML prediction
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

# Electrode group keys and lengths
ELECTRODE_KEYS = [
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

ELECTRODE_LENGTHS = [
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


# =============================================================================
# Mathematical and Registration Functions
# =============================================================================


def calculate_affine_transform(pfrom, pto):
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


def calculate_transformed_points(Amat, bvec, pts):
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


def calculate_predicted_position(predictor, x_pa):
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


def convert_opencv_to_pixmap(cv_img, target_size=None):
    """
    Convert OpenCV BGR image to QPixmap. If target_size is provided,
    scale to that size with aspect ratio kept; otherwise keep native size.

    Args:
        cv_img: OpenCV image (BGR format)

    Returns:
        QtGui.QPixmap: Qt-compatible pixmap
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qimg = QtGui.QImage(
        rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
    )
    if target_size is not None:
        qimg = qimg.scaled(
            target_size.width(),
            target_size.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
    return QtGui.QPixmap.fromImage(qimg)


def convert_arrays_to_electrode_dict(array_list):
    """
    Maps a list of arrays back into a dictionary with predefined keys and lengths.

    Args:
        array_list (list): List of arrays to be mapped back to dictionary

    Returns:
        dict: Dictionary with electrode keys mapped to their corresponding arrays
    """

    reversed_dict = {}
    start_index = 0

    for key, length in zip(ELECTRODE_KEYS, ELECTRODE_LENGTHS):
        reversed_dict[key] = array_list[start_index : start_index + length]
        start_index += length

    return reversed_dict


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
    brain10_5p = {}
    for group in ELECTRODE_KEYS:
        if group in atlas10_5:
            brain10_5p[group] = calculate_transformed_points(
                Amat, bvec, atlas10_5[group]
            )

    return brain10_5p


def extract_eeg_system_points(brain10_5p):
    """
    Extract and organize points for different EEG systems (10-10 and 10-20).

    Args:
        brain10_5p (dict): Processed brain atlas points

    Returns:
        dict: Organized points for different EEG systems
    """
    systems = {}

    # 10-10 system points
    cm_1010 = brain10_5p["cm"][EEG_SYSTEM_INDICES["1010"]]
    sm_1010 = brain10_5p["sm"][EEG_SYSTEM_INDICES["1010"]]

    systems["1010"] = {
        "cm": cm_1010,
        "sm": sm_1010,
        "front_sm": sm_1010[: len(sm_1010) // 2 + 1],
        "back_sm": sm_1010[len(sm_1010) // 2 + 1 :],
        "aal": brain10_5p["aal"][EEG_SYSTEM_INDICES["aal_aar_1010"]],
        "aar": brain10_5p["aar"][EEG_SYSTEM_INDICES["aal_aar_1010"]],
        "apl": brain10_5p["apl"][EEG_SYSTEM_INDICES["aal_aar_1010"]],
        "apr": brain10_5p["apr"][EEG_SYSTEM_INDICES["aal_aar_1010"]],
        "cal_2": brain10_5p["cal_2"][EEG_SYSTEM_INDICES["cal_car"]],
        "car_2": brain10_5p["car_2"][EEG_SYSTEM_INDICES["cal_car"]],
        "cal_4": brain10_5p["cal_4"][EEG_SYSTEM_INDICES["cal_car"]],
        "car_4": brain10_5p["car_4"][EEG_SYSTEM_INDICES["cal_car"]],
        "cal_6": brain10_5p["cal_6"][EEG_SYSTEM_INDICES["cal_car_6"]],
        "car_6": brain10_5p["car_6"][EEG_SYSTEM_INDICES["cal_car_6"]],
        "cpl_2": brain10_5p["cpl_2"][EEG_SYSTEM_INDICES["cal_car"]],
        "cpr_2": brain10_5p["cpr_2"][EEG_SYSTEM_INDICES["cal_car"]],
        "cpl_4": brain10_5p["cpl_4"][EEG_SYSTEM_INDICES["cal_car"]],
        "cpr_4": brain10_5p["cpr_4"][EEG_SYSTEM_INDICES["cal_car"]],
        "cpl_6": brain10_5p["cpl_6"][EEG_SYSTEM_INDICES["cal_car_6"]],
        "cpr_6": brain10_5p["cpr_6"][EEG_SYSTEM_INDICES["cal_car_6"]],
    }

    # 10-20 system points
    cm_1020 = brain10_5p["cm"][EEG_SYSTEM_INDICES["1020"]]
    sm_1020 = brain10_5p["sm"][EEG_SYSTEM_INDICES["1020"]]

    systems["1020"] = {
        "cm": cm_1020,
        "sm": sm_1020,
        "front_sm": sm_1020[: len(sm_1020) // 2 + 1],
        "back_sm": sm_1020[len(sm_1020) // 2 + 1 :],
        "aal": brain10_5p["aal"][EEG_SYSTEM_INDICES["aal_aar_1020"]],
        "aar": brain10_5p["aar"][EEG_SYSTEM_INDICES["aal_aar_1020"]],
        "apl": brain10_5p["apl"][EEG_SYSTEM_INDICES["aal_aar_1020"]],
        "apr": brain10_5p["apr"][EEG_SYSTEM_INDICES["aal_aar_1020"]],
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
        brain10_5p["sm"][0:9],
        brain10_5p["cm"],
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
        brain10_5p["sm"][9:],
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
        systems["1010"]["front_sm"],
        systems["1010"]["cm"],
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
        systems["1010"]["back_sm"],
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
        systems["1020"]["cm"],
        systems["1020"]["aal"],
        systems["1020"]["aar"],
        systems["1020"]["front_sm"],
    ]

    back_1020_points = [
        systems["1020"]["apl"],
        systems["1020"]["apr"],
        systems["1020"]["back_sm"],
    ]

    return {
        "front_105": front_105_points,
        "back_105": back_105_points,
        "front_1010": front_1010_points,
        "back_1010": back_1010_points,
        "front_1020": front_1020_points,
        "back_1020": back_1020_points,
    }


# =============================================================================
# Face Landmark Processing Functions
# =============================================================================


def extract_face_landmarks(results, landmark_indices):
    """
    Extract specific face landmarks for ML prediction.

    Args:
        results: MediaPipe results object containing detected landmarks
        landmark_indices (list): List of landmark indices to extract

    Returns:
        np.array: Reshaped predictor array (1, n_features) for ML models
    """
    pts = results.face_landmarks.landmark
    landmark_face_predictor = landmark_pb2.NormalizedLandmarkList(
        landmark=[pts[i] for i in landmark_indices]
    )
    predictor = landmark2numpy(landmark_face_predictor)
    return np.reshape(predictor, (1, -1))


def predict_cranial_points(predictor, model_data):
    """
    Predict cranial landmark positions using trained ML models.

    Args:
        predictor (np.array): Face landmark features for prediction
        model_data (dict): Dictionary containing trained models (x_lpa, x_rpa, x_iz, x_cz)

    Returns:
        dict: Dictionary with predicted cranial points (lpa, rpa, iz, cz)
    """
    return {
        "lpa": np.transpose(
            calculate_predicted_position(predictor, model_data["x_lpa"])
        ),
        "rpa": np.transpose(
            calculate_predicted_position(predictor, model_data["x_rpa"])
        ),
        "iz": np.transpose(calculate_predicted_position(predictor, model_data["x_iz"])),
        "cz": np.transpose(calculate_predicted_position(predictor, model_data["x_cz"])),
    }


# =============================================================================
# Transformation Method Applications
# =============================================================================


def apply_alternative_method(results, atlas10_5):
    """
    Apply alternative geometric transformation method using face landmarks.

    Args:
        results: MediaPipe results object
        atlas10_5 (dict): Atlas data for transformation reference points

    Returns:
        tuple: (Amat, bvec) transformation matrix and translation vector
    """
    pts = results.face_landmarks.landmark
    landmark_face_subset = landmark_pb2.NormalizedLandmarkList(
        landmark=[
            pts[168],  # Nasion
            pts[10],  # Upper lip center
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

    return calculate_affine_transform(
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


def apply_ml_prediction_method(results, atlas10_5_3points, model_data):
    """
    Apply ML-based prediction method to calculate transformation matrix.

    Args:
        results: MediaPipe results object
        atlas10_5_3points (dict): 3-point atlas data for reference
        model_data (dict): Dictionary containing ML models (x_lpa, x_rpa, x_iz, x_cz)

    Returns:
        tuple: (Amat, bvec) transformation matrix and translation vector
    """

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

    return calculate_affine_transform(source_points, target_points)


# =============================================================================
# Data Processing and Moving Average
# =============================================================================


def update_mov_avg_buffer(data_list, new_data, window_size):
    """
    Update moving average list and return averaged result when window is full.

    Args:
        data_list (list): Current list of data arrays for averaging
        new_data (np.array): New data array to add to the moving average
        window_size (int): Size of the moving average window

    Returns:
        tuple: (averaged_result, updated_data_list) where averaged_result is
               dict or None, and updated_data_list is the maintained window
    """
    data_list.append(new_data)

    # Maintain window size
    if len(data_list) > window_size:
        data_list = data_list[-window_size:]

    # Return averaged result when window is full
    if len(data_list) == window_size:
        averaged_data = np.mean(np.stack(data_list, axis=0), axis=0)
        return convert_arrays_to_electrode_dict(averaged_data), data_list

    return None, data_list


def apply_slider_adjustment(brain10_5p, adjustment):
    """
    Apply vertical position adjustment to all brain electrode points.

    Args:
        brain10_5p (dict): Dictionary of brain electrode positions
        adjustment (float): Vertical adjustment value to apply to y-coordinates

    Returns:
        dict: Dictionary with adjusted electrode positions
    """
    adjusted_brain10_5p = {}
    for key, points in brain10_5p.items():
        if isinstance(points, (list, np.ndarray)) and len(points) > 0:
            adjusted_points = copy.deepcopy(points)
            for i in range(len(adjusted_points)):
                if len(adjusted_points[i]) >= 2:
                    adjusted_points[i][1] = points[i][1] + adjustment
            adjusted_brain10_5p[key] = adjusted_points
        else:
            adjusted_brain10_5p[key] = points

    return adjusted_brain10_5p


# =============================================================================
# Visualization and Drawing Functions
# =============================================================================


def render_electrode_markers(
    image, points_list, front_color, back_color=None, thickness=None, radius=None
):
    """
    Draw landmarks on image with specified visual properties.

    Args:
        image: OpenCV image to draw on
        points_list (list): List of point arrays to draw
        front_color (tuple): RGB color for front-facing landmarks
        back_color (tuple, optional): RGB color for back-facing landmarks
        thickness (int, optional): Line thickness for drawing
        radius (int, optional): Circle radius for landmark points
    """
    back_color = back_color or front_color

    if points_list and len(points_list) > 0:
        # Filter out empty or invalid points
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


def render_electrode_overlay(
    image, brain10_5p, results, checkbox_states, opsize, opthick, show_back
):
    """
    Draw brain electrode landmarks on image based on system selection.

    Args:
        image: OpenCV image to draw on
        brain10_5p (dict): Processed brain atlas points
        results: MediaPipe results object
        checkbox_states (dict): Dictionary of checkbox states for different systems
        opsize (int): Size of electrode markers
        opthick (int): Thickness of electrode markers
        show_back (bool): Whether to show posterior electrode positions
    """
    if results.face_landmarks is None:
        return

    points = get_drawing_point_groups(brain10_5p)

    # Configuration for different EEG systems
    systems_config = {
        "105": {
            "checkbox": checkbox_states.get("105", False),
            "front_points": points["front_105"],
            "back_points": points["back_105"],
            "front_color": (255, 255, 0),  # Light Blue
            "back_color": (0, 255, 0),
        },
        "1010": {
            "checkbox": checkbox_states.get("1010", False),
            "front_points": points["front_1010"],
            "back_points": points["back_1010"],
            "front_color": (0, 255, 0),  # Green
            "back_color": (0, 255, 0),
        },
        "1020": {
            "checkbox": checkbox_states.get("1020", False),
            "front_points": points["front_1020"],
            "back_points": points["back_1020"],
            "front_color": (0, 125, 255),  # Orange-red
            "back_color": (0, 255, 0),
        },
    }

    # Draw each system if enabled
    for system_name, config in systems_config.items():
        if config["checkbox"]:
            # Draw front electrode positions
            render_electrode_markers(
                image,
                config["front_points"],
                config["front_color"],
                config["back_color"],
                opthick,
                opsize,
            )
            # Draw back electrode positions if enabled
            if show_back:
                render_electrode_markers(
                    image,
                    config["back_points"],
                    config["front_color"],
                    config["back_color"],
                    opthick,
                    opsize,
                )


# =============================================================================
# Video Processing Utilities
# =============================================================================


def process_video_stream(
    video_capture,
    holistic_processor,
    frame_processor_callback,
    should_continue_callback,
    frame_size=(None, None),
):
    """
    Process video stream with MediaPipe holistic detection and custom frame processing.

    Args:
        video_capture: OpenCV VideoCapture object
        holistic_processor: MediaPipe Holistic processor instance
        frame_processor_callback: Function to process each frame (image, results, iteration, state)
        should_continue_callback: Function that returns True to continue processing
        frame_size (tuple): (width, height) to resize frames, or (None, None) to skip resize

    Returns:
        dict: Final processing state from frame_processor_callback
    """
    iteration = 0
    processing_state = {}

    try:
        while video_capture.isOpened():
            # Allow GUI to process events (Qt framework compatibility)
            try:
                QtWidgets.QApplication.processEvents()
            except ImportError:
                pass

            ret, frame = video_capture.read()
            if not ret:
                break

            # Process frame with MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic_processor.process(image)

            # Prepare image for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Execute custom frame processing
            processing_state = frame_processor_callback(
                image, results, iteration, processing_state
            )

            iteration += 1

            # Check continuation condition
            if not should_continue_callback():
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

    return processing_state
