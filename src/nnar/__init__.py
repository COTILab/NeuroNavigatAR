"""
NeuroNavigatAR - Real-time EEG/fNIRS electrode positioning using computer vision
"""

__version__ = "0.1.0"

# Import main classes
from .main_gui import nnar

# Import utility functions
from .utils import (
    calculate_affine_transform,
    calculate_transformed_points,
    calculate_predicted_position,
    landmark2numpy,
    numpy2landmark,
    convert_opencv_to_pixmap,
    convert_arrays_to_electrode_dict,
    apply_registration_to_atlas_points,
    get_drawing_point_groups,
    extract_face_landmarks,
    predict_cranial_points,
    apply_alternative_method,
    apply_ml_prediction_method,
    update_mov_avg_buffer,
    apply_slider_adjustment,
    render_electrode_markers,
    render_electrode_overlay,
    process_video_stream,
)


__all__ = [
    'nnar',
    'calculate_affine_transform',
    'calculate_transformed_points',
    'calculate_predicted_position',
    'landmark2numpy',
    'numpy2landmark',
    'convert_opencv_to_pixmap',
    'convert_arrays_to_electrode_dict',
    'apply_registration_to_atlas_points',
    'get_drawing_point_groups',
    'extract_face_landmarks',
    'predict_cranial_points',
    'apply_alternative_method',
    'apply_ml_prediction_method',
    'update_mov_avg_buffer',
    'apply_slider_adjustment',
    'render_electrode_markers',
    'render_electrode_overlay',
    'process_video_stream',
]

def main():
    """Entry point for command-line usage"""
    import sys
    from PyQt5 import QtWidgets
    
    app = QtWidgets.QApplication(sys.argv)
    window = nnar()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()