import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import jdata as jd
import copy
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

from .utils import *

mp_holistic = mp.solutions.holistic  # Mediapipe Solutions

# Load atlas files
atlas10_5_3points = jd.load("data/atlases/1020atlas_Colin27.json")
atlas10_5_5points = jd.load("data/atlases/1020atlas_Colin27_5points.json")
atlas10_5 = copy.deepcopy(atlas10_5_3points)

# Load trained models
x_lpa = np.array(jd.load("data/models/x_lpa_all.json"))
x_rpa = np.array(jd.load("data/models/x_rpa_all.json"))
x_iz = np.array(jd.load("data/models/x_iz_all.json"))
x_cz = np.array(jd.load("data/models/x_cz_all.json"))

# Create model_data dictionary
model_data = {"x_lpa": x_lpa, "x_rpa": x_rpa, "x_iz": x_iz, "x_cz": x_cz}


class nnar(QtWidgets.QMainWindow):

    def __init__(self):
        super(nnar, self).__init__()

        self.setObjectName("NeuroNavigatAR")
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        # Create a main horizontal layout
        self.main_layout = QtWidgets.QHBoxLayout(self.centralwidget)

        self.setup_left_layout()
        self.setup_right_layout()

        self.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)

        self.horizontalSlider_smcm.valueChanged.connect(self.scaletext_smcm)
        self.horizontalSlider_movavg.valueChanged.connect(self.scaletext_movavg)
        self.horizontalSlider_opsize.valueChanged.connect(self.scaletext_opsize)
        self.button.clicked.connect(self.loadImage)
        self.button_close.clicked.connect(self.closeVideo)

        self.current_size = (self.label1.width(), self.label1.height())

        self.playing = False

    def setup_left_layout(self):
        # Left control panel layout
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setAlignment(QtCore.Qt.AlignTop)
        left_layout.addStretch(2)

        self._setup_buttons(left_layout)
        self._setup_sliders(left_layout)
        self._setup_radio_buttons(left_layout)
        self._setup_dropdown(left_layout)
        self.dropdown.currentIndexChanged.connect(self.selectionchange)
        self._setup_checkboxes(left_layout)

        left_layout.addStretch(6)
        self.main_layout.addLayout(left_layout, 1)

    def _setup_buttons(self, parent_layout):
        """Setup video control buttons"""
        layout = QtWidgets.QHBoxLayout()

        button_configs = [
            (
                "button",
                "Video On",
                "assets/icons/on_button.png",
                "Click this button to start the video.",
                self.loadImage,
            ),
            (
                "button_close",
                "Video Stop",
                "assets/icons/off_button.png",
                "Click this button to stop the video.",
                self.closeVideo,
            ),
        ]

        for attr_name, text, icon_path, tooltip, callback in button_configs:
            button = QtWidgets.QPushButton(text)
            button.setIcon(QtGui.QIcon(icon_path))
            button.setIconSize(QtCore.QSize(32, 32))
            button.setToolTip(tooltip)
            button.clicked.connect(callback)
            setattr(self, attr_name, button)
            layout.addWidget(button)

        parent_layout.addLayout(layout)
        parent_layout.addStretch(1)

    def _setup_sliders(self, parent_layout):
        """Setup all slider controls"""
        slider_configs = [
            {
                "name": "smcm",
                "label": "Model\nPositions:",
                "min_val": 0,
                "max_val": 100,
                "default": 50,
                "display_val": "0.5",
                "tooltip": "Adjust head model position vertically.",
                "callback": self.scaletext_smcm,
            },
            {
                "name": "movavg",
                "label": "Moving\nAverage:",
                "min_val": 1,
                "max_val": 10,
                "default": 1,
                "display_val": "1",
                "tooltip": "Window numbers used to average to increase stability.",
                "callback": self.scaletext_movavg,
            },
            {
                "name": "opsize",
                "label": "Optode\nSize:",
                "min_val": 1,
                "max_val": 10,
                "default": 3,
                "display_val": "3",
                "tooltip": "Size of displayed optodes.",
                "callback": self.scaletext_opsize,
            },
        ]

        for config in slider_configs:
            self._create_slider(parent_layout, config)

    def _create_slider(self, parent_layout, config):
        """Create a single slider with label"""
        layout = QtWidgets.QHBoxLayout()

        # Label
        label_text = QtWidgets.QLabel(config["label"])
        label_text.setToolTip(config["tooltip"])
        layout.addWidget(label_text)

        # Slider
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(config["min_val"])
        slider.setMaximum(config["max_val"])
        slider.setValue(config["default"])
        slider.setSingleStep(1)
        slider.setToolTip(config["tooltip"])
        slider.valueChanged.connect(config["callback"])
        layout.addWidget(slider)

        # Value label
        value_label = QtWidgets.QLabel(config["display_val"])
        value_label.setToolTip(config["tooltip"])
        layout.addWidget(value_label)

        # Store references
        setattr(self, f"horizontalSlider_{config['name']}", slider)
        setattr(self, f"label_{config['name']}", value_label)
        setattr(
            self,
            f"label_text{'' if config['name'] == 'smcm' else '2' if config['name'] == 'movavg' else '3'}",
            label_text,
        )

        parent_layout.addLayout(layout)

    def _setup_radio_buttons(self, parent_layout):
        """Setup radio button group"""
        layout = QtWidgets.QHBoxLayout()

        self.radioButton_3p = QtWidgets.QRadioButton("3 point")
        self.radioButton_5p = QtWidgets.QRadioButton("5 point")
        self.radioButton_5p.setChecked(True)

        layout.addWidget(self.radioButton_3p)
        layout.addWidget(self.radioButton_5p)

        parent_layout.addLayout(layout)
        parent_layout.addStretch(1)

    def _setup_dropdown(self, parent_layout):
        """Setup atlas selection dropdown"""
        layout = QtWidgets.QVBoxLayout()

        self.dropdown = QtWidgets.QComboBox()
        self.dropdown.addItems(list(ATLAS_MAPPING.keys()))
        self.dropdown.setToolTip(
            "Choose different atlas model for providing head surface."
        )

        layout.addWidget(self.dropdown)
        parent_layout.addLayout(layout)
        parent_layout.addStretch(1)

    def _setup_checkboxes(self, parent_layout):
        """Setup all checkbox controls"""
        # Posterior display checkbox
        self._create_centered_checkbox(
            parent_layout, "checkbox_backhead", "Display posterior part optodes"
        )
        parent_layout.addStretch(1)

        # System checkboxes
        checkbox_configs = [
            ("Display 10-5 points", "checkbox_105", False),
            ("Display 10-10 points", "checkbox_1010", True),
            ("Display 10-20 points", "checkbox_1020", False),
        ]

        layout = QtWidgets.QVBoxLayout()
        for text, attr_name, default_checked in checkbox_configs:
            self._create_centered_checkbox(layout, attr_name, text, default_checked)

        parent_layout.addLayout(layout)
        parent_layout.addStretch(1)

        # Alternative method checkbox
        self._create_centered_checkbox(
            parent_layout, "checkbox_old", "Use alternative method"
        )

    def _create_centered_checkbox(self, parent_layout, attr_name, text, checked=False):
        """Create a centered checkbox"""
        layout = QtWidgets.QHBoxLayout()
        layout.addStretch(1)

        checkbox = QtWidgets.QCheckBox(text)
        checkbox.setChecked(checked)
        setattr(self, attr_name, checkbox)

        layout.addWidget(checkbox)
        layout.addStretch(1)
        parent_layout.addLayout(layout)

    def setup_right_layout(self):
        # # Right side layout (Image Display)
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setAlignment(QtCore.Qt.AlignCenter)

        # Image Display
        self.label1 = QtWidgets.QLabel()
        self.label1.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.label1.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(self.label1)

        self.main_layout.addLayout(right_layout, 3)

    def selectionchange(self, i):
        """Select atlas based on dropdown menu"""
        global atlas10_5, atlas10_5_3points, atlas10_5_5points  # Keep these for now

        selected_text = self.dropdown.itemText(i)
        if selected_text in ATLAS_MAPPING:
            atlas_file, atlas_file_5points = ATLAS_MAPPING[selected_text]

            # Update global variables (not ideal, but maintaining current structure)
            import jdata as jd

            atlas10_5 = jd.load(atlas_file)
            atlas10_5_3points = jd.load(atlas_file)
            if atlas_file_5points:  # Check if 5-point file exists
                atlas10_5_5points = jd.load(atlas_file_5points)
            else:
                atlas10_5_5points = None

    def update_pixmap(self, width, height):
        if self.playing:
            return
        pixmap = QtGui.QPixmap("assets/icons/NeuroNavigatAR_logo.png")
        scaled_pixmap = pixmap.scaled(
            int(width),
            int(height),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.label1.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        new_width = self.size().width() // 3 * 2
        new_height = self.size().height() // 1.2
        self.update_pixmap(new_width, new_height)
        super(nnar, self).resizeEvent(event)

    def scaletext_smcm(self, value):
        self.label_smcm.setText(str(value / 100))

    def scaletext_movavg(self, value):
        self.label_movavg.setText(str(value))

    def scaletext_opsize(self, value):
        self.label_opsize.setText(str(value))

    def closeVideo(self):
        self.videoStatus = False

    def Brain_LMs_plotting(self, brain10_5p, results, image):
        """Plot brain landmarks on image"""
        checkbox_states = {
            "105": self.checkbox_105.isChecked(),
            "1010": self.checkbox_1010.isChecked(),
            "1020": self.checkbox_1020.isChecked(),
        }

        opsize = self.horizontalSlider_opsize.value()
        opthick = opsize - 1
        show_back = self.checkbox_backhead.isChecked()

        draw_brain_landmarks(
            image, brain10_5p, results, checkbox_states, opsize, opthick, show_back
        )

    def livefacemask(self, image, results, moving_averaged_Brain_LMs_list, iteration):
        if results.face_landmarks is not None:
            # Choose atlas based on radio button
            atlas10_5 = copy.deepcopy(
                atlas10_5_3points
                if self.radioButton_3p.isChecked()
                else atlas10_5_5points
            )

            if self.checkbox_old.isChecked():
                # Use alternative method
                Amat, bvec = apply_alternative_method(results, atlas10_5)
            else:
                # Use ML prediction method
                Amat, bvec = apply_ml_prediction_method(
                    results, atlas10_5_3points, model_data
                )

            brain10_5p = apply_registration_to_atlas_points(Amat, bvec, atlas10_5)
            brain10_5p = apply_slider_adjustment(
                brain10_5p, adjustment=(self.horizontalSlider_smcm.value() - 50) / 100
            )

            # Handle moving average
            brain10_5p_array = np.concatenate(list(brain10_5p.values()), axis=0)
            averaged_result, moving_averaged_Brain_LMs_list = update_moving_average(
                moving_averaged_Brain_LMs_list,
                brain10_5p_array,
                window_size=self.horizontalSlider_movavg.value(),
            )

            if averaged_result:
                self.Brain_LMs_plotting(averaged_result, results, image)

        return moving_averaged_Brain_LMs_list

    def loadImage(self):
        if self.playing:
            return
        self.playing = True

        self.videoStatus = True
        vid = cv2.VideoCapture(0)

        try:
            # Define frame processor
            def process_frame(image, results, iteration, state):
                # Get or initialize moving average list
                moving_list = state.get("moving_averaged_Brain_LMs_list", [])

                # Process landmarks
                moving_list = self.livefacemask(image, results, moving_list, iteration)

                # Display the result
                flipped_image = cv2.flip(image, 1)
                self.label1.setPixmap(convert_cv_qt(flipped_image, self.label1.size()))

                # Return updated state
                return {"moving_averaged_Brain_LMs_list": moving_list}

            # Define continue condition
            def should_continue():
                return self.videoStatus

            # Process video stream
            with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                smooth_landmarks=True,
            ) as holistic:
                process_video_stream(
                    vid,
                    holistic,
                    process_frame,
                    should_continue,
                    (self.label1.width(), self.label1.height()),
                )

        finally:
            vid.release()
            cv2.destroyAllWindows()  # Clean up any OpenCV windows
