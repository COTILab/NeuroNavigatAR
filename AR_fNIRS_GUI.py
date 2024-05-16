import cv2
import numpy as np
import mediapipe as mp # Import mediapipe
from mediapipe.framework.formats import landmark_pb2
import jdata as jd
import copy
from PyQt5 import QtCore, QtGui, QtWidgets

import sys
sys.path.append('GUI_functions')

from livefacemaskGUI import affinemap, reg1020, landmark2numpy, numpy2landmark, my_reg1020, convert_cv_qt, interpolate_datasets_dict, map_arrays_to_dict

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
atlas10_5 = jd.load('1020atlas/1020atlas_Colin27.json') # default
atlas10_5_3points = jd.load('1020atlas/1020atlas_Colin27.json')
atlas10_5_5points = jd.load('1020atlas/1020atlas_Colin27_5points.json') # default

import scipy.io
lpa_mat = scipy.io.loadmat('Trained model/x_lpa_all.mat')
x_lpa = lpa_mat['x_lpa']
rpa_mat = scipy.io.loadmat('Trained model/x_rpa_all.mat')
x_rpa = rpa_mat['x_rpa']

iz_mat = scipy.io.loadmat('Trained model/x_iz_all.mat')
x_iz = iz_mat['x_iz']
cz_mat = scipy.io.loadmat('Trained model/x_cz_all.mat')
x_cz = cz_mat['x_cz']

from PyQt5 import QtWidgets, QtCore, QtGui
import sys

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setObjectName("MainWindow")
        # self.resize(1200, 600)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        # Create a main horizontal layout
        self.main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        # self.main_layout = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self.centralwidget)
        
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

    # def closeEvent(self, event):
    #     reply = QtWidgets.QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
    #                                            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)

    #     if reply == QtWidgets.QMessageBox.Yes:
    #         event.accept()
    #         print('Window closed')
    #     else:
    #         event.ignore()

    def setup_left_layout(self):
        # Left control panel layout
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setAlignment(QtCore.Qt.AlignTop)  # Ensure alignment is top
        left_layout.addStretch(2)

        ###### Button Layout ######
        button_layout = QtWidgets.QHBoxLayout()
        # self.button = QtWidgets.QPushButton("Turn on Video")
        # self.button_close = QtWidgets.QPushButton("Turn off Video")
        # button_layout.addWidget(self.button)
        # button_layout.addWidget(self.button_close)

        self.button = QtWidgets.QPushButton("Video On")
        self.button.setIcon(QtGui.QIcon('GUI_icon/on_button.png'))  # Replace 'path/to/on_icon.png' with the actual path
        self.button.setIconSize(QtCore.QSize(32, 32))  # Adjust icon size as needed
        button_layout.addWidget(self.button)
        
        self.button_close = QtWidgets.QPushButton("Video Stop")
        self.button_close.setIcon(QtGui.QIcon('GUI_icon/off_button.png'))  # Replace 'path/to/off_icon.png' with the actual path
        self.button_close.setIconSize(QtCore.QSize(32, 32))  # Adjust icon size as needed
        button_layout.addWidget(self.button_close)
        self.button.setToolTip("Click this button to start the video.")
        self.button_close.setToolTip("Click this button to stop the video.")
        
        left_layout.addLayout(button_layout)
        left_layout.addStretch(1)

        ###### Slider Layouts ######
        # smcm adjustment slider
        slider_layout = QtWidgets.QHBoxLayout()
        self.label_text = QtWidgets.QLabel("Model\nPositions:")
        slider_layout.addWidget(self.label_text)
        
        self.horizontalSlider_smcm = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.horizontalSlider_smcm.setMinimum(0)
        self.horizontalSlider_smcm.setMaximum(100)
        self.horizontalSlider_smcm.setValue(50)
        self.horizontalSlider_smcm.setSingleStep(1)
        slider_layout.addWidget(self.horizontalSlider_smcm)

        self.label_smcm = QtWidgets.QLabel("0.5")
        slider_layout.addWidget(self.label_smcm)
        left_layout.addLayout(slider_layout)

        # moving average slider
        slider_layout_2 = QtWidgets.QHBoxLayout()
        self.label_text2 = QtWidgets.QLabel("Moving\nAverage:")
        slider_layout_2.addWidget(self.label_text2)
        
        self.horizontalSlider_movavg = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.horizontalSlider_movavg.setMinimum(1)
        self.horizontalSlider_movavg.setMaximum(10)
        self.horizontalSlider_movavg.setValue(1)
        slider_layout_2.addWidget(self.horizontalSlider_movavg)

        self.label_movavg = QtWidgets.QLabel("1")
        slider_layout_2.addWidget(self.label_movavg)
        left_layout.addLayout(slider_layout_2)

        # optode size slider
        slider_layout_3 = QtWidgets.QHBoxLayout()
        self.label_text3 = QtWidgets.QLabel("Optode\nSize:")
        slider_layout_3.addWidget(self.label_text3)
        
        self.horizontalSlider_opsize = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.horizontalSlider_opsize.setMinimum(1)
        self.horizontalSlider_opsize.setMaximum(10)
        self.horizontalSlider_opsize.setValue(3)
        slider_layout_3.addWidget(self.horizontalSlider_opsize)

        self.label_opsize = QtWidgets.QLabel("3")
        slider_layout_3.addWidget(self.label_opsize)
        left_layout.addLayout(slider_layout_3)

        # add tooltip
        self.horizontalSlider_smcm.setToolTip("Adjust head model position vertically.")
        self.label_smcm.setToolTip("Adjust head model position vertically.")
        self.label_text.setToolTip("Adjust head model position vertically.")
        self.horizontalSlider_movavg.setToolTip("Window numbers used to average to increase stability.")
        self.label_movavg.setToolTip("Window numbers used to average to increase stability.")
        self.label_text2.setToolTip("Window numbers used to average to increase stability.")
        left_layout.addStretch(1)

        ###### Radio button Layout ######
        radio_button_layout = QtWidgets.QHBoxLayout()
        self.radioButton_3p = QtWidgets.QRadioButton('3 point')
        self.radioButton_5p = QtWidgets.QRadioButton('5 point')
        self.radioButton_5p.setChecked(True)  # Default selection
        
        radio_button_layout.addWidget(self.radioButton_3p)
        radio_button_layout.addWidget(self.radioButton_5p)
        left_layout.addLayout(radio_button_layout)
        left_layout.addStretch(1)
        
        ###### Dropdown Layout ######
        dropdown_layout = QtWidgets.QVBoxLayout()
        # dropdown_layout.setAlignment(QtCore.Qt.AlignTop)
        self.dropdown = QtWidgets.QComboBox()
        self.dropdown.addItems(['Atlas (Colin27)',
                               'Atlas (Age 20-24)', 'Atlas (Age 25-29)',
                               'Atlas (Age 30-34)', 'Atlas (Age 35-39)',
                               'Atlas (Age 40-44)', 'Atlas (Age 45-49)',
                               'Atlas (Age 50-54)', 'Atlas (Age 55-59)',
                               'Atlas (Age 60-64)', 'Atlas (Age 65-69)',
                               'Atlas (Age 70-74)', 'Atlas (Age 75-79)',
                               'Atlas (Age 80-84)'])
        dropdown_layout.addWidget(self.dropdown)
        
        self.dropdown.setToolTip("Choose different altas model for providing head surface.")

        left_layout.addLayout(dropdown_layout)
        left_layout.addStretch(1)
        
        ###### Checkbox Layout (alternative method) ######
        checkbox_layout_head = QtWidgets.QHBoxLayout()
        checkbox_layout_head.addStretch(1)
        self.checkbox_backhead = QtWidgets.QCheckBox("Display posterior part optodes")
        checkbox_layout_head.addWidget(self.checkbox_backhead)
        checkbox_layout_head.addStretch(1)
        left_layout.addLayout(checkbox_layout_head)

        left_layout.addStretch(1)

        ###### Checkbox Layout (1020 point) ######
        checkbox_layout = QtWidgets.QVBoxLayout()
        checkbox_texts = {
                "Display 10-5 points": 'checkbox_105',
                "Display 10-10 points": 'checkbox_1010',
                "Display 10-20 points": 'checkbox_1020'
            }
        
        # Create checkboxes with specific names
        for text, var_name in checkbox_texts.items():
            hbox = QtWidgets.QHBoxLayout()
            hbox.addStretch(1)  # Adds a stretchable space that adjusts with the window
            setattr(self, var_name, QtWidgets.QCheckBox(text))  # Dynamically create and assign checkbox to attribute
            chk_box = getattr(self, var_name)
            if var_name == 'checkbox_1010':
                chk_box.setChecked(True)
            hbox.addWidget(chk_box)
            hbox.addStretch(1)  # You can adjust the ratio of stretches to control spacing
            checkbox_layout.addLayout(hbox)
        left_layout.addLayout(checkbox_layout)
        left_layout.addStretch(1)

        ###### Checkbox Layout (alternative method) ######
        checkbox_layout2 = QtWidgets.QHBoxLayout()
        checkbox_layout2.addStretch(1)
        self.checkbox_old = QtWidgets.QCheckBox("Use alternative method")
        checkbox_layout2.addWidget(self.checkbox_old)
        checkbox_layout2.addStretch(1)
        left_layout.addLayout(checkbox_layout2)

        left_layout.addStretch(6)

        self.main_layout.addLayout(left_layout, 1)
        # left_widget = QtWidgets.QWidget()
        # left_widget.setLayout(left_layout)
        # self.main_layout.addWidget(left_widget)

    def setup_right_layout(self):
        # # Right side layout (Image Display)
        right_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        right_layout.setAlignment(QtCore.Qt.AlignCenter)

        # Image Display
        self.label1 = QtWidgets.QLabel()
        self.label1.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.label1.setAlignment(QtCore.Qt.AlignCenter)  # Ensure the pixmap is centered
        # self.update_pixmap(800, 500)  # Initial pixmap size
        right_layout.addWidget(self.label1)
        
        self.main_layout.addLayout(right_layout, 3)
        # right_widget = QtWidgets.QWidget()
        # right_widget.setLayout(right_layout)
        # self.main_layout.addWidget(right_widget)

        # Create right layout for video display
        # self.label1 = QtWidgets.QLabel(self.centralwidget)
        # self.label1.setStyleSheet("background-color: white")
        # self.label1.setScaledContents(True)
    
        # # Add QSizeGrip to make QLabel resizable
        # size_grip = QtWidgets.QSizeGrip(self.label1)
        # size_grip.setStyleSheet("background-color: transparent")
        
        # right_layout = QtWidgets.QVBoxLayout()
        # right_layout.addWidget(self.label1)
        # right_layout.addWidget(size_grip, 0, QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight)
    
        # self.main_layout.addLayout(right_layout)

    def update_pixmap(self, width, height):
        # pixmap = QtGui.QPixmap(width, height)
        # pixmap.fill(QtGui.QColor('darkGray'))
        # self.label1.setPixmap(pixmap)

        pixmap = QtGui.QPixmap("GUI_icon/logo_tmp.png")  # Load an image file
        scaled_pixmap = pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.label1.setPixmap(scaled_pixmap)
        
    def resizeEvent(self, event):
        # This is where you dynamically adjust the pixmap size
        new_width = self.size().width() // 3 * 2  # Example: half of window width
        new_height = self.size().height() // 1.2  # Example: half of window height
        self.update_pixmap(new_width, new_height)
        super(MainWindow, self).resizeEvent(event)

        # main_layout.addLayout(left_layout, 1)  # The second parameter controls the proportion
        # main_layout.addLayout(right_layout, 3)  # The right side is given more space

        # # Set the layout to the central widget
        # self.centralwidget.setLayout(main_layout)
        # MainWindow.setCentralWidget(self.centralwidget)
        # self.statusbar = QtWidgets.QStatusBar(MainWindow)
        # MainWindow.setStatusBar(self.statusbar)

        # # Connect signals
        # # self.horizontalSlider_smcm.valueChanged.connect(self.scaletext_smcm)
        # # self.horizontalSlider_movavg.valueChanged.connect(self.scaletext_movavg)
        # self.button.clicked.connect(self.loadImage)
        # self.button_close.clicked.connect(self.closeVideo)

    # def scaletext_smcm(self, value):
    #     self.label_smcm.setText(str(value / 100))

    # def scaletext_movavg(self, value):
    #     self.label_movavg.setText(str(value))



    # def retranslateUi(self, MainWindow):
    #     _translate = QtCore.QCoreApplication.translate
    #     MainWindow.setWindowTitle(_translate("MainWindow", "LiveFaceMask"))
    #     self.label_smcm       .setText(_translate("MainWindow", str(0.53)))
    #     self.label_movavg     .setText(_translate("MainWindow", str(1)))
    #     self.label_opsize     .setText(_translate("MainWindow", str(1)))
    #     self.name_smcm        .setText(_translate("MainWindow", "model\npositions")) #------ to change -----#
    #     self.name_movavg      .setText(_translate("MainWindow", "moving\naverage"))
    #     self.button           .setText(_translate("MainWindow", "Turn on Video"))
    #     self.button_close     .setText(_translate("MainWindow", "Turn off Video"))
        
    def scaletext_x  (self, value):
        self.label_x   .setText(str(value/100))
    def scaletext_y  (self, value):
        self.label_y   .setText(str(value/100))
    def scaletext_z  (self, value):
        self.label_z   .setText(str(value/100))    
    def scaletext_smcm(self, value):
        self.label_smcm.setText(str(value/100))
    def scaletext_movavg(self, value):
        self.label_movavg.setText(str(value))
    def scaletext_opsize(self, value):
        self.label_opsize.setText(str(value))
        
        
    def closeVideo(self):
        self.videoStatus = False
        
    def selectionchange(self, i):
        # do something based on the selection
        if self.dropdown.itemText(i) == 'Atlas (Colin27)':
            atlas_file = '1020atlas/1020atlas_Colin27.json'
            atlas_file_5points = '1020atlas/1020atlas_Colin27_5points.json'
        
        if self.dropdown.itemText(i) == 'Atlas (Age 20-24)':
            atlas_file = '1020atlas/1020atlas_20-24Years.json'
            atlas_file_5points = '1020atlas/1020atlas_20-24Years_5points.json'
        if self.dropdown.itemText(i) == 'Atlas (Age 25-29)':
            atlas_file = '1020atlas/1020atlas_25-29Years.json'
            atlas_file_5points = '1020atlas/1020atlas_25-29Years_5points.json'
            
        if self.dropdown.itemText(i) == 'Atlas (Age 30-34)':
            atlas_file = '1020atlas/1020atlas_30-34Years.json'
            atlas_file_5points = '1020atlas/1020atlas_30-34Years_5points.json'
        if self.dropdown.itemText(i) == 'Atlas (Age 35-39)':
            atlas_file = '1020atlas/1020atlas_35-39Years.json'
            atlas_file_5points = '1020atlas/1020atlas_35-39Years_5points.json'
            
        if self.dropdown.itemText(i) == 'Atlas (Age 40-44)':
            atlas_file = '1020atlas/1020atlas_40-44Years.json'
            atlas_file_5points = '1020atlas/1020atlas_40-44Years_5points.json'
        if self.dropdown.itemText(i) == 'Atlas (Age 45-49)':
            atlas_file = '1020atlas/1020atlas_45-49Years.json'
            atlas_file_5points = '1020atlas/1020atlas_45-49Years_5points.json'
            
        if self.dropdown.itemText(i) == 'Atlas (Age 50-54)':
            atlas_file = '1020atlas/1020atlas_50-54Years.json'
            atlas_file_5points = '1020atlas/1020atlas_50-54Years_5points.json'
        if self.dropdown.itemText(i) == 'Atlas (Age 55-59)':
            atlas_file = '1020atlas/1020atlas_55-59Years.json'
            atlas_file_5points = '1020atlas/1020atlas_55-59Years_5points.json'
            
        if self.dropdown.itemText(i) == 'Atlas (Age 60-64)':
            atlas_file = '1020atlas/1020atlas_60-64Years.json'
            atlas_file_5points = '1020atlas/1020atlas_60-64Years_5points.json'
        if self.dropdown.itemText(i) == 'Atlas (Age 65-69)':
            atlas_file = '1020atlas/1020atlas_65-69Years.json'
            atlas_file_5points = '1020atlas/1020atlas_65-69Years_5points.json'
        
        if self.dropdown.itemText(i) == 'Atlas (Age 70-74)':
            atlas_file = '1020atlas/1020atlas_70-74Years.json'
            atlas_file_5points = '1020atlas/1020atlas_70-74Years_5points.json'
        if self.dropdown.itemText(i) == 'Atlas (Age 75-79)':
            atlas_file = '1020atlas/1020atlas_75-79Years.json'
            atlas_file_5points = '1020atlas/1020atlas_75-79Years_5points.json'
        
        if self.dropdown.itemText(i) == 'Atlas (Age 80-84)':
            atlas_file = '1020atlas/1020atlas_80-84Years.json'
            atlas_file_5points = '1020atlas/1020atlas_80-84Years_5points.json'
        atlas10_5=jd.load(atlas_file)
        atlas10_5_3points=jd.load(atlas_file)
        atlas10_5_5points=jd.load(atlas_file_5points)
#         print("Now selected", atlas_file)        
        
    def Brain_LMs_plotting(self, brain10_5p, results, image):
        #1010 system
        cm_1010 = brain10_5p["cm"][[0,2,4,6,8,10,12,14,16]]
        sm_1010 = brain10_5p["sm"][[0,2,4,6,8,10,12,14,16]]
        front_cm_1010 = cm_1010[:len(cm_1010)//2 + 1]
        back_cm_1010 = cm_1010[len(cm_1010)//2 + 1:]
        
        aal_1010 = brain10_5p["aal"][[1,3,5,7]]
        aar_1010 = brain10_5p["aar"][[1,3,5,7]]
        apl_1010 = brain10_5p["apl"][[1,3,5,7]]
        apr_1010 = brain10_5p["apr"][[1,3,5,7]]
        
        cal_2_1010 = brain10_5p["cal_2"][[1,3,5]]
        car_2_1010 = brain10_5p["car_2"][[1,3,5]]
        cal_4_1010 = brain10_5p["cal_4"][[1,3,5]]
        car_4_1010 = brain10_5p["car_4"][[1,3,5]]
        cal_6_1010 = brain10_5p["cal_6"][[3]]
        car_6_1010 = brain10_5p["car_6"][[3]]
        
        cpl_2_1010 = brain10_5p["cpl_2"][[1,3,5]]
        cpr_2_1010 = brain10_5p["cpr_2"][[1,3,5]]
        cpl_4_1010 = brain10_5p["cpl_4"][[1,3,5]]
        cpr_4_1010 = brain10_5p["cpr_4"][[1,3,5]]
        cpl_6_1010 = brain10_5p["cpl_6"][[3]]
        cpr_6_1010 = brain10_5p["cpr_6"][[3]]
        
        #1020 system
        cm_1020 = brain10_5p["cm"][[0,4,8,12,16]]
        sm_1020 = brain10_5p["sm"][[0,4,8,12,16]]
        front_cm_1020 = sm_1020[:len(cm_1020)//2 + 1] 
        back_cm_1020 = sm_1020[len(cm_1020)//2 + 1:] 
        
        aal_1020 = brain10_5p["aal"][[3,7]]
        aar_1020 = brain10_5p["aar"][[3,7]]
        apl_1020 = brain10_5p["apl"][[3,7]]
        apr_1020 = brain10_5p["apr"][[3,7]]

        opsize = self.horizontalSlider_opsize.value()
        opthick = opsize-1
    
    # 1. Draw face landmarks
#         mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
#                                  mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
#                                  mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
#                                  )   

        if(results.face_landmarks is not None):
    #             mp_drawing.draw_landmarks(image, landmark_face_predictor_for_all, None, 
    #                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=5),
    #                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=5)
    #                                  )             
    #             mp_drawing.draw_landmarks(image, numpy2landmark(predicted_cz), None, 
    #                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=5),
    #                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=5)
    #                                  )   
            # mp_drawing.draw_landmarks(image, nz, None, 
            #                      mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=3),
            #                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3)
            #                      )  
                # mp_drawing.draw_landmarks(image, numpy2landmark([iz]), None, 
                #                      mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=3),
                #                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3)
                #                      ) 
            # mp_drawing.draw_landmarks(image, numpy2landmark(np.vstack((lpa_to_plot, rpa_to_plot))), None, 
            #                      mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=3),
            #                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3)
            #                      ) 
            # -------105 system--------
            if self.checkbox_105.isChecked():
                mp_drawing.draw_landmarks(image, numpy2landmark(np.vstack((brain10_5p["aal"],brain10_5p["aar"], brain10_5p["sm"], brain10_5p["cm"][0:9]))), None, 
                                     mp_drawing.DrawingSpec(color=(255,255,0), thickness=opthick, circle_radius=opsize),
                                     mp_drawing.DrawingSpec(color=(0,255,0), thickness=opthick, circle_radius=opsize)
                                     ) 
                mp_drawing.draw_landmarks(image, numpy2landmark(np.vstack((brain10_5p["cal_1"], brain10_5p["car_1"], brain10_5p["cal_2"], brain10_5p["car_2"], brain10_5p["cal_3"], brain10_5p["car_3"], brain10_5p["cal_4"], brain10_5p["car_4"],\
                                                                          brain10_5p["cal_5"], brain10_5p["car_5"], brain10_5p["cal_6"], brain10_5p["car_6"], brain10_5p["cal_7"], brain10_5p["car_7"]))), None, 
                         mp_drawing.DrawingSpec(color=(255,255,0), thickness=opthick, circle_radius=opsize),
                         mp_drawing.DrawingSpec(color=(0,255,0), thickness=opthick, circle_radius=opsize)
                         ) 
                if self.checkbox_backhead.isChecked():
                ## back of the head ##
                    mp_drawing.draw_landmarks(image, numpy2landmark(np.vstack((brain10_5p["apl"],brain10_5p["apr"], brain10_5p["cm"][9:]))), None, 
                                     mp_drawing.DrawingSpec(color=(255,255,0), thickness=opthick, circle_radius=opsize),
                                     mp_drawing.DrawingSpec(color=(0,255,0), thickness=opthick, circle_radius=opsize)
                                     ) 
                    mp_drawing.draw_landmarks(image, numpy2landmark(np.vstack((brain10_5p["cpl_1"], brain10_5p["cpr_1"], brain10_5p["cpl_2"], brain10_5p["cpr_2"], brain10_5p["cpl_3"], brain10_5p["cpr_3"], brain10_5p["cpl_4"], brain10_5p["cpr_4"],\
                                                                              brain10_5p["cpl_5"], brain10_5p["cpr_5"], brain10_5p["cpl_6"], brain10_5p["cpr_6"], brain10_5p["cpl_7"], brain10_5p["cpr_7"]))), None, 
                             mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=3),
                             mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3)
                             ) 
    
            # -------1010 system--------
            if self.checkbox_1010.isChecked():
                mp_drawing.draw_landmarks(image, numpy2landmark(np.vstack((sm_1010, front_cm_1010, aal_1010, aar_1010))), None, 
                                     mp_drawing.DrawingSpec(color=(0,255,0,60), thickness=opthick, circle_radius=opsize),
                                     mp_drawing.DrawingSpec(color=(0,255,0,60), thickness=opthick, circle_radius=opsize)
                                     ) 
                mp_drawing.draw_landmarks(image, numpy2landmark(np.vstack((cal_2_1010, car_2_1010, cal_4_1010, car_4_1010, cal_6_1010, car_6_1010))), None, 
                                     mp_drawing.DrawingSpec(color=(0,255,0,60), thickness=opthick, circle_radius=opsize),
                                     mp_drawing.DrawingSpec(color=(0,255,0,60), thickness=opthick, circle_radius=opsize)
                                     ) 
                if self.checkbox_backhead.isChecked():
                ## back of the head ##
                    mp_drawing.draw_landmarks(image, numpy2landmark(np.vstack((back_cm_1010, apl_1010, apr_1010))), None, 
                                     mp_drawing.DrawingSpec(color=(0,255,0,60), thickness=opthick, circle_radius=opsize),
                                     mp_drawing.DrawingSpec(color=(0,255,0,60), thickness=opthick, circle_radius=opsize)
                                     ) 
                    mp_drawing.draw_landmarks(image, numpy2landmark(np.vstack((cpl_2_1010, cpr_2_1010, cpl_4_1010, cpr_4_1010, cpl_6_1010, cpr_6_1010))), None, 
                                         mp_drawing.DrawingSpec(color=(0,255,0,60), thickness=2, circle_radius=3),
                                         mp_drawing.DrawingSpec(color=(0,255,0,60), thickness=2, circle_radius=3)
                                         ) 
                    
                    
            # -------1020 system--------
            if self.checkbox_1020.isChecked():
                mp_drawing.draw_landmarks(image, numpy2landmark(np.vstack((sm_1020, front_cm_1020, aal_1020, aar_1020))), None, 
                                     mp_drawing.DrawingSpec(color=(0,125,255,128), thickness=opthick, circle_radius=opsize),
                                     mp_drawing.DrawingSpec(color=(0,255,0,128), thickness=opthick, circle_radius=opsize)
                                     ) 
                if self.checkbox_backhead.isChecked():
                ## back of the head ##
                    mp_drawing.draw_landmarks(image, numpy2landmark(np.vstack((apl_1020, apr_1020))), None, 
                                         mp_drawing.DrawingSpec(color=(0,125,255,128), thickness=2, circle_radius=3),
                                         mp_drawing.DrawingSpec(color=(0,255,0,128), thickness=2, circle_radius=3)
                                         ) 
                
                    
    #             mp_drawing.draw_landmarks(image, numpy2landmark(np.vstack((cpl_1, cpr_1, cpl_2, cpr_2, cpl_3, cpr_3, cpl_4, cpr_4,\
    #                                                                       cpl_5, cpr_5, cpl_6, cpr_6, cpl_7, cpr_7))), None, 
    #                      mp_drawing.DrawingSpec(color=(126,37,3), thickness=1, circle_radius=3),
    #                      mp_drawing.DrawingSpec(color=(126,37,3), thickness=1, circle_radius=3)
    #                      )  
    
    def livefacemask(self, image, results, moving_averaged_Brain_LMs_list, iteration):
        if(results.face_landmarks is not None):
            if self.radioButton_3p.isChecked():
                atlas10_5 = copy.deepcopy(atlas10_5_3points)
            else:
                atlas10_5 = copy.deepcopy(atlas10_5_5points)
            # -----------------------------------------------------
            
            # -------------- Linear Matrix ---------------
            pts=results.face_landmarks.landmark
            landmark_face_predictor_for_all = landmark_pb2.NormalizedLandmarkList(
                landmark=[
                    pts[33],pts[133],pts[168],pts[362],pts[263],pts[4],pts[61],pts[291],pts[10],pts[332],pts[389],pts[323],pts[397]\
                ,pts[378],pts[152],pts[149],pts[172],pts[93],pts[162],pts[103]])
            predictor_for_all = landmark2numpy(landmark_face_predictor_for_all)
            predictor_for_all = np.reshape(predictor_for_all, (1, -1))

            nz = landmark_pb2.NormalizedLandmarkList(landmark=[pts[168]])
            predicted_lpa = np.transpose(my_reg1020(predictor_for_all, x_lpa))
            predicted_rpa = np.transpose(my_reg1020(predictor_for_all, x_rpa))
            predicted_iz = np.transpose(my_reg1020(predictor_for_all, x_iz))
            predicted_cz = np.transpose(my_reg1020(predictor_for_all, x_cz))
            
            Amat, bvec = affinemap(np.array([atlas10_5_3points['nz'], atlas10_5_3points['lpa'], atlas10_5_3points['rpa'], atlas10_5_3points['iz'], atlas10_5_3points['cz']]),
                                   np.array([landmark2numpy(nz)[0], predicted_lpa[0], predicted_rpa[0], predicted_iz[0], predicted_cz[0]]))


#             print(Amat)
            lpa_to_plot = predicted_lpa
            rpa_to_plot = predicted_rpa
            # -------------- Old Method ---------------
            if self.checkbox_old.isChecked():
                pts=results.face_landmarks.landmark
                landmark_face_subset = landmark_pb2.NormalizedLandmarkList(
                    landmark=[
                        pts[168],
                        pts[10],
                        {'x':2*pts[234].x-pts[227].x, 'y':2*pts[234].y-pts[227].y, 'z':2*pts[234].z-pts[227].z},
                        {'x':2*pts[454].x-pts[447].x, 'y':2*pts[454].y-pts[447].y, 'z':2*pts[454].z-pts[447].z}
                    ])

                Amat, bvec = affinemap(np.array([atlas10_5['nz'], atlas10_5['sm'][1], atlas10_5['rpa'], atlas10_5['lpa']]),
                                       landmark2numpy(landmark_face_subset))
                lpa_to_plot = landmark2numpy(landmark_face_subset)[2,:]
                rpa_to_plot = landmark2numpy(landmark_face_subset)[3,:]
            
            # -----------------------------------------------------
                
            tmp = reg1020(Amat, bvec, [atlas10_5['lpa'], atlas10_5['rpa'], atlas10_5['iz']])
            iz = tmp[2]

            aal = reg1020(Amat, bvec, atlas10_5['aal'])
            aar = reg1020(Amat, bvec, atlas10_5['aar'])
            sm = reg1020(Amat, bvec, atlas10_5['sm'])
            cm = reg1020(Amat, bvec, atlas10_5['cm'])
            
            apl = reg1020(Amat, bvec, atlas10_5['apl'])
            apr = reg1020(Amat, bvec, atlas10_5['apr'])
            
            cal_1 = reg1020(Amat, bvec, atlas10_5['cal_1'])
            car_1 = reg1020(Amat, bvec, atlas10_5['car_1'])
            cal_2 = reg1020(Amat, bvec, atlas10_5['cal_2'])
            car_2 = reg1020(Amat, bvec, atlas10_5['car_2'])
            cal_3 = reg1020(Amat, bvec, atlas10_5['cal_3'])
            car_3 = reg1020(Amat, bvec, atlas10_5['car_3'])
            cal_4 = reg1020(Amat, bvec, atlas10_5['cal_4'])
            car_4 = reg1020(Amat, bvec, atlas10_5['car_4'])
            cal_5 = reg1020(Amat, bvec, atlas10_5['cal_5'])
            car_5 = reg1020(Amat, bvec, atlas10_5['car_5'])
            cal_6 = reg1020(Amat, bvec, atlas10_5['cal_6'])
            car_6 = reg1020(Amat, bvec, atlas10_5['car_6'])
            cal_7 = reg1020(Amat, bvec, atlas10_5['cal_7'])
            car_7 = reg1020(Amat, bvec, atlas10_5['car_7'])
            
            cpl_1 = reg1020(Amat, bvec, atlas10_5['cpl_1'])
            cpr_1 = reg1020(Amat, bvec, atlas10_5['cpr_1'])
            cpl_2 = reg1020(Amat, bvec, atlas10_5['cpl_2'])
            cpr_2 = reg1020(Amat, bvec, atlas10_5['cpr_2'])
            cpl_3 = reg1020(Amat, bvec, atlas10_5['cpl_3'])
            cpr_3 = reg1020(Amat, bvec, atlas10_5['cpr_3'])
            cpl_4 = reg1020(Amat, bvec, atlas10_5['cpl_4'])
            cpr_4 = reg1020(Amat, bvec, atlas10_5['cpr_4'])
            cpl_5 = reg1020(Amat, bvec, atlas10_5['cpl_5'])
            cpr_5 = reg1020(Amat, bvec, atlas10_5['cpr_5'])
            cpl_6 = reg1020(Amat, bvec, atlas10_5['cpl_6'])
            cpr_6 = reg1020(Amat, bvec, atlas10_5['cpr_6'])
            cpl_7 = reg1020(Amat, bvec, atlas10_5['cpl_7'])
            cpr_7 = reg1020(Amat, bvec, atlas10_5['cpr_7'])
                        
            def move_by_slider(arr):
                arr_out = copy.deepcopy(arr)
                for i in range(0, len(arr)):
                    arr_out[i][1] = arr[i][1]+(self.horizontalSlider_smcm.value()-50)/100
                return arr_out
            aal = move_by_slider(aal)
            aar = move_by_slider(aar)
            apl = move_by_slider(apl)
            apr = move_by_slider(apr)
            cm = move_by_slider(cm)
            sm = move_by_slider(sm)
            
            cal_1 = move_by_slider(cal_1)
            car_1 = move_by_slider(car_1)
            cal_2 = move_by_slider(cal_2)
            car_2 = move_by_slider(car_2)
            cal_3 = move_by_slider(cal_3)
            car_3 = move_by_slider(car_3)
            cal_4 = move_by_slider(cal_4)
            car_4 = move_by_slider(car_4)
            cal_5 = move_by_slider(cal_5)
            car_5 = move_by_slider(car_5)
            cal_6 = move_by_slider(cal_6)
            car_6 = move_by_slider(car_6)
            cal_7 = move_by_slider(cal_7)
            car_7 = move_by_slider(car_7)
            
            cpl_1 = move_by_slider(cpl_1)
            cpr_1 = move_by_slider(cpr_1)
            cpl_2 = move_by_slider(cpl_2)
            cpr_2 = move_by_slider(cpr_2)
            cpl_3 = move_by_slider(cpl_3)
            cpr_3 = move_by_slider(cpr_3)
            cpl_4 = move_by_slider(cpl_4)
            cpr_4 = move_by_slider(cpr_4)
            cpl_5 = move_by_slider(cpl_5)
            cpr_5 = move_by_slider(cpr_5)
            cpl_6 = move_by_slider(cpl_6)
            cpr_6 = move_by_slider(cpr_6)
            cpl_7 = move_by_slider(cpl_7)
            cpr_7 = move_by_slider(cpr_7)


            # ------- save 10-5 points for plotting -----
            brain10_5p = {}
            brain10_5p['aal'] = aal
            brain10_5p['aar'] = aar
            brain10_5p['apl'] = apl
            brain10_5p['apr'] = apr
            brain10_5p['cm'] = cm
            brain10_5p['sm'] = sm
            brain10_5p['cal_1'] = cal_1
            brain10_5p['car_1'] = car_1
            brain10_5p['cal_2'] = cal_2
            brain10_5p['car_2'] = car_2
            brain10_5p['cal_3'] = cal_3
            brain10_5p['car_3'] = car_3
            brain10_5p['cal_4'] = cal_4
            brain10_5p['car_4'] = car_4
            brain10_5p['cal_5'] = cal_5
            brain10_5p['car_5'] = car_5
            brain10_5p['cal_6'] = cal_6
            brain10_5p['car_6'] = car_6
            brain10_5p['cal_7'] = cal_7
            brain10_5p['car_7'] = car_7

            brain10_5p['cpl_1'] = cpl_1
            brain10_5p['cpr_1'] = cpr_1
            brain10_5p['cpl_2'] = cpl_2
            brain10_5p['cpr_2'] = cpr_2
            brain10_5p['cpl_3'] = cpl_3
            brain10_5p['cpr_3'] = cpr_3
            brain10_5p['cpl_4'] = cpl_4
            brain10_5p['cpr_4'] = cpr_4
            brain10_5p['cpl_5'] = cpl_5
            brain10_5p['cpr_5'] = cpr_5
            brain10_5p['cpl_6'] = cpl_6
            brain10_5p['cpr_6'] = cpr_6
            brain10_5p['cpl_7'] = cpl_7
            brain10_5p['cpr_7'] = cpr_7

            # ------- END -----
            brain10_5p_array = list(brain10_5p.values())
            brain10_5p_array = np.concatenate(brain10_5p_array, axis=0)
            
            # moving_averaged_Brain_LMs_list.append(brain10_5p_array)

            # moving_window_size = self.horizontalSlider_movavg.value()
            # if iteration >= moving_window_size:
            #     moving_averaged_Brain_LMs_list.pop(0)
            #     moving_averaged_Brain_LMs_arr = np.stack(moving_averaged_Brain_LMs_list, axis=0)
            #     Avged_moving_averaged_Brain_LMs_arr = np.mean(moving_averaged_Brain_LMs_arr, axis=0)

            #     Avged_moving_averaged_Brain_LMs_dict = map_arrays_to_dict(Avged_moving_averaged_Brain_LMs_arr)
            #     self.Brain_LMs_plotting(Avged_moving_averaged_Brain_LMs_dict, results, image)


            moving_averaged_Brain_LMs_list.append(brain10_5p_array)
            moving_window_size = self.horizontalSlider_movavg.value()
        
            # # Adjust the data list to match the current window size
            # if len(moving_averaged_Brain_LMs_list) > moving_window_size:
            #     moving_averaged_Brain_LMs_list.pop(0)
        
            # # Check if there's enough data to compute the moving average
            # if len(moving_averaged_Brain_LMs_list) == moving_window_size:
            #     moving_averaged_Brain_LMs_arr = np.stack(moving_averaged_Brain_LMs_list, axis=0)
            #     Avged_moving_averaged_Brain_LMs_arr = np.mean(moving_averaged_Brain_LMs_arr, axis=0)
        
            #     Avged_moving_averaged_Brain_LMs_dict = map_arrays_to_dict(Avged_moving_averaged_Brain_LMs_arr)
            #     self.Brain_LMs_plotting(Avged_moving_averaged_Brain_LMs_dict, results, image)

            if len(moving_averaged_Brain_LMs_list) > moving_window_size:
                    # Trim the list to the current window size by removing the oldest entries
                    moving_averaged_Brain_LMs_list = moving_averaged_Brain_LMs_list[-moving_window_size:]
            
            # Check if there's enough data to compute the moving average
            if len(moving_averaged_Brain_LMs_list) == moving_window_size:
                moving_averaged_Brain_LMs_arr = np.stack(moving_averaged_Brain_LMs_list, axis=0)
                Avged_moving_averaged_Brain_LMs_arr = np.mean(moving_averaged_Brain_LMs_arr, axis=0)
        
                Avged_moving_averaged_Brain_LMs_dict = map_arrays_to_dict(Avged_moving_averaged_Brain_LMs_arr)
                self.Brain_LMs_plotting(Avged_moving_averaged_Brain_LMs_dict, results, image)

        return moving_averaged_Brain_LMs_list

    def loadImage(self):
        self.videoStatus = True
    ####### LOAD IMAGD ########     
#         image = cv2.imread("subject52_2_forBIOESympo_2to200.jpg")
#         with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True) as holistic:
#             results = holistic.process(image)
#             self.livefacemask(image, results)
#             FlippedImage = cv2.flip(image, 1)
#             self.label1.setPixmap(convert_cv_qt(FlippedImage))
        
    ####### VIDEO ########
        vid = cv2.VideoCapture(0)
        iteration = 0
        moving_averaged_Brain_LMs_list = []
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True) as holistic:     
            while(vid.isOpened()):
                QtWidgets.QApplication.processEvents()
                ret, frame = vid.read()
                # desired_size = (640*2, ) # Enter your desired width and height as a tuple
                # frame = cv2.resize(frame, desired_size)

                # Resize frame to current size
                frame = self.resize_frame_keep_aspect(frame, self.label1.width(), self.label1.height())
                
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                image.flags.writeable = False    
                # Make Detections
                results = holistic.process(image)
                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                moving_averaged_Brain_LMs_list = self.livefacemask(image, results, moving_averaged_Brain_LMs_list, iteration)
                iteration += 1
                
#                 cv2.imshow('Raw Webcam Feed', image)
                FlippedImage = cv2.flip(image, 1)
                self.label1.setPixmap(convert_cv_qt(FlippedImage))
        
                if self.videoStatus == False:
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    def resize_frame_keep_aspect(self, frame, label_width, label_height):
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Calculate the aspect ratio of the frame
        aspect_ratio = width / height
        
        # Compute new dimensions based on aspect ratio
        if label_width / label_height > aspect_ratio:
            new_height = label_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = label_width
            new_height = int(new_width / aspect_ratio)
        
        # Ensure the new dimensions are even numbers (common requirement for some video processing tasks)
        new_width += new_width % 2
        new_height += new_height % 2
        
        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        return resized_frame
    def plot(self):
            # random data
            import random
            data = [random.random() for i in range(10)]

            # clearing old figure
            self.figure.clear()

            # create an axis
            ax = self.figure.add_subplot(111)

            # plot data
            ax.plot(data, '*-')

            # refresh canvas
#             self.canvas.draw()
            self.label1.setPixmap()

if __name__ == "__main__":
    app = QtWidgets.QApplication.instance()  # Check if an instance already exists
    if not app:  # Create a new instance if not
        app = QtWidgets.QApplication(sys.argv)

    mainWin = MainWindow()
    mainWin.show()
    app.exec_()
        