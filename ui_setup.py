import pyqtgraph as pg
from PySide6.QtWidgets import (QVBoxLayout, QWidget, QPushButton, QLabel,
                             QHBoxLayout, QDoubleSpinBox, QSpacerItem,
                             QSizePolicy, QFrame, QSlider, QCheckBox, QMainWindow)
from PySide6.QtCore import Qt, QSize

# --- Constants ---
SLIDER_PRECISION = 1000 # For mapping float values to integer slider

class Ui_OnsetDetectorApp(object):
    def setupUi(self, MainWindow: QMainWindow):
        """Creates and arranges UI elements onto the MainWindow."""
        MainWindow.setObjectName("OnsetDetectorApp")
        MainWindow.resize(1400, 800) # Use resize instead of setGeometry for simplicity here

        central_widget = QWidget(MainWindow)
        MainWindow.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setObjectName("main_layout")

        # --- Left Panel: Plot Area ---
        plot_container = QWidget()
        plot_container.setObjectName("plot_container")
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setObjectName("plot_layout")

        MainWindow.plot_widget = pg.PlotWidget(name="WaveformPlot")
        MainWindow.plot_widget.setObjectName("plot_widget")
        MainWindow.plot_widget.setLabel('left', '幅度')
        MainWindow.plot_widget.setLabel('bottom', '时间 (秒)')
        MainWindow.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        # --- 修改：设置鼠标交互模式 ---
        vb = MainWindow.plot_widget.getViewBox()
        vb.setMouseMode(pg.ViewBox.PanMode)
        vb.setMouseEnabled(x=True, y=False)
        vb.menu = None # 禁用右键菜单

        MainWindow.waveform_item = MainWindow.plot_widget.plot(pen=pg.mkPen('b', width=1), name="Waveform")
        MainWindow.onset_scatter_item = pg.ScatterPlotItem(
            pen=None, symbol='o', size=8, brush=pg.mkBrush(255, 0, 0, 200), name="Onsets"
        )
        MainWindow.plot_widget.addItem(MainWindow.onset_scatter_item)
        MainWindow.plot_widget.addLegend(offset=(-10, 10))

        # Add InfiniteLine for playback position visualization
        MainWindow.playback_position_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('g', width=1, style=Qt.DashLine))
        MainWindow.plot_widget.addItem(MainWindow.playback_position_line, ignoreBounds=True)
        MainWindow.playback_position_line.hide() # Initially hidden

        plot_layout.addWidget(MainWindow.plot_widget)
        main_layout.addWidget(plot_container, 3)

        # --- Right Panel: Controls ---
        controls_container = QWidget()
        controls_container.setObjectName("controls_container")
        controls_container.setFixedWidth(300)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setObjectName("controls_layout")
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # File Selection
        MainWindow.file_button = QPushButton("选择音频文件")
        MainWindow.file_button.setObjectName("file_button")
        MainWindow.file_label = QLabel("尚未选择文件")
        MainWindow.file_label.setObjectName("file_label")
        MainWindow.file_label.setWordWrap(True)
        controls_layout.addWidget(MainWindow.file_button)
        controls_layout.addWidget(MainWindow.file_label)
        controls_layout.addSpacing(10)

        # Separator
        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.HLine); line1.setFrameShadow(QFrame.Shadow.Sunken)
        controls_layout.addWidget(line1)
        controls_layout.addSpacing(10)

        # Onset Detection Controls
        onset_label = QLabel("Onset 检测参数:")
        onset_label.setObjectName("onset_label")
        onset_label.setStyleSheet("font-weight: bold;")
        controls_layout.addWidget(onset_label)

        # Delta (Sensitivity) Control - Slider + SpinBox
        delta_label = QLabel("Delta (灵敏度):")
        delta_label.setObjectName("delta_label")
        controls_layout.addWidget(delta_label)
        delta_input_layout = QHBoxLayout()
        delta_input_layout.setObjectName("delta_input_layout")
        MainWindow.delta_slider = QSlider(Qt.Orientation.Horizontal)
        MainWindow.delta_slider.setObjectName("delta_slider")
        MainWindow.delta_slider.setRange(1, SLIDER_PRECISION)
        MainWindow.delta_slider.setValue(int(0.05 * SLIDER_PRECISION))
        MainWindow.delta_spinbox = QDoubleSpinBox()
        MainWindow.delta_spinbox.setObjectName("delta_spinbox")
        MainWindow.delta_spinbox.setRange(0.001, 1.0)
        MainWindow.delta_spinbox.setSingleStep(0.01)
        MainWindow.delta_spinbox.setValue(0.03)
        MainWindow.delta_spinbox.setDecimals(3)
        MainWindow.delta_spinbox.setFixedWidth(80)
        delta_input_layout.addWidget(MainWindow.delta_slider)
        delta_input_layout.addWidget(MainWindow.delta_spinbox)
        controls_layout.addLayout(delta_input_layout)
        delta_tip = QLabel("较小值更灵敏")
        delta_tip.setObjectName("delta_tip")
        delta_tip.setStyleSheet("font-size: 9pt; color: gray;")
        controls_layout.addWidget(delta_tip)

        MainWindow.detect_button = QPushButton("检测 Onsets")
        MainWindow.detect_button.setObjectName("detect_button")
        MainWindow.detect_button.setEnabled(False)
        controls_layout.addWidget(MainWindow.detect_button)
        MainWindow.status_label = QLabel("")
        MainWindow.status_label.setObjectName("status_label")
        controls_layout.addWidget(MainWindow.status_label)
        controls_layout.addSpacing(10)

        # Separator
        line2 = QFrame(); line2.setFrameShape(QFrame.Shape.HLine); line2.setFrameShadow(QFrame.Shadow.Sunken)
        controls_layout.addWidget(line2)
        controls_layout.addSpacing(10)

        # View Duration Control
        view_label = QLabel("视图窗口大小 (秒):")
        view_label.setObjectName("view_label")
        view_label.setStyleSheet("font-weight: bold;")
        controls_layout.addWidget(view_label)
        view_input_layout = QHBoxLayout()
        view_input_layout.setObjectName("view_input_layout")
        MainWindow.view_duration_slider = QSlider(Qt.Orientation.Horizontal)
        MainWindow.view_duration_slider.setObjectName("view_duration_slider")
        MainWindow.view_duration_slider.setRange(1 * SLIDER_PRECISION, 60 * SLIDER_PRECISION)
        MainWindow.view_duration_slider.setValue(int(10 * SLIDER_PRECISION))
        MainWindow.view_duration_spinbox = QDoubleSpinBox()
        MainWindow.view_duration_spinbox.setObjectName("view_duration_spinbox")
        MainWindow.view_duration_spinbox.setRange(1.0, 60.0)
        MainWindow.view_duration_spinbox.setSingleStep(0.5)
        MainWindow.view_duration_spinbox.setValue(10.0)
        MainWindow.view_duration_spinbox.setDecimals(1)
        MainWindow.view_duration_spinbox.setFixedWidth(70)
        view_input_layout.addWidget(MainWindow.view_duration_slider)
        view_input_layout.addWidget(MainWindow.view_duration_spinbox)
        controls_layout.addLayout(view_input_layout)
        controls_layout.addSpacing(10)

        # Separator
        line3 = QFrame(); line3.setFrameShape(QFrame.Shape.HLine); line3.setFrameShadow(QFrame.Shadow.Sunken)
        controls_layout.addWidget(line3)
        controls_layout.addSpacing(10)

         # Audio Playback Controls
        play_label = QLabel("音频播放:")
        play_label.setObjectName("play_label")
        play_label.setStyleSheet("font-weight: bold;")
        controls_layout.addWidget(play_label)

        play_buttons_layout = QHBoxLayout()
        play_buttons_layout.setObjectName("play_buttons_layout")
        MainWindow.play_pause_button = QPushButton("播放")
        MainWindow.play_pause_button.setObjectName("play_pause_button")
        MainWindow.play_pause_button.setEnabled(False)
        play_buttons_layout.addWidget(MainWindow.play_pause_button)
        controls_layout.addLayout(play_buttons_layout)

        MainWindow.play_main_checkbox = QCheckBox("播放主音乐")
        MainWindow.play_main_checkbox.setObjectName("play_main_checkbox")
        MainWindow.play_main_checkbox.setChecked(True)
        MainWindow.play_main_checkbox.setEnabled(False)
        MainWindow.play_clicks_checkbox = QCheckBox("播放打点声音")
        MainWindow.play_clicks_checkbox.setObjectName("play_clicks_checkbox")
        MainWindow.play_clicks_checkbox.setChecked(True)
        MainWindow.play_clicks_checkbox.setEnabled(False)
        controls_layout.addWidget(MainWindow.play_main_checkbox)
        controls_layout.addWidget(MainWindow.play_clicks_checkbox)

        # --- ADDED: Follow Playback Checkbox ---
        MainWindow.follow_playback_checkbox = QCheckBox("跟随播放")
        MainWindow.follow_playback_checkbox.setObjectName("follow_playback_checkbox")
        MainWindow.follow_playback_checkbox.setChecked(True) # Default to on
        MainWindow.follow_playback_checkbox.setEnabled(False) # Initially disabled, enable when playback is possible
        controls_layout.addWidget(MainWindow.follow_playback_checkbox)
        # --- END ADDED ---

        # Add Seek Slider and Time Label
        seek_layout = QHBoxLayout()
        seek_layout.setObjectName("seek_layout")
        MainWindow.seek_slider = QSlider(Qt.Orientation.Horizontal)
        MainWindow.seek_slider.setObjectName("seek_slider")
        MainWindow.seek_slider.setRange(0, 1000) # Placeholder range
        MainWindow.seek_slider.setEnabled(False)
        MainWindow.time_label = QLabel("00:00 / 00:00")
        MainWindow.time_label.setObjectName("time_label")
        MainWindow.time_label.setFixedWidth(100)
        MainWindow.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        seek_layout.addWidget(MainWindow.seek_slider)
        seek_layout.addWidget(MainWindow.time_label)
        controls_layout.addLayout(seek_layout)

        controls_layout.addSpacing(10)

        # Separator
        line4 = QFrame(); line4.setFrameShape(QFrame.Shape.HLine); line4.setFrameShadow(QFrame.Shadow.Sunken)
        controls_layout.addWidget(line4)
        controls_layout.addSpacing(10)

        # Export Button
        MainWindow.export_button = QPushButton("导出 Onsets...")
        MainWindow.export_button.setObjectName("export_button")
        MainWindow.export_button.setEnabled(False)
        controls_layout.addWidget(MainWindow.export_button)


        # Add Spacer to push controls up
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        controls_layout.addItem(spacer)

        main_layout.addWidget(controls_container, 1)

        # --- Status Bar ---
        # Status bar is part of QMainWindow, set message from the main class
        # MainWindow.statusBar().showMessage("准备就绪")

        self.retranslateUi(MainWindow) # Add call to retranslateUi

        # Connections are handled in the main class after calling setupUi


    def retranslateUi(self, MainWindow):
        """Sets the text for UI elements (can be used for localization)."""
        _translate = MainWindow.setWindowTitle # Simple alias for demonstration
        _translate("音频 Onset 检测与可视化工具")
        # Add translations for other widgets if needed, e.g.
        # MainWindow.file_button.setText(_translate("选择音频文件"))
        # MainWindow.status_label.setText(_translate(""))
        # ... and so on for labels, button texts, etc.
        # Keeping it simple for now, main window title is often sufficient initially. 