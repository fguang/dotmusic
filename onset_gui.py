import sys
import numpy as np
import librosa
import pyqtgraph as pg
import tempfile # For temporary audio file
import soundfile # For writing temporary audio file
import os # Import os for file deletion

# --- 从 ui_setup.py 导入 UI 类 ---
from ui_setup import Ui_OnsetDetectorApp
# --- 导入新的 AudioHandler ---
from audio_handler import AudioHandler # ADDED import

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QPushButton, QFileDialog, QHBoxLayout, QMessageBox,
                             QLabel, QDoubleSpinBox, QSpacerItem, QSizePolicy, QFrame,
                             QSlider, QCheckBox, QMenu)
from PySide6.QtCore import Qt, Slot, QTimer, QUrl # ADDED QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput # ADDED QtMultimedia imports

# --- Constants ---
# SLIDER_PRECISION is now defined in ui_setup.py, can be removed if not used elsewhere directly
# SLIDER_PRECISION = 1000 # For mapping float values to integer slider

# --- pyqtgraph Configuration ---
pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# --- Main Application Window ---
class OnsetDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # self.setWindowTitle("音频 Onset 检测与可视化工具") # Handled by setupUi
        # self.setGeometry(100, 100, 1400, 800) # Handled by setupUi (resize)

        # --- 数据存储 ---
        # REMOVED audio data attributes, managed by AudioHandler now
        # self.audio_file_path = None
        # self.y = None # Audio waveform (keep as float32 for processing)
        # self.y_int16 = None # Audio waveform as int16 for WAV playback
        # self.sr = None # Sampling rate
        # self.time_axis = None # Time axis for plotting

        # --- 创建 AudioHandler 实例 ---
        self.audio_handler = AudioHandler()

        # --- Onset 和 Click 数据 ---
        self.onset_times = np.array([]) # Store detected onset times
        self.clicks_audio = None # Generated click track audio data (float32)
        self.clicks_audio_int16 = None # Click track as int16 for WAV playback

        # --- 音频播放状态 ---
        # playback_position_line is created in setupUi
        # 主音频播放器
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.temp_audio_file_path = None # Store path to temporary main audio file
        # 打点音轨播放器
        self.click_player = QMediaPlayer()
        self.click_audio_output = QAudioOutput()
        self.click_player.setAudioOutput(self.click_audio_output)
        self.temp_click_file_path = None # Store path to temporary click audio file

        self.is_seeking = False # Flag to prevent feedback loop during slider seek
        self.was_playing_before_seek = False # Remember state before seeking

        # --- 设置 UI ---
        Ui_OnsetDetectorApp().setupUi(self) # 调用从 ui_setup.py 导入的 UI 设置方法
        # self._init_ui() # Remove call to the old method

        # --- 连接信号 ---
        self._connect_signals() # Connect signals and slots

        # Initialize mute state based on checkboxes AFTER UI setup
        self._toggle_main_mute(self.play_main_checkbox.isChecked())
        self._toggle_click_mute(self.play_clicks_checkbox.isChecked())

        # --- 初始状态栏消息 ---
        self.statusBar().showMessage("准备就绪") # Set initial status bar message after setup


    def _connect_signals(self):
        """Connects UI element signals to slots."""
        self.file_button.clicked.connect(self.open_file_dialog)
        self.detect_button.clicked.connect(self.run_onset_detection)

        # Delta synchronization
        self.delta_slider.valueChanged.connect(self._sync_delta_slider_to_spinbox)
        self.delta_spinbox.valueChanged.connect(self._sync_delta_spinbox_to_slider)
        # Optional: Trigger detection on value change (can be slow)
        # self.delta_spinbox.editingFinished.connect(self.run_onset_detection) # Detect when editing finished

        # View Duration synchronization and update
        self.view_duration_slider.valueChanged.connect(self._sync_view_duration_slider_to_spinbox)
        self.view_duration_spinbox.valueChanged.connect(self._sync_view_duration_spinbox_to_slider)
        self.view_duration_spinbox.valueChanged.connect(self._update_plot_x_range) # Update view when value changes

        # Playback controls
        self.play_pause_button.clicked.connect(self._handle_play_pause)
        # Connect checkboxes to mute functions
        self.play_main_checkbox.toggled.connect(self._toggle_main_mute)
        self.play_clicks_checkbox.toggled.connect(self._toggle_click_mute)

        # Media Player signals (Main Player drives UI updates)
        self.media_player.positionChanged.connect(self._update_position)
        self.media_player.durationChanged.connect(self._update_duration)
        self.media_player.playbackStateChanged.connect(self._update_playback_state)
        self.media_player.errorOccurred.connect(self._handle_media_error)
        # Add error handler for click player too
        self.click_player.errorOccurred.connect(self._handle_click_media_error)

        # Seek Slider signals
        self.seek_slider.sliderMoved.connect(self._set_position_display) # Only update display while moving
        self.seek_slider.sliderPressed.connect(self._slider_pressed)
        self.seek_slider.sliderReleased.connect(self._slider_released)

        # Export Button
        self.export_button.clicked.connect(self.show_export_menu)


    # --- Slot Implementations ---

    @Slot()
    def open_file_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择音频文件", ".", "Audio Files (*.wav *.mp3 *.aac *.flac)")
        if filename:
            self._reset_playback() # 清理之前的播放状态
            self.file_label.setText(f"加载中:\n{filename}") # Show filename immediately
            self.statusBar().showMessage("正在加载音频...")
            self._reset_state_for_new_file() # 清理旧数据（现在会调用 audio_handler.reset）
            QApplication.processEvents()
            # 使用 QTimer 异步调用新的加载方法
            QTimer.singleShot(50, lambda: self.load_audio_via_handler(filename))

    def _cleanup_temp_file(self, file_path_attr):
        """Safely cleans up a temporary file path stored in an attribute."""
        file_path = getattr(self, file_path_attr, None)
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Successfully removed temporary file: {file_path}")
            except OSError as e:
                print(f"Error removing temporary file {file_path}: {e}", file=sys.stderr)
            except Exception as e:
                 print(f"Unexpected error removing temporary file {file_path}: {e}", file=sys.stderr)
        setattr(self, file_path_attr, None) # Clear the attribute
        self.play_pause_button.setEnabled(False)
        self.seek_slider.setEnabled(False)
        self.seek_slider.setValue(0)
        self.time_label.setText("00:00 / 00:00")
        if self.playback_position_line:
            self.playback_position_line.hide()

    def _reset_playback(self):
        """Stops media players and cleans up temporary resources."""
        print("Resetting playback...")
        if self.media_player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
            self.media_player.stop()
            self.media_player.setSource(QUrl()) # Clear source immediately
        if self.click_player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
            self.click_player.stop()
            self.click_player.setSource(QUrl()) # Clear source immediately

        # Use the cleanup helper function
        self._cleanup_temp_file('temp_audio_file_path')
        self._cleanup_temp_file('temp_click_file_path')

        # Reset UI elements related to playback
        self.play_pause_button.setText("播放")
        self.play_pause_button.setEnabled(False)
        self.seek_slider.setEnabled(False)
        self.seek_slider.setValue(0)
        self.time_label.setText("00:00 / 00:00")
        if self.playback_position_line:
            self.playback_position_line.hide()

    def _reset_state_for_new_file(self):
         """Clears data related to the previous file."""
         print("Resetting state for new file...")
         self._reset_playback() # 首先停止播放并清理临时文件
         # --- 调用 AudioHandler 的重置方法 ---
         self.audio_handler.reset()
         # --- 清理 Onset 和 Click 数据 ---
         # REMOVED direct reset of audio data: self.y = None; self.y_int16 = None; self.sr = None; self.time_axis = None
         self.onset_times = np.array([]); self.clicks_audio = None; self.clicks_audio_int16 = None
         # --- 重置 UI 元素 ---
         self.waveform_item.setData([], [])
         self.onset_scatter_item.setData([], [])
         self.detect_button.setEnabled(False)
         self.play_main_checkbox.setEnabled(False); self.play_main_checkbox.setChecked(True)
         self.play_clicks_checkbox.setEnabled(False); self.play_clicks_checkbox.setChecked(True)
         self._toggle_main_mute(True) # 确保初始未静音
         self._toggle_click_mute(True) # 确保初始未静音
         self.export_button.setEnabled(False)
         self.status_label.setText("")
         self.file_label.setText("尚未选择文件") # 重置文件标签
         self.statusBar().showMessage("准备就绪")


    # RENAMED/REPLACED load_audio with load_audio_via_handler
    def load_audio_via_handler(self, file_path):
        """Loads audio using AudioHandler and updates UI accordingly."""
        if not file_path: return

        load_successful = self.audio_handler.load_audio(file_path)

        if load_successful:
            # 使用 audio_handler 获取文件路径
            loaded_path = self.audio_handler.get_file_path()
            print(f"音频加载成功: {loaded_path}")
            self.file_label.setText(f"已加载:\n{loaded_path}")
            self.statusBar().showMessage("音频加载完成，准备检测 Onsets")

            # 启用控件
            self.detect_button.setEnabled(True)
            self.play_main_checkbox.setEnabled(True) # 启用主音轨切换

            # 更新波形图（现在从 handler 获取数据）
            self.update_waveform_plot()
            # 准备主音轨源（现在从 handler 获取数据）
            self._prepare_main_audio_source()

            # 加载后自动开始检测
            QTimer.singleShot(10, self.run_onset_detection)

        else:
            error_msg = f"加载音频时出错: {file_path}"
            print(error_msg, file=sys.stderr)
            self.file_label.setText(f"加载失败:\n{file_path}")
            self.statusBar().showMessage("音频加载失败")
            # 显示错误消息给用户
            QMessageBox.critical(self, "错误", f"无法加载音频文件:\n{file_path}")
            # 确保状态完全重置
            self._reset_state_for_new_file()


    def update_waveform_plot(self):
        # 从 AudioHandler 获取波形数据
        time_axis, y = self.audio_handler.get_waveform_data()
        # 检查数据是否存在且有效
        if y is not None and time_axis is not None and len(time_axis) > 0:
            print("正在更新波形图...")
            self.waveform_item.setData(time_axis, y)
            # 自动调整 Y 轴范围，然后禁用它以允许用户缩放
            self.plot_widget.enableAutoRange(axis=pg.ViewBox.YAxis)
            self.plot_widget.autoRange()
            # 使用 QTimer 延迟禁用自动范围，确保在渲染后执行
            QTimer.singleShot(0, lambda: self.plot_widget.disableAutoRange(axis=pg.ViewBox.YAxis))
            print("波形图更新完毕，Y 轴范围已固定。")
            # 更新 X 轴范围以匹配视图持续时间设置
            self._update_plot_x_range()
        else:
            print("没有音频数据可供绘制。")
            self.waveform_item.setData([], [])
            self.onset_scatter_item.setData([], [])


    @Slot()
    def run_onset_detection(self):
        # 检查 AudioHandler 是否已加载音频
        if not self.audio_handler.is_loaded():
            self.status_label.setText("请先加载音频文件")
            return

        # ... (UI updates: disable buttons, set status) ...
        current_delta = self.delta_spinbox.value()
        self.statusBar().showMessage(f"正在检测 Onsets (delta={current_delta:.3f})...")
        self.status_label.setText("检测中...")
        self.detect_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.play_clicks_checkbox.setEnabled(False)
        self.play_main_checkbox.setEnabled(False)
        # Temporarily disable playback controls during detection
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.click_player.pause()
        self.play_pause_button.setEnabled(False)
        self.seek_slider.setEnabled(False)

        QApplication.processEvents()
        QTimer.singleShot(10, lambda: self.detect_onsets_thread(current_delta))

    def detect_onsets_thread(self, delta_value):
        try:
            print(f"开始检测 Onsets, delta={delta_value}")
            # 从 AudioHandler 获取检测所需的数据 (float32 y, sr)
            y_detect, sr_detect = self.audio_handler.get_detection_data()
            if y_detect is None or sr_detect is None:
                 print("错误：无法获取用于检测的音频数据。")
                 raise ValueError("音频数据尚未加载") # 引发错误以进入 except 块

            # 使用从 handler 获取的数据进行检测
            self.onset_times = librosa.onset.onset_detect(
                y=y_detect, sr=sr_detect, units='time', backtrack=True, delta=delta_value
            )
            print(f"检测完成，找到 {len(self.onset_times)} 个 Onsets。")
            self.status_label.setText(f"检测到 {len(self.onset_times)} 个 Onsets")
            self.statusBar().showMessage("Onset 检测完成")
            # 更新标记（现在从 handler 获取 y 用于缩放）
            self.update_onset_markers()
            # 生成打点音轨（现在从 handler 获取 sr 和 y 的长度）
            self.generate_click_track()
            # 准备打点音轨源（现在从 handler 获取 sr）
            self._prepare_click_audio_source()

            # Re-enable controls after processing
            self.play_clicks_checkbox.setEnabled(True)
            self.play_main_checkbox.setEnabled(True)
            self.export_button.setEnabled(True)
             # Only re-enable playback if duration is valid
            if self.media_player.duration() > 0:
                 self.play_pause_button.setEnabled(True)
                 self.seek_slider.setEnabled(True)

        except Exception as e:
             error_msg = f"Onset 检测失败: {e}"
             print(error_msg, file=sys.stderr)
             self.status_label.setText("检测失败")
             self.statusBar().showMessage("Onset 检测失败")
             QMessageBox.warning(self, "检测错误", f"Onset 检测过程中发生错误:\n{e}")
             self.onset_times = np.array([])
             self.clicks_audio = None; self.clicks_audio_int16 = None
             self.onset_scatter_item.setData([], [])
             # Ensure click source is cleared if generation failed
             self.click_player.setSource(QUrl())
             self._cleanup_temp_file('temp_click_file_path')
             self.play_clicks_checkbox.setEnabled(False)
        finally:
             self.detect_button.setEnabled(True) # Always re-enable detection button


    def update_onset_markers(self):
        # 从 handler 获取 float32 的 y 用于定位标记
        _, y_float = self.audio_handler.get_waveform_data()

        if y_float is None:
             self.onset_scatter_item.setData([], [])
             return

        if self.onset_times.size > 0:
            # 使用 float 波形的最大绝对值来缩放标记位置
            y_max = np.max(np.abs(y_float)) if len(y_float) > 0 else 1.0
            marker_y_position = y_max * 1.05 # 定位在波形上方一点
            scatter_data = [{'pos': (t, marker_y_position), 'data': t} for t in self.onset_times]
            self.onset_scatter_item.setData(scatter_data)
            print(f"更新了 {len(self.onset_times)} 个 Onset 标记。")
        else:
            self.onset_scatter_item.setData([], [])
            print("没有 Onset 标记可更新。")

    def generate_click_track(self):
        """Generates float32 click track and converts to int16."""
        # 从 AudioHandler 获取参考数据
        y_ref, sr_ref = self.audio_handler.get_detection_data() # float32 y
        y_int16_ref, _ = self.audio_handler.get_playback_data() # int16 y

        if y_ref is None or sr_ref is None or y_int16_ref is None: # 需要参考音频长度和采样率
            print("无法生成打点音轨：缺少参考音频数据。")
            self.clicks_audio = None
            self.clicks_audio_int16 = None
            return

        print("正在生成打点声音...")
        click_freq = 1500
        click_duration_sec = 0.03
        click_frames = int(click_duration_sec * sr_ref) # 使用 handler 的 sr
        click_times_rel = np.arange(click_frames) / sr_ref # 使用 handler 的 sr
        # 生成 [-0.6, 0.6] 范围内的 float32 点击信号
        click_signal = (0.6 * np.sin(2 * np.pi * click_freq * click_times_rel)).astype(np.float32)

        # 使用 handler 的 y 创建 float32 零数组
        self.clicks_audio = np.zeros_like(y_ref, dtype=np.float32)

        if self.onset_times.size > 0:
            # 使用 handler 的 sr 转换时间到采样点
            onset_samples = librosa.time_to_samples(self.onset_times, sr=sr_ref)

            for sample_idx in onset_samples:
                start = sample_idx
                end = min(sample_idx + click_frames, len(self.clicks_audio))
                len_to_add = end - start
                if start < len(self.clicks_audio) and len_to_add > 0:
                     signal_part = click_signal[:len_to_add]
                     # Add click signal to the float32 array
                     self.clicks_audio[start:end] += signal_part
            # Optional: Normalize float clicks track if needed (max should be ~0.6 here)
            # max_click_abs = np.max(np.abs(self.clicks_audio))
            # if max_click_abs > 1.0:
            #     self.clicks_audio /= max_click_abs
            #     print("Normalized click track.")

        # 转换最终的 float32 clicks_audio 为 int16 用于播放
        print("Converting click audio to int16 for WAV playback.")
        self.clicks_audio_int16 = (self.clicks_audio * 32767).astype(np.int16)
        # 确保长度与 int16 参考音频一致 (可能由于舍入误差)
        if len(self.clicks_audio_int16) != len(y_int16_ref):
             target_len = len(y_int16_ref)
             current_len = len(self.clicks_audio_int16)
             print(f"Warning: Generated int16 click track length ({current_len}) differs from reference ({target_len}). Adjusting.")
             if target_len > current_len:
                 self.clicks_audio_int16 = np.pad(self.clicks_audio_int16, (0, target_len - current_len))
             elif target_len < current_len:
                 self.clicks_audio_int16 = self.clicks_audio_int16[:target_len]


        print("打点声音生成完毕 (float32 and int16)。")


    # --- Slider/SpinBox Sync and View Update Slots ---
    @Slot(int)
    def _sync_delta_slider_to_spinbox(self, value):
        val = value / self.delta_slider.maximum() # Use maximum for scaling
        if abs(self.delta_spinbox.value() - val) > 0.0001: # Prevent recursive loop
            self.delta_spinbox.setValue(val * (self.delta_spinbox.maximum() - self.delta_spinbox.minimum()) + self.delta_spinbox.minimum()) # Scale to spinbox range

    @Slot(float)
    def _sync_delta_spinbox_to_slider(self, value):
        # Scale value from spinbox range (0.01-1.0) to slider range (1-1000)
        slider_max = self.delta_slider.maximum()
        spin_min = self.delta_spinbox.minimum()
        spin_max = self.delta_spinbox.maximum()
        val = int(((value - spin_min) / (spin_max - spin_min)) * (slider_max - self.delta_slider.minimum()) + self.delta_slider.minimum()) 
        if self.delta_slider.value() != val: # Prevent recursive loop
            self.delta_slider.setValue(val)

    @Slot(int)
    def _sync_view_duration_slider_to_spinbox(self, value):
        # Scale value from slider range (1000-60000) to spinbox range (1.0-60.0)
        slider_min = self.view_duration_slider.minimum()
        slider_max = self.view_duration_slider.maximum()
        spin_min = self.view_duration_spinbox.minimum()
        spin_max = self.view_duration_spinbox.maximum()
        val = (((value - slider_min) / (slider_max - slider_min)) * (spin_max - spin_min) + spin_min)
        if abs(self.view_duration_spinbox.value() - val) > 0.001:
            self.view_duration_spinbox.setValue(val)

    @Slot(float)
    def _sync_view_duration_spinbox_to_slider(self, value):
        # Scale value from spinbox range (1.0-60.0) to slider range (1000-60000)
        slider_min = self.view_duration_slider.minimum()
        slider_max = self.view_duration_slider.maximum()
        spin_min = self.view_duration_spinbox.minimum()
        spin_max = self.view_duration_spinbox.maximum()
        val = int(((value - spin_min) / (spin_max - spin_min)) * (slider_max - slider_min) + slider_min)
        if self.view_duration_slider.value() != val:
            self.view_duration_slider.setValue(val)

    @Slot()
    def _update_plot_x_range(self):
        # 从 handler 获取时间轴
        time_axis, _ = self.audio_handler.get_waveform_data()
        if time_axis is None or len(time_axis) == 0: return # 检查长度
        # ... (rest of range calculation) ...
        view_duration = self.view_duration_spinbox.value()
        current_range = self.plot_widget.getViewBox().viewRange()[0]
        current_center = (current_range[0] + current_range[1]) / 2
        view_start = max(0, current_center - view_duration / 2)
        # 确保 view_end 不超过实际时间轴最大值
        max_time = time_axis[-1] # 已经检查过 len > 0
        view_end = min(max_time, view_start + view_duration)
        view_start = max(0, view_end - view_duration) # 根据可能被裁剪的末端调整开始

        print(f"Updating X range: [{view_start:.2f}, {view_end:.2f}]")
        self.plot_widget.setXRange(view_start, view_end, padding=0)

    # --- Audio Playback Slots ---
    @Slot()
    def _handle_play_pause(self):
        """Toggles playback state for both players."""
        # Check if main source is valid (click source might not be ready yet)
        if not self.media_player.source().isValid() or self.media_player.duration() <= 0:
             print("Play/Pause: Main source not valid or duration is zero.")
             # Ensure button state reflects reality if called erroneously
             self.play_pause_button.setText("播放")
             self.playback_position_line.hide()
             return

        main_state = self.media_player.playbackState()

        if main_state == QMediaPlayer.PlaybackState.PlayingState:
            print("Pausing players...")
            self.media_player.pause()
            if self.click_player.source().isValid(): # Only pause click if valid
                self.click_player.pause()
        else: # Paused or Stopped state
            print("Playing players...")
            pos = self.seek_slider.value()

            # Ensure players are at the correct position BEFORE playing
            self.media_player.setPosition(pos)
            if self.click_player.source().isValid():
                self.click_player.setPosition(pos) # Sync click player position

            # Use QTimer to ensure position is set before play starts
            QTimer.singleShot(10, self._delayed_play)

    def _delayed_play(self):
        """Starts playback after a short delay."""
        print("Executing delayed play...")
        self.media_player.play()
        if self.click_player.source().isValid():
            self.click_player.play()

    @Slot(int)
    def _update_position(self, position):
        """Updates the seek slider and time label based on main player position."""
        if not self.is_seeking: # Only update slider if user isn't dragging it
             # Prevent setting value beyond max during natural playback end
             max_val = self.seek_slider.maximum()
             if position <= max_val:
                  self.seek_slider.setValue(position)
             else: # Handle reaching the end
                  self.seek_slider.setValue(max_val)

        duration = self.media_player.duration()
        # Ensure duration is positive before formatting time
        valid_duration = duration if duration > 0 else 0
        self.time_label.setText(f"{self._format_time(position)} / {self._format_time(valid_duration)}")

        if self.playback_position_line:
            self.playback_position_line.setPos(position / 1000.0)


    @Slot(int)
    def _update_duration(self, duration):
        """Sets the range of the seek slider and updates the time label based on the main player."""
        if duration > 0:
            print(f"Duration updated: {duration} ms")
            self.seek_slider.setRange(0, duration)
            self.seek_slider.setEnabled(True)
            # Only enable play if detection isn't running
            if self.detect_button.isEnabled(): # Check if detection finished
                 self.play_pause_button.setEnabled(True)
            # Update time label immediately
            current_pos = self.media_player.position()
            self.time_label.setText(f"{self._format_time(current_pos)} / {self._format_time(duration)}")
        else:
            print("Duration invalid or zero.")
            self.seek_slider.setRange(0, 1) # Avoid division by zero etc.
            self.seek_slider.setEnabled(False)
            self.play_pause_button.setEnabled(False)
            self.time_label.setText("00:00 / 00:00")

    @Slot(QMediaPlayer.PlaybackState)
    def _update_playback_state(self, state):
        """Updates the UI based on the main player's state."""
        print(f"Main player state changed to: {state}")
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_pause_button.setText("暂停")
            self.playback_position_line.show()
        elif state == QMediaPlayer.PlaybackState.PausedState:
            self.play_pause_button.setText("播放")
            self.playback_position_line.show() # Keep line visible on pause
        elif state == QMediaPlayer.PlaybackState.StoppedState:
            self.play_pause_button.setText("播放")
            self.playback_position_line.hide()
            # Explicitly stop the click player when main player stops.
            if self.click_player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
                print("Main stopped, ensuring click player stops too.")
                self.click_player.stop()
            # Reset slider to beginning ONLY if playback finished naturally at the end?
            # current_pos = self.media_player.position() # This might be 0 already
            # duration = self.media_player.duration()
            # if duration > 0 and current_pos >= duration - 50: # Allow small tolerance
            #      print("Playback stopped at end, resetting slider to 0")
            #      self.seek_slider.setValue(0)
            # Let's keep the slider at the end for now, user can click play to restart from 0.


    @Slot(int)
    def _set_position_display(self, position):
        """Updates the time label and line position while the slider is being moved."""
        duration = self.media_player.duration()
        valid_duration = duration if duration > 0 else 0
        self.time_label.setText(f"{self._format_time(position)} / {self._format_time(valid_duration)}")
        if self.playback_position_line:
             self.playback_position_line.setPos(position / 1000.0)


    @Slot()
    def _slider_pressed(self):
        """Flags that the user is interacting with the slider and pauses players."""
        self.is_seeking = True
        # Remember if it was playing before pausing
        self.was_playing_before_seek = (self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState)
        if self.was_playing_before_seek:
            print("Slider pressed, pausing players for seek.")
            self.media_player.pause()
            if self.click_player.source().isValid():
                self.click_player.pause()


    @Slot()
    def _slider_released(self):
        """Sets the player positions when the user releases the slider."""
        if not self.is_seeking: return # Prevent accidental calls
        self.is_seeking = False
        new_position = self.seek_slider.value()
        print(f"Slider released, setting position for both players to: {new_position}")

        # Set position for both players
        self.media_player.setPosition(new_position)
        if self.click_player.source().isValid():
            self.click_player.setPosition(new_position)

        # Update the visual line immediately after setting position
        if self.playback_position_line:
             self.playback_position_line.setPos(new_position / 1000.0)

        # Optional: Resume playback if it was playing before seek started
        if self.was_playing_before_seek:
             print("Resuming playback after seek.")
             # Use timer to ensure position is set before playing again
             QTimer.singleShot(10, self._delayed_play)
        self.was_playing_before_seek = False # Reset flag


    # --- Mute Control Slots ---
    @Slot(bool)
    def _toggle_main_mute(self, checked):
        """Mutes/unmutes the main audio output."""
        is_muted = not checked
        self.audio_output.setMuted(is_muted)
        print(f"Main Audio Muted: {is_muted}")

    @Slot(bool)
    def _toggle_click_mute(self, checked):
        """Mutes/unmutes the click audio output."""
        is_muted = not checked
        self.click_audio_output.setMuted(is_muted)
        print(f"Click Audio Muted: {is_muted}")


    @Slot(QMediaPlayer.Error, str)
    def _handle_media_error(self, error, error_string):
         """Handles errors reported by the main QMediaPlayer."""
         # Check if error is meaningful (not Abort etc.)
         if error != QMediaPlayer.Error.NoError and error != QMediaPlayer.Error.ResourceError: # ResourceError happens on stop/source change
            print(f"Main Media Player Error ({error}): {error_string}", file=sys.stderr)
            QMessageBox.warning(self, "播放错误 (主轨道)", f"播放主音轨时发生错误:\n{error_string}")
            self._reset_playback() # Reset playback UI on significant error

    @Slot(QMediaPlayer.Error, str)
    def _handle_click_media_error(self, error, error_string):
         """Handles errors reported by the click QMediaPlayer."""
         if error != QMediaPlayer.Error.NoError and error != QMediaPlayer.Error.ResourceError:
            print(f"Click Media Player Error ({error}): {error_string}", file=sys.stderr)
            QMessageBox.warning(self, "播放错误 (打点轨道)", f"播放打点音轨时发生错误:\n{error_string}")
            # Don't reset everything, maybe just stop/clear the click player?
            self.click_player.stop()
            self.click_player.setSource(QUrl())
            self._cleanup_temp_file('temp_click_file_path')
            self.play_clicks_checkbox.setEnabled(False) # Disable click track if it errors


    # --- Export Functionality --- 
    @Slot()
    def show_export_menu(self):
        """Shows a context menu for export options."""
        if self.onset_times.size == 0:
            QMessageBox.information(self, "无数据", "尚未检测到 Onset 数据可供导出。")
            return

        menu = QMenu(self)
        export_visible_action = menu.addAction("导出当前视图 Onsets (.txt)")
        export_all_action = menu.addAction("导出全部 Onsets (.txt)")
        export_visible_csv_action = menu.addAction("导出当前视图 Onsets (.csv)")
        export_all_csv_action = menu.addAction("导出全部 Onsets (.csv)")


        action = menu.exec(self.export_button.mapToGlobal(self.export_button.rect().bottomLeft()))

        if action == export_visible_action:
            self._export_onsets_logic(visible_only=True, format='txt')
        elif action == export_all_action:
            self._export_onsets_logic(visible_only=False, format='txt')
        elif action == export_visible_csv_action:
            self._export_onsets_logic(visible_only=True, format='csv')
        elif action == export_all_csv_action:
             self._export_onsets_logic(visible_only=False, format='csv')


    def _export_onsets_logic(self, visible_only=False, format='txt'):
        """Handles the logic for exporting onset times."""
        if self.onset_times.size == 0:
             print("No onsets to export.")
             return

        onsets_to_export = self.onset_times
        default_suffix = f".{format}"
        file_filter = f"文本文件 (*.{format})" if format == 'txt' else f"CSV 文件 (*.{format})"
        default_filename = "onsets_export"

        if visible_only:
            if self.time_axis is None or len(self.time_axis) == 0:
                 QMessageBox.information(self, "无视图", "无法确定当前视图范围。")
                 return
            current_x_range = self.plot_widget.getViewBox().viewRange()[0]
            view_start, view_end = current_x_range
            onsets_to_export = self.onset_times[
                (self.onset_times >= view_start) & (self.onset_times <= view_end)
            ]
            default_filename = f"onsets_view_{view_start:.1f}s_to_{view_end:.1f}s"
            print(f"Exporting {len(onsets_to_export)} visible onsets in range [{view_start:.2f}, {view_end:.2f}]")
            if onsets_to_export.size == 0:
                 QMessageBox.information(self, "无可见数据", "当前视图范围内没有检测到的 Onset 点。")
                 return
        else:
             print(f"Exporting all {len(onsets_to_export)} onsets.")
             default_filename = "onsets_all"

        # Add format suffix
        default_filename += default_suffix

        # Get save path
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存 Onset 时间", default_filename, file_filter
        )

        if save_path:
            try:
                print(f"Saving onsets to {save_path} (Format: {format})")
                np.savetxt(save_path, onsets_to_export, fmt='%.6f', delimiter=',' if format=='csv' else '\n')
                self.statusBar().showMessage(f"Onset 数据已导出到 {save_path}")
                print("保存成功。")
            except Exception as e:
                error_msg = f"导出 Onset 数据时出错: {e}"
                print(error_msg, file=sys.stderr)
                QMessageBox.critical(self, "导出错误", f"无法保存文件:\n{e}")
                self.statusBar().showMessage("导出失败")

    # --- Overrides ---
    def closeEvent(self, event):
         """Ensure audio stops and temp files are cleaned when closing."""
         print("Closing application, stopping playback and cleaning up...")
         self._reset_playback() # Use the reset function to stop and clean
         event.accept() # Proceed with closing

    def _format_time(self, ms):
        """Formats milliseconds to MM:SS string."""
        if ms < 0: ms = 0 # Ensure non-negative
        total_seconds = round(ms / 1000) # Round to nearest second
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    # --- Audio Source Preparation ---

    def _prepare_audio_source(self, player, audio_data_int16, sample_rate, temp_file_path_attr, description):
        """Generic function to prepare and set audio source for a player."""
        # sample_rate is now required
        if audio_data_int16 is None or sample_rate is None:
            print(f"Cannot prepare {description} source: Missing data or sample rate.")
            player.setSource(QUrl()) # 清除源（如果数据缺失）
            self._cleanup_temp_file(temp_file_path_attr) # 清理旧文件
            return False

        print(f"Preparing {description} audio source (SR={sample_rate})...")

        # Stop player before changing source
        player.stop()
        # Clean up previous temp file for this player
        self._cleanup_temp_file(temp_file_path_attr)

        temp_file = None
        try:
            # Create a new temporary WAV file using delete=False
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file_path = temp_file.name
            setattr(self, temp_file_path_attr, temp_file_path) # Store the path

            # Write int16 data using soundfile, providing sample_rate
            soundfile.write(temp_file_path, audio_data_int16, sample_rate, format='WAV', subtype='PCM_16')
            temp_file.close() # Close the handle, file persists because delete=False
            print(f"Temporary {description} audio written to: {temp_file_path}")

            # Set the new source for the media player
            player.setSource(QUrl.fromLocalFile(temp_file_path))
            print(f"{description} source set successfully.")
            return True

        except Exception as e:
            print(f"Error preparing {description} audio source: {e}", file=sys.stderr)
            QMessageBox.critical(self, "音频错误", f"准备{description}音轨源时出错:\n{e}")
            # Clean up the file if creation failed mid-way
            if temp_file:
                 temp_file.close() # Close handle first
            self._cleanup_temp_file(temp_file_path_attr) # Attempt removal
            player.setSource(QUrl()) # Ensure source is cleared
            return False


    def _prepare_main_audio_source(self):
        """Prepares the main audio track using the generic function."""
        # 从 AudioHandler 获取播放数据 (int16 y, sr)
        y_play, sr_play = self.audio_handler.get_playback_data()
        success = self._prepare_audio_source(self.media_player,
                                           y_play, # Pass int16 data
                                           sr_play, # Pass sample rate
                                           'temp_audio_file_path',
                                           "main")
        if success:
            # Reset position visually after setting source, duration comes later via signal
            self.seek_slider.setValue(0)
            self.time_label.setText(f"00:00 / {self._format_time(0)}")
        else:
             self._reset_playback() # Full reset if main source fails


    def _prepare_click_audio_source(self):
        """Prepares the click track audio using the generic function."""
        # 获取用于参考长度和采样率的 handler 数据
        y_int16_ref, sr_ref = self.audio_handler.get_playback_data()

        # 检查 click 音轨 int16 数据是否存在且与参考长度匹配
        if self.clicks_audio_int16 is None:
             if y_int16_ref is not None:
                 print("Warning: Click track int16 data missing, creating zeros.")
                 self.clicks_audio_int16 = np.zeros_like(y_int16_ref, dtype=np.int16)
             else:
                 print("Error: Cannot prepare click source without reference audio.")
                 self.play_clicks_checkbox.setEnabled(False)
                 self.play_clicks_checkbox.setChecked(False)
                 return # 不能继续

        # 再次检查长度 (可能在 generate_click_track 中已修正, 但再次检查更安全)
        if y_int16_ref is not None and len(self.clicks_audio_int16) != len(y_int16_ref):
             print("Warning: Click track int16 length mismatch before prepare, padding/truncating.")
             target_len = len(y_int16_ref)
             current_len = len(self.clicks_audio_int16)
             if target_len > current_len:
                self.clicks_audio_int16 = np.pad(self.clicks_audio_int16, (0, target_len - current_len))
             elif target_len < current_len:
                self.clicks_audio_int16 = self.clicks_audio_int16[:target_len]

        # 使用 handler 的 sr
        success = self._prepare_audio_source(self.click_player,
                                           self.clicks_audio_int16, # Pass int16 click data
                                           sr_ref, # Pass sample rate from handler
                                           'temp_click_file_path',
                                           "click")
        if not success:
             # Don't do a full reset, just disable the click track UI
             print("Disabling click track due to preparation error.")
             self.play_clicks_checkbox.setEnabled(False)
             self.play_clicks_checkbox.setChecked(False) # Uncheck it


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OnsetDetectorApp()
    window.show()
    sys.exit(app.exec()) 
