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
from PySide6.QtCore import Qt, Slot, QTimer, QUrl, QObject, Signal, QThread # ADDED QThread, Signal, QObject
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput # ADDED QtMultimedia imports

# --- Constants ---
# SLIDER_PRECISION is now defined in ui_setup.py, can be removed if not used elsewhere directly
# SLIDER_PRECISION = 1000 # For mapping float values to integer slider

# --- pyqtgraph Configuration ---
pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# --- Audio Loading Worker ---
class AudioLoaderWorker(QObject):
    # Signal emitted when loading finishes: sends file path, loaded y data, and sr
    finished = Signal(str, object, int)
    # Signal emitted on error: sends file path and the exception object
    error = Signal(str, object)

    def __init__(self, audio_handler, file_path):
        super().__init__()
        self.audio_handler = audio_handler # Use the main app's handler instance
        self.file_path = file_path
        self.is_cancelled = False

    @Slot()
    def run(self):
        """Loads the audio file using the audio handler."""
        try:
            print(f"Worker thread started for: {self.file_path}")
            # Call the modified load_audio which now returns data or raises error
            y, sr = self.audio_handler.load_audio(self.file_path)
            # Check if cancelled during loading
            if self.is_cancelled:
                 print(f"Worker thread cancelled for: {self.file_path}")
                 self.error.emit(self.file_path, "加载被取消") # Emit specific error or just return
                 return
            # Emit success signal with the loaded data
            self.finished.emit(self.file_path, y, sr)
            print(f"Worker thread finished successfully for: {self.file_path}")
        except Exception as e:
            print(f"Worker thread encountered error for {self.file_path}: {e}", file=sys.stderr)
            if not self.is_cancelled: # Don't emit error if cancelled
                 self.error.emit(self.file_path, e) # Emit error signal with exception

    def cancel(self):
        print(f"Cancelling worker thread for: {self.file_path}")
        self.is_cancelled = True


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
        # REMOVED: self.clicks_audio_int16 = None # Click track as int16 for WAV playback (will be generated on demand)

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

        # --- 添加 QThread 相关属性 ---
        self.load_thread = None
        self.load_worker = None

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

        # --- 新增：尝试预初始化音频后端 ---
        self._try_preload_audio_backend()
        # --- 结束新增 ---

        # self.follow_playback_checkbox = QCheckBox("Follow Playback") # REMOVED - Now created in setupUi
        # ... 然后将这个 checkbox 添加到你的布局中 ...

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
            # --- 取消正在进行的加载 (如果存在) ---
            if self.load_thread is not None and self.load_thread.isRunning():
                 print("检测到正在进行的加载，尝试取消...")
                 if self.load_worker:
                     self.load_worker.cancel()
                 # 等待线程优雅退出 (可选，但推荐)
                 # self.load_thread.quit()
                 # self.load_thread.wait(500) # 等待最多 500ms
                 # 或者强制终止 (不推荐，可能导致资源问题)
                 # self.load_thread.terminate()
                 # self.load_thread.wait()
                 print("旧加载线程已取消/终止。")
                 # 重置线程和 worker 引用
                 self.load_thread = None
                 self.load_worker = None

            self._reset_playback() # 清理之前的播放状态
            self._reset_state_for_new_file() # 清理旧数据和 UI 状态

            self.file_label.setText(f"开始加载: {filename}") # Show filename immediately
            self.statusBar().showMessage(f"正在后台加载音频: {os.path.basename(filename)}...")
            self.detect_button.setEnabled(False) # 禁用检测直到加载完成
            self.play_pause_button.setEnabled(False) # 禁用播放
            self.seek_slider.setEnabled(False)
            self.play_main_checkbox.setEnabled(False)
            self.play_clicks_checkbox.setEnabled(False)
            self.export_button.setEnabled(False)
            self.follow_playback_checkbox.setEnabled(False)
            QApplication.processEvents() # 确保 UI 更新

            # --- 启动后台线程加载 ---
            self.load_thread = QThread(self)
            self.load_worker = AudioLoaderWorker(self.audio_handler, filename)
            self.load_worker.moveToThread(self.load_thread)

            # 连接信号槽
            self.load_worker.finished.connect(self._audio_load_finished)
            self.load_worker.error.connect(self._audio_load_error)
            self.load_thread.started.connect(self.load_worker.run)
            # 清理: 线程结束后删除 worker 和 thread 对象
            self.load_thread.finished.connect(self.load_worker.deleteLater)
            self.load_thread.finished.connect(self.load_thread.deleteLater)
            # 重置引用，以便下次检查
            self.load_thread.finished.connect(self._reset_thread_refs)

            print("启动加载线程...")
            self.load_thread.start()

    # --- 新增：槽函数，用于重置线程引用 ---
    @Slot()
    def _reset_thread_refs(self):
        print("加载线程已完成，重置引用。")
        self.load_thread = None
        self.load_worker = None


    # --- 新增：处理音频加载完成的槽函数 ---
    @Slot(str, object, int)
    def _audio_load_finished(self, file_path, y_data, sr_data):
        """在主线程中处理加载完成的音频数据。"""
        print(f"主线程收到加载完成信号: {file_path}")

        # 检查收到的文件路径是否是当前请求的路径（防止旧线程的延迟信号）
        # 注意：这里简化处理，假设最后一个启动的线程是有效的
        # 在复杂场景下，可能需要更严格的检查

        # 更新 AudioHandler 实例的数据
        self.audio_handler.audio_file_path = file_path
        self.audio_handler.y = y_data
        self.audio_handler.sr = sr_data
        # 创建时间轴 (现在在这里创建)
        self.audio_handler.time_axis = np.arange(len(y_data)) / sr_data
        print(f"AudioHandler 实例更新完成: SR={sr_data}, Samples={len(y_data)}")


        # 更新 UI
        loaded_path = self.audio_handler.get_file_path() # 从 handler 获取最终路径
        self.file_label.setText(f"已加载: {loaded_path}")
        self.statusBar().showMessage("音频加载完成，准备检测 Onsets")

        # 启用控件
        self.detect_button.setEnabled(True)
        self.play_main_checkbox.setEnabled(True)
        self.follow_playback_checkbox.setEnabled(True) # 启用跟随 (播放是否可用取决于 prepare)

        # 更新波形图
        self.update_waveform_plot() # 使用 handler 中的新数据

        # 准备主音轨源 (现在会触发 int16 转换)
        # 准备音频源也可能耗时，虽然通常比加载快，但如果需要，也可移至线程
        print("开始准备主音轨...")
        QApplication.processEvents() # 让UI有机会更新
        success = self._prepare_main_audio_source() # 这个函数内部会调用 get_playback_data
        if success:
            print("主音轨准备成功。")
            # 播放按钮和滑块的启用现在由 _update_duration 控制，它会在 prepare 成功后被触发
        else:
            print("主音轨准备失败。")
            self.statusBar().showMessage("音频加载完成，但准备播放失败")
            # 保持播放相关控件禁用状态


    # --- 新增：处理音频加载错误的槽函数 ---
    @Slot(str, object)
    def _audio_load_error(self, file_path, error):
        """在主线程中处理加载过程中发生的错误。"""
        print(f"主线程收到加载错误信号: {file_path}, Error: {error}")

        # 检查是否是当前文件加载出错
        # (简化处理，同上)

        # Corrected f-string
        error_msg = f"加载音频时出错: {file_path}, 错误详情: {error}"
        print(error_msg, file=sys.stderr)
        self.file_label.setText(f"加载失败: {file_path}")
        self.statusBar().showMessage(f"音频加载失败: {os.path.basename(file_path)}")

        # 显示错误消息给用户
        if isinstance(error, str) and error == "加载被取消":
             QMessageBox.information(self, "操作取消", f"音频文件加载已取消: {file_path}")
        else:
             QMessageBox.critical(self, "加载错误", f"无法加载音频文件:\n{file_path}\n\n{error}")

        # 确保状态完全重置 (按钮等在 open_file_dialog 开始时已禁用)
        self._reset_state_for_new_file() # 确保清理


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
         self.onset_times = np.array([]); self.clicks_audio = None; # REMOVED: self.clicks_audio_int16 = None
         # --- 重置 UI 元素 ---
         self.waveform_item.setData([], [])
         self.onset_scatter_item.setData([], [])
         self.detect_button.setEnabled(False)
         self.play_main_checkbox.setEnabled(False); self.play_main_checkbox.setChecked(True)
         self.play_clicks_checkbox.setEnabled(False); self.play_clicks_checkbox.setChecked(True)
         self._toggle_main_mute(True) # 确保初始未静音
         self._toggle_click_mute(True) # 确保初始未静音
         self.export_button.setEnabled(False)
         # --- ADDED: Reset follow checkbox ---
         self.follow_playback_checkbox.setEnabled(False)
         # --- END ADDED ---
         self.status_label.setText("")
         self.file_label.setText("尚未选择文件") # 重置文件标签
         self.statusBar().showMessage("准备就绪")


    def update_waveform_plot(self):
        # 从 AudioHandler 获取波形数据 (现在是 unnormalized float32)
        time_axis, y = self.audio_handler.get_waveform_data()
        # 检查数据是否存在且有效
        if y is not None and time_axis is not None and len(time_axis) > 0:
            print("正在更新波形图...")
            self.waveform_item.setData(time_axis, y)

            # 调整 Y 轴范围 (可能需要手动调整或保留之前的逻辑)
            # 由于数据未归一化，自动范围可能更合适，或者需要用户手动缩放
            self.plot_widget.enableAutoRange(axis=pg.ViewBox.YAxis)
            self.plot_widget.autoRange()
            # 如果希望加载后固定范围，可以在 autoRange 后禁用
            # QTimer.singleShot(0, lambda: self.plot_widget.disableAutoRange(axis=pg.ViewBox.YAxis))
            print("波形图更新完毕。")

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
        # --- ADDED: Disable follow checkbox ---
        self.follow_playback_checkbox.setEnabled(False)
        # --- END ADDED ---
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
            # --- ADDED: Re-enable follow checkbox if playback possible ---
            if self.media_player.duration() > 0:
                self.follow_playback_checkbox.setEnabled(True)
            # --- END ADDED ---
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
             self.clicks_audio = None; # REMOVED: self.clicks_audio_int16 = None
             self.onset_scatter_item.setData([], [])
             # Ensure click source is cleared if generation failed
             self.click_player.setSource(QUrl())
             self._cleanup_temp_file('temp_click_file_path')
             self.play_clicks_checkbox.setEnabled(False)
             # --- Ensure follow checkbox is disabled on error ---
             self.follow_playback_checkbox.setEnabled(False)
             # ---
        finally:
             self.detect_button.setEnabled(True) # Always re-enable detection button


    def update_onset_markers(self):
        # 从 handler 获取 float32 的 y 用于定位标记 (现在是未归一化的)
        _, y_float = self.audio_handler.get_waveform_data()

        if y_float is None or len(y_float) == 0: # Add length check
             self.onset_scatter_item.setData([], [])
             return

        if self.onset_times.size > 0:
            # 使用未归一化波形的最大绝对值来缩放标记位置
            y_max = np.max(np.abs(y_float)) if len(y_float) > 0 else 1.0
            marker_y_position = y_max * 1.05 # 定位在波形上方一点
            # 处理 y_max 可能为 0 的情况
            if abs(marker_y_position) < 1e-6: marker_y_position = 0.1 # Or some small default height

            scatter_data = [{'pos': (t, marker_y_position), 'data': t} for t in self.onset_times]
            self.onset_scatter_item.setData(scatter_data)
            print(f"更新了 {len(self.onset_times)} 个 Onset 标记。")
        else:
            self.onset_scatter_item.setData([], [])
            print("没有 Onset 标记可更新。")

    def generate_click_track(self):
        """Generates float32 click track (unnormalized)."""
        # 从 AudioHandler 获取参考数据 (float32 y unnormalized, sr)
        y_ref, sr_ref = self.audio_handler.get_detection_data()

        # REMOVED y_int16_ref check here, not needed for generation

        if y_ref is None or sr_ref is None: # 需要参考音频长度和采样率
            print("无法生成打点音轨：缺少参考音频数据。")
            self.clicks_audio = None
            # self.clicks_audio_int16 = None # Removed
            return

        print("正在生成打点声音...")
        click_freq = 1500
        click_duration_sec = 0.03
        click_frames = int(click_duration_sec * sr_ref) # 使用 handler 的 sr
        click_times_rel = np.arange(click_frames) / sr_ref # 使用 handler 的 sr
        # 生成点击信号，幅度可以根据需要调整，这里保持 0.6
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

        # --- 移除了在这里转换为 int16 的步骤 ---
        # print("Converting click audio to int16 for WAV playback.")
        # self.clicks_audio_int16 = ... (转换逻辑移到 _prepare_click_audio_source)

        print("打点声音生成完毕 (float32)。")


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
        """Updates the seek slider, time label, playback line, and optionally pans the view."""
        if not self.is_seeking: # Only update slider if user isn't dragging it
            max_val = self.seek_slider.maximum()
            # Prevent setting value beyond max during natural playback end
            if position <= max_val:
                 self.seek_slider.setValue(position)
            else: # Handle reaching the end
                 self.seek_slider.setValue(max_val)

        duration = self.media_player.duration()
        # Ensure duration is positive before formatting time
        valid_duration = duration if duration > 0 else 0
        self.time_label.setText(f"{self._format_time(position)} / {self._format_time(valid_duration)}")

        current_time_sec = position / 1000.0

        # --- 添加: 跟随播放逻辑 ---
        # 假设你有一个名为 follow_playback_checkbox 的 QCheckBox
        follow_checkbox = getattr(self, 'follow_playback_checkbox', None)
        # 检查复选框是否存在且被选中，并且播放器正在播放且用户没有在拖动滑块
        if (follow_checkbox and follow_checkbox.isChecked() and
                not self.is_seeking and
                self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState):

            view_duration = self.view_duration_spinbox.value() # 获取当前视图宽度（秒）
            view_box = self.plot_widget.getViewBox()
            current_range = view_box.viewRange()[0] # 获取当前视图的 X 轴范围 [start, end]
            view_start, view_end = current_range

            # --- Get Total Duration --- (Ensure it's valid before proceeding)
            total_duration_sec = self.audio_handler.get_duration()
            if total_duration_sec is None or total_duration_sec <= 0:
                print("Warning: Could not get valid total audio duration for follow playback.", file=sys.stderr)
                # Optionally, update the playback line position even if scrolling fails
                if self.playback_position_line:
                    self.playback_position_line.setPos(current_time_sec)
                return # Cannot calculate scroll without duration

            # --- Follow Playback Logic --- CHANGE: Handle out-of-view cases
            # print(f"Follow check: time={current_time_sec:.3f}, view=[{view_start:.3f}, {view_end:.3f}], duration={view_duration:.3f}, total_dur={total_duration_sec:.3f}") # DEBUG
            new_view_start = view_start # Default to current view
            new_view_end = view_end
            needs_update = False

            if current_time_sec < view_start:
                # Case 1: Playhead is before the current view -> Center the view
                # print(f"Follow Case 1: Before view") # DEBUG
                # print(f"Playhead ({current_time_sec:.2f}s) before view [{view_start:.2f}s, {view_end:.2f}s]. Centering view.")
                new_view_start = max(0.0, current_time_sec - view_duration / 2.0)
                new_view_end = min(total_duration_sec, new_view_start + view_duration)
                new_view_start = max(0.0, new_view_end - view_duration) # Adjust start based on clamped end
                needs_update = True
            elif current_time_sec >= view_end:
                # Case 2: Playhead is at or after the current view end -> Page scroll
                # print(f"Follow Case 2: At/After view end") # DEBUG
                # Prevent scrolling if playhead is already at the very end of the audio
                if view_end < total_duration_sec: # Only scroll if the current view doesn't already reach the end
                    # print(f"Playhead ({current_time_sec:.2f}s) reached/passed view end [{view_end:.2f}s]. Paging scroll.")
                    # print(f"Follow Case 2a: Paging scroll needed") # DEBUG
                    new_view_start = view_end # Start the new view where the old one ended
                    new_view_end = min(total_duration_sec, new_view_start + view_duration)
                    new_view_start = max(0.0, new_view_end - view_duration) # Adjust start based on clamped end
                    needs_update = True
                # else:
                     # print(f"Follow Case 2b: Already at end, no scroll") # DEBUG
                # else: playhead is at or beyond the end, or view already includes the end, no need to scroll further
            # else:
                 # print(f"Follow Case 3: Inside view, no scroll needed") # DEBUG
            # Case 3: view_start <= current_time_sec < view_end -> Playhead is inside, do nothing (needs_update is False)

            # --- Apply new view range if needed ---
            if needs_update:
                 # Optional: Check if the change is significant enough to avoid tiny adjustments
                 # (This might be less relevant now as jumps/pages should be significant)
                 if abs(new_view_start - view_start) > 0.001 or abs(new_view_end - view_end) > 0.001:
                     # print(f"Updating X range to: [{new_view_start:.2f}, {new_view_end:.2f}]")
                     # print(f"Follow Applying update: New range=[{new_view_start:.3f}, {new_view_end:.3f}]") # DEBUG
                     self.plot_widget.setXRange(new_view_start, new_view_end, padding=0)
                 # else:
                     # print(f"Follow Skipping minor update (range change too small)") # DEBUG
            # else:
            #    print("Follow No update needed flag") # DEBUG

        # --- 更新播放指示线位置 ---
        # 确保在视图可能更新后再设置指示线位置
        if self.playback_position_line:
            # 使用 current_time_sec (秒) 来设置指示线的位置
            self.playback_position_line.setPos(current_time_sec)
        # --- 结束: 跟随播放逻辑 ---

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
                 # --- ADDED: Enable follow checkbox ---
                 self.follow_playback_checkbox.setEnabled(True)
                 # --- END ADDED ---
            # Update time label immediately
            current_pos = self.media_player.position()
            self.time_label.setText(f"{self._format_time(current_pos)} / {self._format_time(duration)}")
        else:
            print("Duration invalid or zero.")
            self.seek_slider.setRange(0, 1) # Avoid division by zero etc.
            self.seek_slider.setEnabled(False)
            self.play_pause_button.setEnabled(False)
            # --- ADDED: Disable follow checkbox ---
            self.follow_playback_checkbox.setEnabled(False)
            # --- END ADDED ---
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
            self._reset_playback() # Reset playback UI on significant error (this will disable follow checkbox via _reset_state)

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

        # --- 获取当前音频文件目录作为默认导出目录 ---
        default_dir = "." # 默认为当前工作目录
        current_audio_path = self.audio_handler.get_file_path()
        if current_audio_path and os.path.exists(os.path.dirname(current_audio_path)): # Check if path exists
            default_dir = os.path.dirname(current_audio_path)
        # --- 结束获取默认目录 ---

        onsets_to_export = self.onset_times
        default_suffix = f".{format}"
        file_filter = f"文本文件 (*.{format})" if format == 'txt' else f"CSV 文件 (*.{format})"
        default_filename_base = "onsets_export" # 基础文件名

        if visible_only:
            # 从 handler 获取时间轴 (需要确保 audio_handler 已加载)
            time_axis, _ = self.audio_handler.get_waveform_data() # Get data from handler
            if time_axis is None or len(time_axis) == 0: # Check handler data validity
                 QMessageBox.information(self, "无视图", "无法确定当前视图范围。")
                 return
            current_x_range = self.plot_widget.getViewBox().viewRange()[0]
            view_start, view_end = current_x_range
            onsets_to_export = self.onset_times[
                (self.onset_times >= view_start) & (self.onset_times <= view_end)
            ]
            # 修改默认文件名逻辑
            default_filename_base = f"onsets_view_{view_start:.1f}s_to_{view_end:.1f}s"
            print(f"Exporting {len(onsets_to_export)} visible onsets in range [{view_start:.2f}, {view_end:.2f}]")
            if onsets_to_export.size == 0:
                 QMessageBox.information(self, "无可见数据", "当前视图范围内没有检测到的 Onset 点。")
                 return
        else:
             print(f"Exporting all {len(onsets_to_export)} onsets.")
             default_filename_base = "onsets_all" # 基础文件名

        # 组合完整默认文件名 (不含路径)
        default_filename = default_filename_base + default_suffix

        # Get save path - 使用获取到的 default_dir
        # 将 default_dir 和 default_filename 结合，作为getSaveFileName的第三个参数
        default_save_path = os.path.join(default_dir, default_filename)
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存 Onset 时间", default_save_path, file_filter # Pass the combined path here
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

    def _prepare_audio_source(self, player, get_data_func, temp_file_path_attr, description):
        """
        Generic function to prepare and set audio source for a player.
        Uses a function to get the int16 data and sample rate on demand.
        """
        print(f"准备 {description} 音频源...")
        # 调用提供的函数来获取 int16 数据和采样率
        audio_data_int16, sample_rate = get_data_func()

        if audio_data_int16 is None or sample_rate is None:
            print(f"无法准备 {description} 源: 获取数据失败。")
            player.setSource(QUrl()) # 清除源
            self._cleanup_temp_file(temp_file_path_attr) # 清理旧文件
            # 在主音轨失败时也禁用播放按钮和滑块
            if description == "main":
                 self.play_pause_button.setEnabled(False)
                 self.seek_slider.setEnabled(False)
                 self.follow_playback_checkbox.setEnabled(False) # Disable follow if main fails
            elif description == "click":
                 self.play_clicks_checkbox.setEnabled(False) # Disable click check if it fails

            return False

        print(f"获取到 {description} 的 int16 数据 (SR={sample_rate}), Samples={len(audio_data_int16)}")
        print(f"开始写入临时 WAV 文件: {temp_file_path_attr}")
        # --- UI 更新移到外面 ---
        # QApplication.processEvents()

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
            print(f"临时 {description} 音频写入到: {temp_file_path}")

            # Set the new source for the media player
            player.setSource(QUrl.fromLocalFile(temp_file_path))
            print(f"{description} source 设置成功。")
            return True

        except Exception as e:
            print(f"准备 {description} 音频源时出错: {e}", file=sys.stderr)
            QMessageBox.critical(self, "音频错误", f"准备{description}音轨源时出错: {e}")
            # Clean up the file if creation failed mid-way
            if temp_file:
                 temp_file.close() # Close handle first
            self._cleanup_temp_file(temp_file_path_attr) # Attempt removal
            player.setSource(QUrl()) # Ensure source is cleared
             # 在主音轨失败时也禁用播放按钮和滑块
            if description == "main":
                 self.play_pause_button.setEnabled(False)
                 self.seek_slider.setEnabled(False)
                 self.follow_playback_checkbox.setEnabled(False) # Disable follow if main fails
            elif description == "click":
                 self.play_clicks_checkbox.setEnabled(False) # Disable click check if it fails
            return False


    def _prepare_main_audio_source(self):
        """Prepares the main audio track using the generic function."""
        # 传递获取数据的函数给通用准备函数
        success = self._prepare_audio_source(self.media_player,
                                           self.audio_handler.get_playback_data, # Pass the method itself
                                           'temp_audio_file_path',
                                           "main")
        if success:
            # QMediaPlayer 会在源设置后异步发出 durationChanged 信号
            # 不需要在这里手动重置 UI，等待信号处理函数 _update_duration
            pass
        else:
             # 失败处理已在 _prepare_audio_source 内部完成 (禁用按钮等)
             self._reset_playback() # 也可以选择完全重置
        return success # 返回成功状态


    def _prepare_click_audio_source(self):
        """Prepares the click track audio using the generic function."""
        # 获取用于参考长度和采样率的 handler 数据 (float32)
        y_ref, sr_ref = self.audio_handler.get_detection_data()

        # 检查 click 音轨 float32 数据是否存在
        if self.clicks_audio is None:
             print("错误: 无法准备 click source，click audio 数据未生成。")
             self.play_clicks_checkbox.setEnabled(False)
             self.play_clicks_checkbox.setChecked(False)
             return False # 不能继续

        # 定义一个内部函数来获取点击音轨的 int16 数据
        def get_click_playback_data():
            if self.clicks_audio is None or sr_ref is None:
                 return None, None

            print("转换 click 音频为 int16 用于 WAV 播放。")
            clicks_float = self.clicks_audio
            # --- 使用与主音轨相同的方法进行缩放 ---
            max_abs = np.max(np.abs(clicks_float))
            clicks_int16 = None
            if max_abs > 1e-6:
                 clicks_scaled = clicks_float / max_abs
                 # 注意：这里的缩放基于 click 信号自身的最大值，
                 # 如果希望 click 音量相对于主音轨固定，可能需要不同的策略
                 clicks_int16 = (clicks_scaled * 32767 * 0.6).astype(np.int16) # 乘以 0.6 降低音量
            else:
                 clicks_int16 = np.zeros_like(clicks_float, dtype=np.int16)

            # 确保长度与参考音频一致 (如果需要)
            if y_ref is not None and len(clicks_int16) != len(y_ref):
                 print(f"Warning: Generated int16 click track length ({len(clicks_int16)}) differs from reference ({len(y_ref)}). Adjusting.")
                 target_len = len(y_ref)
                 current_len = len(clicks_int16)
                 if target_len > current_len:
                     clicks_int16 = np.pad(clicks_int16, (0, target_len - current_len))
                 elif target_len < current_len:
                     clicks_int16 = clicks_int16[:target_len]

            return clicks_int16, sr_ref # 使用参考采样率

        # 使用通用函数和上面定义的获取函数
        success = self._prepare_audio_source(self.click_player,
                                           get_click_playback_data, # Pass the inner function
                                           'temp_click_file_path',
                                           "click")
        if not success:
             print("禁用 click 音轨，因为准备失败。")
             self.play_clicks_checkbox.setEnabled(False)
             self.play_clicks_checkbox.setChecked(False) # Uncheck it
        return success

    # --- 新增：预初始化方法 ---
    def _try_preload_audio_backend(self):
        """
        Attempts to load a tiny silent audio array to trigger
        initialization of the audio backend (like soundfile).
        This might help prevent crashes on the *first* real file open
        if the issue is related to backend initialization in certain environments.
        """
        print("尝试预初始化音频后端...")
        try:
            # 创建一个非常短的静音 NumPy 数组 (例如 10 个采样点)
            sample_rate = 22050 # 使用一个常见的采样率
            silent_audio = np.zeros(10, dtype=np.float32)

            # 使用 soundfile 尝试将其写入临时文件并立即删除
            # 这会强制 soundfile 加载其底层库 (如 libsndfile)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_f:
                soundfile.write(temp_f.name, silent_audio, sample_rate)
            print("音频后端预初始化尝试成功。")
            # 可选：可以尝试用 librosa 加载一下，进一步确保
            # librosa.load(temp_f.name) # 因为 delete=True，这行会失败，但或许上面的 write 就够了
            # 或者更简单地调用一个 librosa 函数看看是否报错
            _ = librosa.get_duration(y=silent_audio, sr=sample_rate)
            print("Librosa 功能初步检查通过。")

        except Exception as e:
            # 如果预初始化失败，打印警告但不要让程序崩溃
            # 用户在实际打开文件时仍然可能会遇到问题
            print(f"警告：预初始化音频后端时发生错误: {e}", file=sys.stderr)
            # 可以选择弹出一个非阻塞的消息提示用户潜在问题
            # QMessageBox.warning(self, "初始化警告", f"尝试初始化音频系统时出错:\n{e}\n\n后续音频加载可能失败。")
        print("预初始化流程结束。")
    # --- 结束新增 ---


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OnsetDetectorApp()
    # --- 将 _try_preload_audio_backend 调用移到 __init__ 中 ---
    # # 尝试在显示窗口前进行预加载
    # print("Attempting audio backend preload before showing window...")
    # window._try_preload_audio_backend() # REMOVED from here
    # print("Preload attempt finished.")
    window.show()
    sys.exit(app.exec()) 
