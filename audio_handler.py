# audio_handler.py
import sys
import numpy as np
import librosa

class AudioHandler:
    def __init__(self):
        """初始化音频处理器，所有音频数据属性设为 None。"""
        self.audio_file_path = None
        self.y = None  # 波形数据 (float32, normalized)
        self.y_int16 = None # 波形数据 (int16 for playback)
        self.sr = None # 采样率
        self.time_axis = None # 时间轴

    def reset(self):
        """重置所有音频数据和文件路径。"""
        print("重置 AudioHandler 状态...")
        self.audio_file_path = None
        self.y = None
        self.y_int16 = None
        self.sr = None
        self.time_axis = None

    def load_audio(self, file_path):
        """
        从指定路径加载音频文件。

        Args:
            file_path (str): 音频文件的路径。

        Returns:
            bool: 如果加载和处理成功则返回 True，否则返回 False。
        """
        self.reset() # 先清空旧数据
        self.audio_file_path = file_path
        try:
            print(f"AudioHandler: 开始加载: {self.audio_file_path}")
            # 使用 librosa 加载，保持 float32 类型用于处理
            self.y, self.sr = librosa.load(self.audio_file_path, sr=None, mono=True)

            if self.y.dtype != np.float32:
                 self.y = self.y.astype(np.float32)

            # 归一化到 [-1.0, 1.0]
            max_abs = np.max(np.abs(self.y))
            if max_abs > 1e-6: # 避免除以零
                 self.y /= max_abs
            else:
                 print("AudioHandler Warning: 加载的音频似乎是静音或接近静音。", file=sys.stderr)
                 self.y[:] = 0.0 # 确保静音为全零

            # 创建时间轴
            self.time_axis = np.arange(len(self.y)) / self.sr

            # 转换为 int16 用于 WAV 播放
            print("AudioHandler: 转换音频为 int16 用于 WAV 播放。")
            self.y_int16 = (self.y * 32767).astype(np.int16)

            print(f"AudioHandler: 加载完成 - SR={self.sr}, Samples={len(self.y)}, Duration={self.get_duration():.2f}s")
            return True

        except Exception as e:
            print(f"AudioHandler: 加载或处理音频时出错 '{self.audio_file_path}': {e}", file=sys.stderr)
            self.reset() # 出错时重置状态
            return False

    def is_loaded(self):
        """检查音频是否已成功加载。"""
        return self.sr is not None and self.y is not None

    def get_duration(self):
        """获取音频时长（秒）。"""
        if self.is_loaded():
            return len(self.y) / self.sr
        return 0.0

    def get_waveform_data(self):
        """获取用于绘制波形图的数据。"""
        return self.time_axis, self.y

    def get_playback_data(self):
        """获取用于播放准备的数据 (int16)。"""
        return self.y_int16, self.sr

    def get_detection_data(self):
        """获取用于 Onset 检测的数据 (float32)。"""
        return self.y, self.sr

    def get_sample_rate(self):
        """获取采样率。"""
        return self.sr

    def get_file_path(self):
        """获取当前加载的音频文件路径。"""
        return self.audio_file_path
