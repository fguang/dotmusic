# audio_handler.py
import sys
import numpy as np
import librosa

class AudioHandler:
    def __init__(self):
        """初始化音频处理器，所有音频数据属性设为 None。"""
        self.audio_file_path = None
        self.y = None  # 波形数据 (float32, UNNORMALIZED)
        self.sr = None # 采样率
        self.time_axis = None # 时间轴

    def reset(self):
        """重置所有音频数据和文件路径。"""
        print("重置 AudioHandler 状态...")
        self.audio_file_path = None
        self.y = None
        self.sr = None
        self.time_axis = None

    def load_audio(self, file_path):
        """
        从指定路径加载音频文件 (仅解码和类型设置)。
        注意：此方法现在应该在后台线程中调用。

        Args:
            file_path (str): 音频文件的路径。

        Returns:
            tuple: 成功时返回 (y, sr)，失败时返回 (None, None)。
                   或者可以修改为返回 bool 并设置 self.y, self.sr，
                   但返回数据更适合线程模式。我们返回数据。
        Raises:
            Exception: 如果加载或处理过程中发生错误。
        """
        # 不再在这里重置，因为线程可能操作共享实例，调用者负责重置
        # self.reset()
        print(f"AudioHandler: 开始加载 (线程中): {file_path}")
        # 使用 librosa 加载，保持 float32 类型用于处理
        y, sr = librosa.load(file_path, sr=None, mono=True)

        if y.dtype != np.float32:
             y = y.astype(np.float32)

        # --- 移除了归一化 ---
        # max_abs = np.max(np.abs(y))
        # if max_abs > 1e-6:
        #      y /= max_abs
        # else:
        #      print("AudioHandler Warning: 加载的音频似乎是静音或接近静音。", file=sys.stderr)
        #      y[:] = 0.0

        # --- 移除了 int16 转换 ---
        # print("AudioHandler: 不再预先转换音频为 int16。")
        # self.y_int16 = (self.y * 32767).astype(np.int16)

        print(f"AudioHandler: 加载完成 (线程中) - SR={sr}, Samples={len(y)}")
        # 返回加载的数据，而不是设置实例属性，让调用线程来设置
        return y, sr
        # 注意：时间轴现在也需要在加载完成后在主线程中创建

        # --- 修改了错误处理方式，直接抛出异常由线程捕获 ---
        # except Exception as e:
        #     print(f"AudioHandler: 加载或处理音频时出错 '{file_path}': {e}", file=sys.stderr)
        #     # self.reset() # 不应在可能共享的实例中重置
        #     return None, None # 返回 None 表示失败

    def is_loaded(self):
        """检查音频数据是否已在实例中设置。"""
        # 现在检查 self.y 和 self.sr 是否已由主线程设置
        return self.sr is not None and self.y is not None

    def get_duration(self):
        """获取音频时长（秒）。"""
        if self.is_loaded():
            return len(self.y) / self.sr
        return 0.0

    def get_waveform_data(self):
        """获取用于绘制波形图的数据 (float32, unnormalized)。"""
        # 确保 time_axis 已被创建
        if self.time_axis is None and self.is_loaded():
             print("Warning: 时间轴未创建，将在 get_waveform_data 中创建。")
             self.time_axis = np.arange(len(self.y)) / self.sr
        return self.time_axis, self.y

    def get_playback_data(self):
        """
        获取用于播放准备的数据 (int16)。
        如果需要，执行从 float32 到 int16 的转换。
        """
        if not self.is_loaded():
            print("错误：无法获取播放数据，音频未加载。")
            return None, None

        # --- 在需要时执行转换和缩放 ---
        print("AudioHandler: 转换音频为 int16 用于 WAV 播放。")
        y_float = self.y
        max_abs = np.max(np.abs(y_float))
        y_int16 = None
        if max_abs > 1e-6: # 避免除以零
             # 缩放 float 数据到 [-1, 1] 再乘以 32767
             y_scaled = y_float / max_abs
             y_int16 = (y_scaled * 32767).astype(np.int16)
        else:
             # 如果音频是静音，则生成全零的 int16 数组
             print("AudioHandler Warning: 音频似乎是静音，生成零 int16 数据。")
             y_int16 = np.zeros_like(y_float, dtype=np.int16)

        return y_int16, self.sr

    def get_detection_data(self):
        """获取用于 Onset 检测的数据 (float32, unnormalized)。"""
        # 直接返回未归一化的 float32 数据
        return self.y, self.sr

    def get_sample_rate(self):
        """获取采样率。"""
        return self.sr

    def get_file_path(self):
        """获取当前加载的音频文件路径。"""
        return self.audio_file_path
