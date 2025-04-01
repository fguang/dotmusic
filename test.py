# Onset detection example (more suitable for rhythm game notes)
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display

# 1. Get the file path to your audio file
# filename = './【重音Teto】ウキシマ（浮岛） - 001 - 【重音Teto】ウキシマ（浮岛）.mp3'

filename = './Rx 预览 - The Observer - 001 - Rx 预览 - The Observer.mp3'

# 2. Load the audio as a waveform `y` and store the sampling rate `sr`
print(f"加载音频文件: {filename}")
y, sr = librosa.load(filename)

# --- 使用 Onset Detection 寻找音符起始点 ---
print("\n开始检测音符起始点 (Onsets)...")

# 3. 使用 onset_detect 函数查找 onset 时间点
# --- 修改：尝试提高灵敏度 ---
# 降低 delta 值，使其更容易检测到 onset。默认值是根据标准差计算的，
# 这里我们尝试一个较小的值。如果检测到太多无关的点，可以适当调高 delta。
# 如果还是太少，可以尝试更小的 delta 或调整其他参数 (units='frames' 然后手动处理)。
onset_delta = 0.028 # <--- 调低 delta (例如 0.05 或 0.1) 尝试提高灵敏度
print(f"使用 onset_detect, delta={onset_delta} (较低的值更敏感)")
onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True, delta=onset_delta)
print(f"总共检测到 {len(onset_times)} 个 Onset 点。")

# --- 修改：定义观察窗口，而不是固定点数 ---
start_time_sec = 0.0    # <--- 在这里设置你希望开始查看的时间点（单位：秒）
view_duration_sec = 360.0 # <--- 设置观察窗口的持续时间（例如 10 秒）

# 找到第一个大于等于 start_time_sec 的 onset 点的索引
start_index = np.searchsorted(onset_times, start_time_sec)

if start_index >= len(onset_times):
    print(f"在 {start_time_sec} 秒之后没有检测到 Onset 点。")
else:
    # 确定观察窗口的开始和结束时间
    first_onset_in_range = onset_times[start_index]
    view_start = start_time_sec # 窗口从指定的开始时间精确开始
    view_end = start_time_sec + view_duration_sec

    # 筛选出所有落在观察窗口 [view_start, view_end) 内的 onset 点
    onsets_in_view = onset_times[(onset_times >= view_start) & (onset_times < view_end)]

    if len(onsets_in_view) == 0:
        print(f"在时间窗口 [{view_start:.2f}, {view_end:.2f}) 秒内没有检测到 Onset 点。")
    else:
        print(f"\n显示在时间窗口 [{view_start:.2f}, {view_end:.2f}) 秒内的 {len(onsets_in_view)} 个 Onset 点:")
        print(onsets_in_view)

        # --- 修改：可视化观察窗口内的 Onset 点 ---
        print("\n正在生成固定时间窗口的 Onset 可视化图像 (带时间标签)...")

        plt.figure(figsize=(15, 5)) # 创建一个图形

        # 绘制音频波形图
        librosa.display.waveshow(y, sr=sr, alpha=0.6)

        # 确定标记点的 Y 轴位置
        marker_y_position = 1.05

        # 只绘制在观察窗口内的 onset 点
        plt.plot(onsets_in_view,
                 [marker_y_position] * len(onsets_in_view),
                 marker='o', color='r', linestyle='none', markersize=6,
                 label=f'Onsets in [{view_start:.1f}s, {view_end:.1f}s)')

        # 为每个在窗口内的点添加时间文本标签
        text_y_offset = 0.05
        for time_val in onsets_in_view:
            plt.text(x=time_val,
                     y=marker_y_position + text_y_offset,
                     s=f'{time_val:.3f}',
                     color='blue', fontsize=8, ha='center', va='bottom')

        # --- 添加网格线 ---
        plt.grid(True, linestyle=':', alpha=0.7)

        # 设置 x 轴的显示范围严格等于观察窗口
        plt.xlim(view_start, view_end)
        # 调整 Y 轴范围确保文本标签可见
        current_ylim = plt.ylim()
        plt.ylim(current_ylim[0], max(current_ylim[1], marker_y_position + text_y_offset + 0.1))

        plt.title(f'音频波形与 [{view_start:.1f}s, {view_end:.1f}s) 时间窗口内的 Onsets (带时间标签)')
        plt.xlabel('时间 (秒)')
        plt.ylabel('幅度')
        plt.legend()
        plt.tight_layout()

        # 保存图像文件 (文件名反映变化)
        output_plot_filename = f'./【重音Teto】ウキシマ（浮岛）_onsets_labels_{view_start:.0f}s_to_{view_end:.0f}s.png'
        plt.savefig(output_plot_filename)
        print(f"已生成聚焦的 Onset 可视化图像文件 (固定窗口): {output_plot_filename}")
        # plt.show()

        # --- 修改：使用观察窗口内的 Onset 时间点进行听觉验证 ---
        print("\n生成带有点击的音频 (仅包含观察窗口内的 Onsets)...")

        # 生成咔哒声信号 (代码不变)
        click_freq = 1500
        click_duration = 0.03
        click_frames = int(click_duration * sr)
        click_times = np.arange(click_frames) / sr
        click_signal = 0.6 * np.sin(2 * np.pi * click_freq * click_times)

        clicks = np.zeros_like(y)

        # 将 *观察窗口内* 的 Onset 时间转换为样本索引
        selected_onset_samples = librosa.time_to_samples(onsets_in_view, sr=sr)

        # 在 *观察窗口内* 的每个 Onset 样本位置叠加咔哒声
        for onset_sample in selected_onset_samples:
            start = onset_sample
            end = min(onset_sample + click_frames, len(y))
            click_len = end - start
            if start < len(clicks):
                clicks[start:end] += click_signal[:click_len]

        click_weight = 1.0
        y_onsets_selected = y + click_weight * clicks
        y_onsets_selected = np.clip(y_onsets_selected, -1, 1)

        # 保存仅包含部分点击的音频文件 (文件名区分)
        output_filename = f'./【重音Teto】ウキシマ（浮岛）_with_onsets_{view_start:.0f}s_to_{view_end:.0f}s.wav'
        sf.write(output_filename, y_onsets_selected, sr)

        print(f"\n已生成仅包含部分 Onset 点击的音频文件: {output_filename}")
        print("请播放此文件以验证指定时间窗口内的 Onset 检测效果。")
    print("请播放此文件以验证指定时间段内的 Onset 检测效果。")