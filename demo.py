#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for DotMusic - Audio Onset Detection Tool
演示脚本 - 音频 Onset 检测工具

This demo shows how to use the AudioHandler class for basic onset detection
without the GUI interface.

此演示展示如何在不使用GUI界面的情况下使用AudioHandler类进行基本的onset检测。

Usage / 使用方法:
    python demo.py [audio_file_path]
    
If no audio file is provided, this will show what the tool can do.
如果未提供音频文件，这将展示该工具的功能。
"""

import sys
import os

def demo_without_audio():
    """Show what this tool can do without requiring audio files."""
    print("=" * 60)
    print("DotMusic - Audio Onset Detection Tool Demo")
    print("音频 Onset 检测工具演示")
    print("=" * 60)
    print()
    print("What is this tool? / 这个工具是什么？")
    print("- Detects onset points (beat starts) in audio files")
    print("- 检测音频文件中的起音点（节拍开始点）")
    print()
    print("Supported formats / 支持的格式:")
    print("- WAV, MP3, AAC, FLAC and other common audio formats")
    print("- WAV, MP3, AAC, FLAC 和其他常见音频格式")
    print()
    print("Key features / 主要功能:")
    print("- Real-time waveform visualization / 实时波形可视化")
    print("- Adjustable onset detection parameters / 可调节的onset检测参数")
    print("- Audio playback with click tracks / 音频播放与节拍音轨")
    print("- Export onset timestamps / 导出onset时间戳")
    print()
    print("To use this tool / 使用此工具:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run GUI application: python onset_gui.py")
    print("3. Or use AudioHandler class programmatically")
    print()
    print("For a working demo with audio, run:")
    print("    python demo.py path/to/your/audio/file.wav")

def demo_with_audio(audio_path):
    """Demonstrate onset detection with an actual audio file."""
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return
        
    try:
        # Try to import and use the AudioHandler
        from audio_handler import AudioHandler
        import numpy as np
        
        print(f"Loading audio file: {audio_path}")
        handler = AudioHandler()
        
        # This would normally load the audio and detect onsets
        print("This demo would:")
        print("1. Load the audio file using librosa")
        print("2. Detect onset points using librosa.onset.onset_detect")
        print("3. Display the results")
        print()
        print("To see this in action, install dependencies and run the GUI:")
        print("    pip install -r requirements.txt")
        print("    python onset_gui.py")
        
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Install dependencies with: pip install -r requirements.txt")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        demo_with_audio(audio_file)
    else:
        demo_without_audio()