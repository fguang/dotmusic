# DotMusic - Audio Onset Detection Tool

## What is this? / 这是什么？

DotMusic is an audio analysis tool that automatically detects **onset points** (起音点) in music and audio files. Onset points are the moments when new notes or beats begin, making this tool valuable for:

- **Music Analysis**: Find beat patterns and rhythm structures
- **Music Education**: Visualize timing and rhythm for students
- **Audio Editing**: Get precise timestamps for editing decisions
- **Research**: Analyze musical timing and structure

## Key Features / 主要功能

- 🎵 **Real-time Waveform Visualization** - See your audio as it plays
- 🎯 **Automatic Onset Detection** - Find beat points automatically 
- 🎛️ **Adjustable Parameters** - Fine-tune detection sensitivity
- 📊 **Visual Markers** - See onset points overlaid on waveform
- 💾 **Export Results** - Save onset timestamps to TXT/CSV
- 🔊 **Audio Playback** - Play original audio with optional click track
- 📱 **User-friendly GUI** - Easy-to-use interface built with Qt

## Supported Formats / 支持格式

WAV, MP3, AAC, FLAC, and other common audio formats

## Quick Start / 快速开始

1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `python onset_gui.py`
3. Open an audio file and start analyzing!

For detailed instructions, see [readme.md](readme.md).