# Extracted Keyframes Directory

This directory stores the sharp keyframes extracted from walkthrough videos via the Laplacian blur-detection filter.

### Expected Files:
- Keyframes saved as: `frame_00000.jpg`, `frame_00001.jpg`, etc.
- Resolution: 1280x720 HD or original 1080p.

### How to Generate:
To generate keyframes locally from `test/sample_video.mp4`:
```powershell
uv run --python 3.11 --with opencv-python --with numpy python pipeline/extract_frames.py --input test/sample_video.mp4 --output test/extracted_frames --threshold 35 --fps 3
```
