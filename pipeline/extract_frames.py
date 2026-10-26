#!/usr/bin/env python3
"""
Namma Space – OpenCV Laplacian Blur-Filtered Keyframe Extractor
================================================================
Extracts sharp, non-blurry keyframes from indoor walkthrough videos
using OpenCV's Laplacian variance as a sharpness metric.

Usage:
    python extract_frames.py --input video.mp4 --output ./frames --threshold 60
    python extract_frames.py --input video.mp4 --output ./frames --fps 3 --max-width 1920
"""

import argparse
import os
import sys
import time

# Ensure safe UTF-8 terminal output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: OpenCV and NumPy are required.")
    print("Install them with: pip install opencv-python numpy")
    sys.exit(1)


def compute_blur_score(frame):
    """Compute the Laplacian variance of a frame as a sharpness metric.

    Higher values indicate sharper images. Motion-blurred or out-of-focus
    frames produce lower values.

    Args:
        frame: BGR image (numpy array).

    Returns:
        float: Laplacian variance score.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def resize_frame(frame, max_width):
    """Resize frame to fit within max_width while preserving aspect ratio.

    Args:
        frame: BGR image (numpy array).
        max_width: Maximum output width in pixels.

    Returns:
        Resized frame (numpy array).
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_frames(input_path, output_dir, threshold=60.0, sample_fps=2.0, max_width=1280):
    """Extract sharp keyframes from a video file.

    Samples frames at the specified FPS rate, computes the Laplacian
    variance blur score, and saves only frames above the threshold.

    Args:
        input_path: Path to the input video file.
        output_dir: Directory to save extracted frames.
        threshold: Minimum Laplacian variance to consider a frame sharp.
        sample_fps: Number of frames to sample per second of video.
        max_width: Maximum output width in pixels.

    Returns:
        tuple: (total_frames_checked, saved_count, rejected_count)
    """
    if not os.path.isfile(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {input_path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    # Calculate frame sampling interval
    sample_interval = max(1, int(video_fps / sample_fps))

    print(f"{'='*60}")
    print(f"  Namma Space – Keyframe Extractor")
    print(f"{'='*60}")
    print(f"  Input:        {input_path}")
    print(f"  Output:       {output_dir}")
    print(f"  Video FPS:    {video_fps:.1f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration:     {duration:.1f}s")
    print(f"  Sample Rate:  {sample_fps} FPS (every {sample_interval} frames)")
    print(f"  Blur Thresh:  {threshold}")
    print(f"  Max Width:    {max_width}px")
    print(f"{'='*60}")

    frame_idx = 0
    checked = 0
    saved = 0
    rejected = 0
    scores = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            score = compute_blur_score(frame)
            scores.append(score)
            checked += 1

            if score >= threshold:
                resized = resize_frame(frame, max_width)
                filename = f"frame_{saved:05d}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
                status = f"  [+] SHARP"
            else:
                rejected += 1
                status = f"  [-] BLURRY"

            # Progress output
            progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            print(f"  [{progress:5.1f}%] Frame {frame_idx:6d} | Score: {score:8.2f} |{status}")

        frame_idx += 1

    cap.release()
    elapsed = time.time() - start_time

    # Summary statistics
    avg_score = np.mean(scores) if scores else 0
    min_score = np.min(scores) if scores else 0
    max_score = np.max(scores) if scores else 0

    print(f"\n{'='*60}")
    print(f"  EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Frames Checked:  {checked}")
    print(f"  Frames Saved:    {saved} ({saved/checked*100:.1f}% pass rate)" if checked > 0 else "")
    print(f"  Frames Rejected: {rejected}")
    print(f"  Blur Scores:     min={min_score:.1f}  avg={avg_score:.1f}  max={max_score:.1f}")
    print(f"  Time Elapsed:    {elapsed:.1f}s")
    print(f"  Output Dir:      {output_dir}")
    print(f"{'='*60}")

    return checked, saved, rejected


def main():
    parser = argparse.ArgumentParser(
        description="Namma Space – Extract sharp keyframes from indoor walkthrough videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_frames.py --input video.mp4 --output ./frames
  python extract_frames.py --input video.mp4 --output ./frames --threshold 80 --fps 3
  python extract_frames.py --input test/sample_video.mp4 --output test/extracted_frames
        """
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input video file (MP4, MOV, AVI)")
    parser.add_argument("--output", "-o", required=True, help="Output directory for extracted frames")
    parser.add_argument("--threshold", "-t", type=float, default=35.0,
                        help="Laplacian variance threshold for sharpness (default: 35.0 for indoor rooms, 60.0 for bright outdoor)")
    parser.add_argument("--fps", "-f", type=float, default=3.0,
                        help="Frame sampling rate in FPS (default: 3.0)")
    parser.add_argument("--max-width", "-w", type=int, default=1280,
                        help="Maximum output image width in pixels (default: 1280)")

    args = parser.parse_args()

    extract_frames(
        input_path=args.input,
        output_dir=args.output,
        threshold=args.threshold,
        sample_fps=args.fps,
        max_width=args.max_width
    )


if __name__ == "__main__":
    main()
