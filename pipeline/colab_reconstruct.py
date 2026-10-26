#!/usr/bin/env python3
"""
Namma Space – Automated Headless Colab Reconstruction Pipeline
==============================================================
Designed to run directly on Google Colab GPU instances via the
Google Colab CLI (`colab exec -f pipeline/colab_reconstruct.py`)
or directly inside a remote Colab session.

Workflow:
1. Auto-configures CUDA COLMAP, FFmpeg, PyMeshLab, and Trimesh
2. Extracts sharp keyframes from /content/sample_video.mp4
3. Executes GPU SfM (SIFT) + PatchMatch Stereo dense reconstruction
4. Meshes via Screened Poisson (depth=9) & decimates to 150K faces
5. Exports /content/room.glb ready for one-command download
"""

import glob
import os
import sys
import time

VIDEO_PATH = "/content/sample_video.mp4"
WORKSPACE = "/content/workspace"
IMAGES_DIR = os.path.join(WORKSPACE, "images")
SPARSE_DIR = os.path.join(WORKSPACE, "sparse")
DENSE_DIR = os.path.join(WORKSPACE, "dense")
DATABASE_PATH = os.path.join(WORKSPACE, "database.db")
FUSED_PLY = os.path.join(DENSE_DIR, "fused.ply")
MESHED_PLY = os.path.join(WORKSPACE, "room.ply")
OUTPUT_GLB = "/content/room.glb"

BLUR_THRESHOLD = 35.0  # Optimal for realistic indoor room lighting
SAMPLE_FPS = 3.0       # 3 FPS keyframe sampling
MAX_WIDTH = 1280
TARGET_FACES = 150000


def run_command(name, cmd, fallback_cmd=None):
    print(f"\n{'='*60}\n  ▶ {name}\n{'='*60}")
    ret = os.system(cmd)
    if ret != 0:
        if fallback_cmd:
            print(f"⚠️ GPU mode returned code {ret}. Automatically running CPU fallback...")
            ret = os.system(fallback_cmd)
        if ret != 0:
            print(f"❌ {name} failed with code {ret}")
            return False
    print(f"✅ {name} complete")
    return True


def step1_setup_environment():
    print("=" * 60)
    print("  STEP 1: Setting up CUDA Environment & Dependencies")
    print("=" * 60)

    # System libraries
    os.system("apt-get update -qq && apt-get install -y -qq xvfb ffmpeg libglew-dev libqt5opengl5-dev libfreeimage-dev > /dev/null 2>&1")

    # Install precompiled CUDA COLMAP (Ubuntu 22.04 + CUDA + Caspar)
    if os.system("colmap --help > /dev/null 2>&1") != 0:
        print("📦 Downloading CUDA-accelerated COLMAP binary...")
        gpu_url = "https://github.com/lyehe/build_gpu_colmap/releases/download/v4.2.1/COLMAP-4.2.1-ubuntu-22.04-CUDA-Caspar.zip"
        extract_cmd = (
            f"wget -q -O /tmp/colmap.zip {gpu_url} && "
            "rm -rf /tmp/colmap_pkg && "
            "unzip -q -o /tmp/colmap.zip -d /tmp/colmap_pkg && "
            "find /tmp/colmap_pkg -type f -name colmap -exec cp {} /usr/local/bin/ \\; && "
            "chmod +x /usr/local/bin/colmap && "
            "find /tmp/colmap_pkg -mindepth 2 -type d -name lib -exec cp -r {}/* /usr/local/lib/ \\; 2>/dev/null && "
            "ldconfig && "
            "rm -rf /tmp/colmap.zip /tmp/colmap_pkg"
        )
        ret = os.system(extract_cmd)
        if ret != 0 or os.system("colmap --help > /dev/null 2>&1") != 0:
            print("⚠️ Prebuilt download issue, using standard package manager...")
            os.system("apt-get install -y -qq colmap > /dev/null 2>&1")

    # Python dependencies
    print("📦 Installing Python dependencies...")
    os.system("pip install -q opencv-python-headless pymeshlab trimesh[easy] numpy")

    # Verify GPU
    print("\n🖥️ GPU Status:")
    os.system("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No NVIDIA GPU detected'")

    # Directories
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(SPARSE_DIR, exist_ok=True)
    os.makedirs(DENSE_DIR, exist_ok=True)
    print("✅ Environment ready")


def step2_extract_frames():
    print("\n" + "=" * 60)
    print("  STEP 2: OpenCV Laplacian Sharp Frame Extraction")
    print("=" * 60)

    import cv2
    import numpy as np

    if not os.path.exists(VIDEO_PATH):
        # Look for any mp4 in /content
        mp4s = glob.glob("/content/*.mp4")
        if mp4s:
            v_path = mp4s[0]
            print(f"ℹ️ Using detected video: {v_path}")
        else:
            raise FileNotFoundError(f"Video file not found at {VIDEO_PATH}. Upload video first.")
    else:
        v_path = VIDEO_PATH

    cap = cv2.VideoCapture(v_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, int(fps / SAMPLE_FPS))

    print(f"  Video:        {v_path}")
    print(f"  Total Frames: {total_frames} ({total_frames/fps:.1f}s)")
    print(f"  Sampling:     every {sample_interval} frames (~{SAMPLE_FPS} FPS)")
    print(f"  Threshold:    {BLUR_THRESHOLD}")

    frame_idx, saved, rejected = 0, 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if score >= BLUR_THRESHOLD:
                h, w = frame.shape[:2]
                if w > MAX_WIDTH:
                    scale = MAX_WIDTH / w
                    frame = cv2.resize(frame, (MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
                out_name = os.path.join(IMAGES_DIR, f"frame_{saved:05d}.jpg")
                cv2.imwrite(out_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
            else:
                rejected += 1
        frame_idx += 1

    cap.release()
    print(f"✅ Extracted {saved} sharp keyframes (discarded {rejected} blurry frames)")
    if saved < 15:
        raise RuntimeError(f"Too few sharp frames extracted ({saved}). Try lowering BLUR_THRESHOLD.")


def step3_colmap_reconstruction():
    print("\n" + "=" * 60)
    print("  STEP 3: COLMAP Structure-from-Motion & Dense MVS")
    print("=" * 60)

    if os.path.exists(DATABASE_PATH):
        os.remove(DATABASE_PATH)

    # 1. Feature extraction
    cmd_gpu = f"colmap feature_extractor --database_path {DATABASE_PATH} --image_path {IMAGES_DIR} --ImageReader.single_camera 1 --SiftExtraction.use_gpu 1"
    cmd_cpu = f"colmap feature_extractor --database_path {DATABASE_PATH} --image_path {IMAGES_DIR} --ImageReader.single_camera 1 --SiftExtraction.use_gpu 0 --SiftExtraction.num_threads 4"
    run_command("SIFT Feature Extraction", cmd_gpu, cmd_cpu)

    # 2. Sequential matching
    cmd_match_gpu = f"colmap sequential_matcher --database_path {DATABASE_PATH} --SiftMatching.use_gpu 1"
    cmd_match_cpu = f"colmap sequential_matcher --database_path {DATABASE_PATH} --SiftMatching.use_gpu 0 --SiftMatching.num_threads 4"
    run_command("Sequential Feature Matching", cmd_match_gpu, cmd_match_cpu)

    # 3. Sparse Mapper
    cmd_mapper = f"colmap mapper --database_path {DATABASE_PATH} --image_path {IMAGES_DIR} --output_path {SPARSE_DIR}"
    run_command("Sparse Reconstruction (Mapper)", cmd_mapper)

    model_dirs = [os.path.join(SPARSE_DIR, d) for d in os.listdir(SPARSE_DIR) if os.path.isdir(os.path.join(SPARSE_DIR, d))]
    if not model_dirs:
        raise RuntimeError("Sparse reconstruction produced no model. Check input video quality.")
    sparse_0 = model_dirs[0]

    # 4. Undistort
    cmd_undistort = f"colmap image_undistorter --image_path {IMAGES_DIR} --input_path {sparse_0} --output_path {DENSE_DIR} --output_type COLMAP"
    run_command("Image Undistortion", cmd_undistort)

    # 5. PatchMatch Stereo
    cmd_pm = f"xvfb-run -a colmap patch_match_stereo --workspace_path {DENSE_DIR} --workspace_format COLMAP --PatchMatchStereo.geom_consistency true --PatchMatchStereo.gpu_index 0"
    pm_ok = run_command("Dense Stereo (PatchMatch CUDA)", cmd_pm)

    if pm_ok:
        cmd_fusion = f"colmap stereo_fusion --workspace_path {DENSE_DIR} --workspace_format COLMAP --input_type geometric --output_path {FUSED_PLY}"
        run_command("Stereo Fusion", cmd_fusion)

    # Fallback to sparse point cloud if dense stereo is unavailable
    if not os.path.exists(FUSED_PLY) or os.path.getsize(FUSED_PLY) == 0:
        print("\n⚠️ Note: Dense stereo unavailable; converting sparse point cloud to PLY...")
        cmd_convert = f"colmap model_converter --input_path {sparse_0} --output_path {FUSED_PLY} --output_type PLY"
        os.system(cmd_convert)

    if not os.path.exists(FUSED_PLY):
        raise FileNotFoundError(f"Failed to produce point cloud at {FUSED_PLY}")

    size_mb = os.path.getsize(FUSED_PLY) / (1024 * 1024)
    print(f"✅ Point cloud ready: {FUSED_PLY} ({size_mb:.1f} MB)")


def step4_mesh_and_export():
    print("\n" + "=" * 60)
    print("  STEP 4: Screened Poisson Meshing & GLB Export")
    print("=" * 60)

    import pymeshlab
    import trimesh

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(FUSED_PLY)
    print(f"  Loaded points: {ms.current_mesh().vertex_number()}")

    # Compute normals with larger neighborhood for robust indoor orientation
    ms.compute_normal_for_point_sets(k=20, smoothiter=3, flipflag=True)
    print("  ✅ Point normals computed (k=20, smoothiter=3)")

    # Screened Poisson Surface Reconstruction
    # scale=1.0 prevents extrapolation beyond point cloud boundary
    ms.generate_surface_reconstruction_screened_poisson(depth=9, preclean=True, scale=1.0)
    print(f"  ✅ Poisson surface: {ms.current_mesh().vertex_number()} verts, {ms.current_mesh().face_number()} faces")

    # Trim low-confidence Poisson extrapolation using per-vertex quality scalar
    # Poisson stores a quality value; low-quality vertices are phantom geometry
    try:
        ms.compute_selection_by_condition_per_vertex(condselect="(q < 0.5)")
        ms.meshing_remove_selected_vertices_and_faces()
        print(f"  ✅ Trimmed low-confidence regions: {ms.current_mesh().face_number()} faces remaining")
    except Exception as e:
        print(f"  ℹ️ Quality trimming skipped: {e}")

    # Remove small disconnected components (noise islands)
    try:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=100)
        print(f"  ✅ Removed small components: {ms.current_mesh().face_number()} faces remaining")
    except Exception as e:
        print(f"  ℹ️ Component cleanup skipped: {e}")

    # Decimate to 150K faces
    if ms.current_mesh().face_number() > TARGET_FACES:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=TARGET_FACES, preservenormal=True)
        print(f"  ✅ Decimated to {ms.current_mesh().face_number()} faces")

    # Transfer vertex colors if available
    try:
        if ms.mesh_number() > 1:
            ms.set_current_mesh(1)
            ms.transfer_attributes_per_vertex(sourcemesh=0, targetmesh=1)
            print("  ✅ Vertex colors transferred")
    except Exception as e:
        print(f"  ℹ️ Color transfer notice: {e}")

    ms.save_current_mesh(MESHED_PLY)

    # Convert to binary GLB
    mesh = trimesh.load(MESHED_PLY, force="mesh")
    glb_data = mesh.export(file_type="glb")
    with open(OUTPUT_GLB, "wb") as f:
        f.write(glb_data)

    glb_size = os.path.getsize(OUTPUT_GLB) / (1024 * 1024)
    print(f"\n🎉 EXPORT SUCCESSFUL: {OUTPUT_GLB} ({glb_size:.1f} MB)")
    print("   Download using: colab download /content/room.glb public/models/room.glb")


def main():
    start_time = time.time()
    step1_setup_environment()
    step2_extract_frames()
    step3_colmap_reconstruction()
    step4_mesh_and_export()
    elapsed = time.time() - start_time
    print(f"\n⏱️ Full pipeline completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
