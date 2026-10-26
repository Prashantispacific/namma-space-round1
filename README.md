# 🚀 Namma Space — 3D Indoor Reconstruction & Web Walkthrough

**Techfest IIT Bombay 2026-27 | Round 1 Submission**

> Transform smartphone indoor walkthrough videos into photorealistic, interactive 3D digital twins — explorable at 60 FPS in any web browser, on any device.

[![Live Demo](https://img.shields.io/badge/Live_Demo-namma--space.netlify.app-00C7B7?style=for-the-badge&logo=netlify)](https://namma-space.netlify.app/)
[![Colab Pipeline](https://img.shields.io/badge/Pipeline-Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/)
[![Interactive Guide](https://img.shields.io/badge/Architecture-Interactive_Guide_HTML-7928CA?style=for-the-badge)](guide.html)

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     NAMMA SPACE PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📱 CAPTURE          ☁️ RECONSTRUCTION (Colab T4 GPU)           │
│  ┌──────────┐       ┌──────────────────────────────────────┐   │
│  │Smartphone│──MP4──▶│ OpenCV Laplacian Frame Extraction    │   │
│  │Walkthrough│       │ (blur-filter @ 2 FPS sampling)       │   │
│  │ Video    │       ├──────────────────────────────────────┤   │
│  └──────────┘       │ COLMAP Structure-from-Motion         │   │
│                     │ • GPU SIFT Feature Extraction         │   │
│                     │ • Sequential Matching + Loop Detect   │   │
│                     │ • Sparse → Dense Reconstruction       │   │
│                     │ • PatchMatch Stereo (CUDA)            │   │
│                     │ • Stereo Fusion → fused.ply           │   │
│                     ├──────────────────────────────────────┤   │
│                     │ PyMeshLab Meshing                     │   │
│                     │ • Screened Poisson (depth=9)          │   │
│                     │ • Quadric Edge Collapse → 150K faces  │   │
│                     ├──────────────────────────────────────┤   │
│                     │ Trimesh → room.glb                    │   │
│                     └──────────────┬───────────────────────┘   │
│                                    │                            │
│  🌐 WEB VIEWER (Netlify)          ▼                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Three.js v0.160.0 Engine                                │   │
│  │ • GLTFLoader + DRACOLoader                              │   │
│  │ • Desktop: WASD + PointerLock mouse look                │   │
│  │ • Mobile: Virtual joystick + touch look                 │   │
│  │ • Raycast collision + floor snapping                    │   │
│  │ • No-Clip free-fly mode                                 │   │
│  │ • In-browser .glb hot-swap                              │   │
│  │ • Real-time FPS & coordinate telemetry                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
namma-space-round1/
├── index.html                                # Main Web App entry point
├── guide.html                                # Interactive visual guide (Mermaid + KaTeX math)
├── guide.md                                  # Markdown engineering guide
├── agents.md                                 # Autonomous AI agent operating manual & specs
├── css/
│   └── style.css                             # Dark-mode styling & mobile UI
├── js/
│   └── app.js                                # Three.js engine (~500 lines)
├── public/
│   └── models/
│       └── README.md                         # Drop room.glb here
├── pipeline/
│   ├── Namma_Space_Colab_Pipeline.ipynb      # 4-cell Colab T4 GPU Notebook
│   ├── colab_reconstruct.py                  # Headless Colab CLI execution script
│   ├── extract_frames.py                     # Local CLI frame extractor
│   └── requirements.txt                      # Python dependencies
├── test/
│   └── README.md                             # Test data instructions
└── README.md                                 # This file
```

---

## 🎬 Indoor Capture SOP (Standard Operating Procedure)

Follow this checklist before and during filming for optimal 3D reconstruction quality:

| # | Step | Why |
|---|------|-----|
| 1 | **Lock Auto-Exposure** (AE Lock) | Prevents flickering between frames |
| 2 | **Lock Auto-Focus** (AF Lock) at ~2m | Ensures consistent sharpness across all frames |
| 3 | **Disable EIS** (Electronic Image Stabilization) | EIS warps frame edges, degrading SfM matching |
| 4 | **Maximize lighting**: all lights ON, blinds closed | Consistent, shadow-free illumination |
| 5 | **Place 3–5 fiducial sticky notes** on blank walls | Gives SIFT features on textureless surfaces |
| 6 | **Record at 1080p, 30 FPS** | Balances quality vs. file size |
| 7 | **Perimeter crab-walk** at eye height (~1.6m) | Captures wall textures with high overlap |
| 8 | **Repeat perimeter** at waist height (~0.9m) | Covers floor details and furniture bases |
| 9 | **2–3 diagonal passes** through room center | Adds depth diversity for stereo reconstruction |
| 10 | **Maintain 60–70% overlap** between viewpoints | Essential for reliable SIFT feature matching |
| 11 | **Total: 2–5 minutes** per room, move slowly | Minimizes motion blur, maximizes sharp frames |

> 💡 **Tip**: The web viewer includes an interactive SOP checklist — click the **"Capture SOP"** button in the top-right HUD.

---

## ☁️ Google Colab GPU Pipeline

### Prerequisites
- Google account with access to [Google Colab](https://colab.research.google.com/)
- Set runtime to **T4 GPU**: `Runtime → Change runtime type → GPU → T4`

### Step-by-Step

1. **Upload the notebook**:
   - Open `pipeline/Namma_Space_Colab_Pipeline.ipynb` in Google Colab.

2. **Cell 1 — Environment Setup**:
   - Installs COLMAP, FFmpeg, OpenCV, PyMeshLab, and Trimesh.
   - Verifies GPU availability via `nvidia-smi`.
   - Prompts you to upload your walkthrough video (`.mp4`).

3. **Cell 2 — Frame Extraction**:
   - Samples video at 2 FPS.
   - Computes Laplacian variance for each frame as a sharpness metric.
   - Discards motion-blurred frames (score < 60).
   - Resizes to 1280px max width.
   - Saves sharp keyframes to `workspace/images/`.

4. **Cell 3 — COLMAP + Meshing**:
   - Runs GPU-accelerated COLMAP SfM pipeline:
     - SIFT feature extraction (GPU)
     - Sequential matching (GPU)
     - Sparse reconstruction (mapper)
     - Image undistortion
     - PatchMatch stereo (CUDA)
     - Stereo fusion → `fused.ply`
   - Screened Poisson surface reconstruction (depth=9)
   - Decimation to 150,000 faces for 60 FPS web rendering.

5. **Cell 4 — GLB Export**:
   - Converts vertex-colored mesh to binary `.glb` format.
   - Auto-downloads `room.glb` to your local machine.

6. **Load in viewer**: Either place `room.glb` in `public/models/` or use the in-browser **"Load .GLB Model"** button.

---

### ⚡ Alternative: Headless Execution via Google Colab CLI

You can also run the entire cloud reconstruction pipeline directly from your local terminal without opening a browser tab using the newly released **[Google Colab CLI](https://developers.googleblog.com/introducing-the-google-colab-cli/)**:

```bash
# 1. Provision a remote T4 GPU cloud session
colab new --gpu T4

# 2. Upload your indoor walkthrough video
colab upload test/sample_video.mp4 /content/sample_video.mp4

# 3. Execute the headless reconstruction pipeline on the remote GPU
colab exec -f pipeline/colab_reconstruct.py

# 4. Download the generated 3D model directly into the web viewer
colab download /content/room.glb public/models/room.glb

# 5. Terminate the session
colab stop
```

---

## 💻 Local Frame Extraction CLI

For local testing without Google Colab:

```bash
# Install dependencies
pip install -r pipeline/requirements.txt

# Extract sharp frames from a video
python pipeline/extract_frames.py --input test/sample_video.mp4 --output test/extracted_frames --threshold 60

# Options
python pipeline/extract_frames.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input, -i` | (required) | Input video file path |
| `--output, -o` | (required) | Output directory for frames |
| `--threshold, -t` | `60.0` | Laplacian variance threshold |
| `--fps, -f` | `2.0` | Sampling rate (frames per second) |
| `--max-width, -w` | `1280` | Maximum output image width (px) |

---

## 🌐 Web Viewer Features

### Desktop Controls

| Key | Action |
|-----|--------|
| **W / ↑** | Walk forward |
| **S / ↓** | Walk backward |
| **A / ←** | Strafe left |
| **D / →** | Strafe right |
| **Shift** | Sprint (5.5 m/s) |
| **Q / E** | Lower / raise eye height |
| **F** | Toggle No-Clip (free-fly) mode |
| **R** | Reset to spawn position |
| **Mouse** | Look around (after clicking to lock) |

### Mobile Controls (Auto-Detected)

| Control | Action |
|---------|--------|
| **Left Joystick** | Move forward/backward/strafe |
| **Touch-drag right half** | Look around (yaw + pitch) |
| **Free-Fly button** | Toggle collision / free-fly mode |
| **Load .GLB button** | Load model from phone storage |

### HUD Features

| Element | Description |
|---------|-------------|
| **FPS Counter** | Real-time frames per second (color-coded: green ≥50, yellow ≥30, red <30) |
| **Coordinates** | Live camera X, Y, Z position |
| **Copy Coords** | Copies position to clipboard (useful for POI logging in Round 2) |
| **Mode Indicator** | Shows COLLISION (green) or FREE-FLY (orange) |
| **Exposure Slider** | Adjust brightness for dark/bright photogrammetry scans |
| **Load .GLB Model** | Hot-swap any `.glb` file without redeployment |
| **Capture SOP** | Interactive filming checklist |

### Procedural Fallback Room

If no `room.glb` is present, the viewer automatically generates a stylish architectural demo room with:
- 20×20m floor with grid overlay
- Walls with doorway cutouts
- 4 decorative pillars
- Conference table with chairs
- Ceiling light fixtures with point lights
- Accent strip lighting

This ensures the app is **100% playable out-of-the-box** for judges.

---

## 🚀 Deployment

### Local Development (VS Code Live Server)

1. Open the repository in VS Code.
2. Install the **Live Server** extension.
3. Right-click `index.html` → **Open with Live Server**.
4. The viewer opens at `http://127.0.0.1:5500` — works instantly, no build step required.

### Netlify Production Deployment

* **Live URL**: **[https://namma-space.netlify.app/](https://namma-space.netlify.app/)**
* **Deployment Type**: Serverless Zero-Build Static Deployment (connected directly to GitHub repository for automatic CD on push).
* **Manual Redeploy (Alternative)**: Drag and drop the `namma-space-round1/` directory onto the [Netlify Dashboard](https://app.netlify.com).

---

## 🛠 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| 3D Engine | Three.js v0.160.0 (CDN) | WebGL rendering, GLTF loading |
| Mesh Decompression | DRACO (Google gstatic CDN) | Compressed GLB support |
| Camera Controls | PointerLockControls | Desktop FPS-style mouse look |
| Touch Controls | Custom vanilla JS | Mobile joystick + touch look |
| Frame Extraction | OpenCV (Python) | Laplacian blur filtering |
| 3D Reconstruction | COLMAP (GPU) | Structure-from-Motion + MVS |
| Surface Meshing | PyMeshLab | Screened Poisson + decimation |
| Format Conversion | Trimesh | PLY → GLB export |
| Frontend | Vanilla HTML5 + CSS3 + ES6 | Zero-build, zero-install |
| Hosting | Netlify | Static serverless deployment |
| GPU Compute | Google Colab (T4) | Free cloud GPU for reconstruction |

---

## 🎯 Performance Targets

| Metric | Target | How |
|--------|--------|-----|
| **60 FPS on Intel UHD** | ✅ | Pixel ratio capped at 1.5, mesh decimated to 150K faces |
| **60 FPS on Mobile** | ✅ | Efficient raycasting, ACES tone mapping, no heavy post-processing |
| **Instant Load** | ✅ | No npm/webpack build step, CDN-loaded Three.js |
| **Cross-Device** | ✅ | Auto-detects desktop vs. mobile, adapts controls |
| **Zero-Install** | ✅ | Works in any modern browser, hosted on Netlify |

---

## 🧪 Automated Verification & Quality Assurance Suite

Run the full 12-test automated verification suite across all engine, video, model, and pipeline subsystems:

```powershell
uv run --python 3.11 --with opencv-python --with numpy --with trimesh --with pillow python test/test_suite.py
```

- **Zero-Build Web Compliance**: Native ES6 importmap validation, zero bundlers, DOM elements.
- **Physics & Kinematics**: Walk speed ($3.0\text{ m/s}$), sprint ($5.5\text{ m/s}$), eye height ($1.6\text{m}$), and floor normal filtering.
- **Model Budget**: Mesh triangle face budget ($\le 150,000$ faces) and binary GLB integrity.
- **Video & Keyframe Dataset**: Real 1080p walkthrough validation and Laplacian sharpness ($\ge 40$).
- **Colab GPU Pipeline**: Jupyter notebook AST compilation and headless Python scripts.
- **Agent Operating Manual**: AI Agent skill manifests and operating rules.

---

## 📄 License

Built for Techfest IIT Bombay 2026-27 competition. Uses open-source libraries (Three.js, COLMAP, OpenCV, PyMeshLab, Trimesh) under their respective licenses.

---

<p align="center">
  <strong>Namma Space</strong> — Bridging Physical and Digital Worlds<br>
  <em>Techfest IIT Bombay 2026-27</em>
</p>
