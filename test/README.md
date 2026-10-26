# 🧪 Test Suite & Real Capture Benchmark Dataset
### Namma Space | Techfest, IIT Bombay 2026-27

This directory houses the continuous verification test suite and real captured benchmark walkthrough footage.

---

## 📂 Directory Layout

```
test/
├── sample_video.mp4        # Active benchmark walkthrough (141.7s, 1080p HEVC, 4,251 frames)
├── extracted_frames/       # 249 sharp keyframes (threshold 45.0, 70.1% pass rate)
├── benchmark_render.png    # Pixel-perfect headless render proof from Three.js engine
├── test_suite.py           # Exhaustive 12-test automated verification suite
└── README.md               # This verification specification
```

---

## 🚀 Running the Automated Test Suite

Run the full end-to-end verification suite:

```powershell
# Using uv (isolated dependencies)
uv run --python 3.11 --with opencv-python --with numpy --with trimesh --with pillow python test/test_suite.py

# Using standard Python
python test/test_suite.py
```

### Test Coverage (12 Automated Checks):
1. **`TestVideoIntegrity`**: Validates resolution ($\ge 1080\text{p}$), duration ($\ge 30\text{s}$), and keyframe sharpness ($\ge 40$ Laplacian variance).
2. **`TestZeroBuildWebArchitecture`**: Verifies native ES6 `importmap`, zero bundlers, valid `#viewport` and HUD IDs, and CSS design tokens.
3. **`TestKinematicsAndCollisionBudgets`**: Verifies velocity damping ($3.0\text{ m/s}$ walk, $5.5\text{ m/s}$ sprint), player radius ($0.4\text{m}$), and floor raycast normal filtering (`normal.y < 0.2`).
4. **`Test3DModelBudget`**: Verifies binary GLB polygon triangle count stays strictly below **150,000 faces** and file size $\le 25\text{ MB}$.
5. **`TestColabCloudPipeline`**: Validates Jupyter Notebook AST compilation and Python script syntax.
6. **`TestAgentSkillConfiguration`**: Validates Autonomous AI Agent skill manifests (`.agents/skills/colab-operator/SKILL.md`) and operating manual (`agents.md`).

---

## 📹 Video Capture Benchmark Comparison

| Metric | Original Capture (`030116.mp4`) | New High-Density Capture (`044019.mp4`) | Improvement |
|---|---|---|---|
| **Walkthrough Duration** | 69.8s (2,093 frames) | **141.7s (4,251 frames)** | **+103% (2.0x longer)** |
| **Mean Sharpness (Laplacian)** | 40.6 variance | **100.7 variance** | **+148% (2.5x sharper)** |
| **Median Sharpness** | 35.3 variance | **81.5 variance** | **+131% (2.3x sharper)** |
| **Max Frame Sharpness** | 204.3 variance | **558.7 variance** | **+173% (2.7x sharper)** |
| **Sharp Keyframes Extracted** | 81 frames | **249 sharp keyframes** | **+207% (3.1x density)** |
| **Pass Rate (Threshold 45.0)**| 38.6% | **70.1% pass rate** | **Sustained optical clarity** |

---

## 🛠️ Re-Extracting Keyframes

```powershell
uv run --python 3.11 --with opencv-python --with numpy python pipeline/extract_frames.py --input test/sample_video.mp4 --output test/extracted_frames --threshold 45.0 --fps 2.5
```

