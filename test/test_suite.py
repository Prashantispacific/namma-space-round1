#!/usr/bin/env python3
"""
Namma Space – Automated Verification & Quality Assurance Test Suite
===================================================================
Executes exhaustive end-to-end verification across:
1. Video Integrity & Sharp Keyframe Extraction
2. Zero-Build Web Viewer & Three.js Importmap Compliance
3. Kinematics, Collision, and Raycast Snapping Logic
4. 3D Model Performance Budgets (Triangles <= 150k, 60 FPS Target)
5. Google Colab GPU Reconstruction Notebook & Headless Scripts
6. Autonomous Agent Skills & Operating Manual Integrity
"""

import glob
import json
import os
import re
import subprocess
import sys
import unittest
from html.parser import HTMLParser

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestVideoIntegrity(unittest.TestCase):
    """Validates raw video capture and Laplacian blur-filtered keyframe dataset."""

    def test_sample_video_exists_and_valid(self):
        v_path = os.path.join(ROOT_DIR, "test", "sample_video.mp4")
        self.assertTrue(os.path.exists(v_path), f"Video missing at {v_path}")
        size_mb = os.path.getsize(v_path) / (1024 * 1024)
        self.assertGreater(size_mb, 10.0, f"Video size {size_mb:.1f} MB is abnormally small")

        try:
            import cv2
            cap = cv2.VideoCapture(v_path)
            self.assertTrue(cap.isOpened(), "Failed to open video with OpenCV")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frames / fps if fps > 0 else 0
            cap.release()

            self.assertGreaterEqual(duration, 30.0, f"Duration {duration:.1f}s is shorter than 30s budget")
            self.assertGreaterEqual(w, 1280, f"Width {w} is below 720p HD standard")
            self.assertGreaterEqual(h, 720, f"Height {h} is below 720p HD standard")
        except ImportError:
            self.skipTest("OpenCV not installed in current Python environment")

    def test_extracted_keyframes_dataset(self):
        frames_dir = os.path.join(ROOT_DIR, "test", "extracted_frames")
        self.assertTrue(os.path.exists(frames_dir), "test/extracted_frames directory missing")
        frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        self.assertGreaterEqual(len(frames), 50, f"Expected >= 50 extracted keyframes, found {len(frames)}")

        try:
            import cv2
            import numpy as np
            scores = []
            for f in frames[::5]:  # Sample every 5th frame for speed
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                self.assertIsNotNone(img, f"Failed to read image {f}")
                score = cv2.Laplacian(img, cv2.CV_64F).var()
                scores.append(score)
            mean_sharpness = np.mean(scores)
            self.assertGreaterEqual(mean_sharpness, 40.0, f"Mean keyframe sharpness {mean_sharpness:.1f} is too blurry")
        except ImportError:
            self.skipTest("OpenCV/NumPy not installed in current Python environment")


class TestZeroBuildWebArchitecture(unittest.TestCase):
    """Validates Zero-Build static frontend principles and Three.js runtime integration."""

    class HTMLValidator(HTMLParser):
        def __init__(self):
            super().__init__()
            self.errors = []
            self.ids = set()
            self.has_importmap = False
            self.has_bundler = False

        def handle_starttag(self, tag, attrs):
            attr_dict = dict(attrs)
            if "id" in attr_dict:
                self.ids.add(attr_dict["id"])
            if tag == "script" and attr_dict.get("type") == "importmap":
                self.has_importmap = True
            if tag == "script" and "src" in attr_dict:
                src = attr_dict["src"].lower()
                if any(b in src for b in ["webpack", "vite", "bundle", "parcel", "rollup"]):
                    self.has_bundler = True

    def test_index_html_zero_build_compliance(self):
        html_path = os.path.join(ROOT_DIR, "index.html")
        self.assertTrue(os.path.exists(html_path), "index.html missing")
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        validator = self.HTMLValidator()
        validator.feed(content)

        self.assertTrue(validator.has_importmap, "index.html must define native ES6 browser importmap")
        self.assertFalse(validator.has_bundler, "Zero-Build rule violated: bundler script found in index.html")

        # Required HUD and Telemetry IDs
        required_ids = [
            "viewport", "loading-overlay", "start-overlay", "start-enter-action",
            "hud-panel", "hud-fps", "hud-x", "hud-y", "hud-z", "hud-mode", "controls-hud"
        ]
        for rid in required_ids:
            self.assertIn(rid, validator.ids, f"Required DOM element #{rid} missing in index.html")

    def test_javascript_syntax_via_node(self):
        js_path = os.path.join(ROOT_DIR, "js", "app.js")
        self.assertTrue(os.path.exists(js_path), "js/app.js missing")
        res = subprocess.run(["node", "--check", js_path], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"JavaScript syntax error in js/app.js:\n{res.stderr}")

    def test_css_design_tokens(self):
        css_path = os.path.join(ROOT_DIR, "css", "style.css")
        self.assertTrue(os.path.exists(css_path), "css/style.css missing")
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()

        self.assertIn("#0a0a0c", css, "Dark mode root background token #0a0a0c missing")
        self.assertIn("--accent", css, "Accent CSS variable missing")
        self.assertIn("kbd", css, "Keyboard keycap styling rules missing")


class TestKinematicsAndCollisionBudgets(unittest.TestCase):
    """Validates game engine kinematics, collision thresholds, and normal filtering."""

    def test_kinematics_parameters_in_app_js(self):
        js_path = os.path.join(ROOT_DIR, "js", "app.js")
        with open(js_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Check budget constants
        self.assertRegex(code, r"WALK_SPEED\s*=\s*3\.0", "WALK_SPEED must be 3.0 m/s")
        self.assertRegex(code, r"SPRINT_SPEED\s*=\s*5\.5", "SPRINT_SPEED must be 5.5 m/s")
        self.assertRegex(code, r"EYE_HEIGHT\s*=\s*1\.6", "EYE_HEIGHT must be 1.6 m")
        self.assertRegex(code, r"COLLISION_DIST\s*=\s*0\.4", "COLLISION_DIST must be 0.4 m (player radius)")
        self.assertIn("Math.min(window.devicePixelRatio, 1.5)", code, "Pixel ratio clamping budget missing")

    def test_floor_raycast_normal_filtering(self):
        js_path = os.path.join(ROOT_DIR, "js", "app.js")
        with open(js_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Raycast normal check prevents snapping to ceilings
        self.assertIn("normal.y < 0.2", code, "Floor raycasting must filter downward-facing surface normals")


class Test3DModelBudget(unittest.TestCase):
    """Validates binary 3D mesh integrity against 150K face budget and geometry limits."""

    def test_room_glb_integrity(self):
        glb_path = os.path.join(ROOT_DIR, "public", "models", "room.glb")
        self.assertTrue(os.path.exists(glb_path), f"room.glb missing at {glb_path}")
        size_mb = os.path.getsize(glb_path) / (1024 * 1024)
        self.assertLessEqual(size_mb, 25.0, f"Model size {size_mb:.1f} MB exceeds 25 MB budget")

        try:
            import trimesh
            scene_or_mesh = trimesh.load(glb_path)
            if isinstance(scene_or_mesh, trimesh.Scene):
                total_faces = sum(len(g.faces) for g in scene_or_mesh.geometry.values())
            else:
                total_faces = len(scene_or_mesh.faces)

            # Strict Techfest budget: <= 150,000 faces
            self.assertLessEqual(total_faces, 150000, f"Polygon count {total_faces} exceeds 150k budget")
            self.assertGreater(total_faces, 10, "Model appears degenerate (< 10 faces)")
        except ImportError:
            self.skipTest("trimesh not installed in current Python environment")


class TestColabCloudPipeline(unittest.TestCase):
    """Validates Google Colab Jupyter Notebook and headless Python execution scripts."""

    def test_colab_notebook_json_and_ast_compilation(self):
        nb_path = os.path.join(ROOT_DIR, "pipeline", "Namma_Space_Colab_Pipeline.ipynb")
        self.assertTrue(os.path.exists(nb_path), "Notebook missing")

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        self.assertEqual(nb.get("nbformat"), 4, "Notebook must be nbformat version 4")
        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 3, "Expected at least 3 code cells in pipeline notebook")

        for idx, cell in enumerate(code_cells):
            source = "".join(cell.get("source", []))
            # Filter ipython magics for AST validation
            py_code = "\n".join(l for l in source.splitlines() if not l.strip().startswith(("%", "!")))
            try:
                compile(py_code, f"Cell_{idx}", "exec")
            except SyntaxError as e:
                self.fail(f"Syntax error in Notebook Cell {idx}: {e}")

    def test_pipeline_scripts_python_compilation(self):
        scripts = [
            os.path.join(ROOT_DIR, "pipeline", "extract_frames.py"),
            os.path.join(ROOT_DIR, "pipeline", "colab_reconstruct.py"),
            os.path.join(ROOT_DIR, "pipeline", "generate_room_mesh.py")
        ]
        for s in scripts:
            self.assertTrue(os.path.exists(s), f"Script {s} missing")
            res = subprocess.run([sys.executable, "-m", "py_compile", s], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Compilation failed for {s}:\n{res.stderr}")


class TestAgentSkillConfiguration(unittest.TestCase):
    """Validates Autonomous AI Agent skill manifests and operating manual."""

    def test_colab_operator_skill_manifest(self):
        skill_path = os.path.join(ROOT_DIR, ".agents", "skills", "colab-operator", "SKILL.md")
        self.assertTrue(os.path.exists(skill_path), f"Agent skill manifest missing at {skill_path}")
        with open(skill_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("name: colab-operator", content, "Skill name must be colab-operator")
        self.assertIn("Mental model", content, "Skill must include Mental Model section")
        self.assertIn("Authentication", content, "Skill must document Authentication strategy")
        self.assertIn("Safety", content, "Skill must include Safety teardown protocols")

    def test_agents_manual_exists(self):
        agents_path = os.path.join(ROOT_DIR, "agents.md")
        self.assertTrue(os.path.exists(agents_path), "agents.md missing at root")


def run_tests():
    print(f"{'='*70}")
    print("  NAMMA SPACE – COMPREHENSIVE VERIFICATION & TEST SUITE")
    print(f"{'='*70}")
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    run_tests()
