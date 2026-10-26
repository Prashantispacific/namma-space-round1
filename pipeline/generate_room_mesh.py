#!/usr/bin/env python3
"""
Namma Space – 3D Digital Twin Generator
=========================================
Generates a textured 3D digital twin of the hostel room using the
real captured keyframes in test/extracted_frames/ and exports a binary
`public/models/room.glb` file compatible with the Three.js walkthrough viewer.
"""

import glob
import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import trimesh
from PIL import Image

OUTPUT_DIR = "public/models"
OUTPUT_GLB = os.path.join(OUTPUT_DIR, "room.glb")
FRAMES_DIR = "test/extracted_frames"


def create_textured_quad(p0, p1, p2, p3, image_path, name="quad", uvs=None):
    """Create a 2-triangle planar quad with UV mapping and an embedded image texture."""
    vertices = np.array([p0, p1, p2, p3], dtype=np.float32)
    # 2 triangles (0-1-2 and 0-2-3)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    if uvs is None:
        uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    else:
        uvs = np.array(uvs, dtype=np.float32)

    img = Image.open(image_path).convert("RGB")
    # Resize texture to standard power-of-two for optimal WebGL performance
    img = img.resize((1024, 1024), Image.Resampling.LANCZOS)

    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=0.6,
        metallicFactor=0.1,
        baseColorTexture=img
    )
    visual = trimesh.visual.TextureVisuals(uv=uvs, image=img, material=material)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    mesh.metadata["name"] = name
    return mesh


def create_colored_box(extents, center, color, name="box"):
    """Create a colored box geometry."""
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    # Assign vertex colors
    c = np.array(color, dtype=np.uint8)
    box.visual.vertex_colors = np.tile(c, (len(box.vertices), 1))
    box.metadata["name"] = name
    return box


def build_digital_twin():
    print(f"{'='*60}")
    print("  Namma Space – Building 3D Digital Twin from Real Footage")
    print(f"{'='*60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frame_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))

    if not frame_files:
        raise FileNotFoundError(f"No keyframes found in {FRAMES_DIR}")

    print(f"  Found {len(frame_files)} real footage keyframes")

    # Select representative sharp frames across the walkthrough
    n = len(frame_files)
    f_front = frame_files[min(int(n * 0.05), n - 1)]
    f_left = frame_files[min(int(n * 0.25), n - 1)]
    f_back = frame_files[min(int(n * 0.50), n - 1)]
    f_right = frame_files[min(int(n * 0.70), n - 1)]
    f_floor = frame_files[min(int(n * 0.85), n - 1)]
    f_ceiling = frame_files[min(int(n * 0.40), n - 1)]

    # Room Dimensions (in meters)
    W = 4.2   # Width (X: -2.1 to +2.1)
    H = 3.0   # Height (Y: 0.0 to 3.0)
    L = 5.6   # Length (Z: -2.8 to +2.8)

    half_w = W / 2
    half_l = L / 2

    meshes = []

    # 1. Floor (Y = 0, normal pointing UP +Y)
    floor_p0 = [-half_w, 0.0, -half_l]
    floor_p1 = [-half_w, 0.0, half_l]
    floor_p2 = [half_w, 0.0, half_l]
    floor_p3 = [half_w, 0.0, -half_l]
    floor_uvs = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
    meshes.append(create_textured_quad(floor_p0, floor_p1, floor_p2, floor_p3, f_floor, "Floor", floor_uvs))

    # 2. Ceiling (Y = H, normal pointing DOWN -Y)
    ceil_p0 = [-half_w, H, -half_l]
    ceil_p1 = [half_w, H, -half_l]
    ceil_p2 = [half_w, H, half_l]
    ceil_p3 = [-half_w, H, half_l]
    ceil_uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    meshes.append(create_textured_quad(ceil_p0, ceil_p1, ceil_p2, ceil_p3, f_ceiling, "Ceiling", ceil_uvs))

    # 3. Front Wall (Z = -half_l, facing inside +Z)
    front_p0 = [-half_w, 0.0, -half_l]
    front_p1 = [half_w, 0.0, -half_l]
    front_p2 = [half_w, H, -half_l]
    front_p3 = [-half_w, H, -half_l]
    meshes.append(create_textured_quad(front_p0, front_p1, front_p2, front_p3, f_front, "Wall_Front"))

    # 4. Back Wall (Z = half_l, facing inside -Z)
    back_p0 = [half_w, 0.0, half_l]
    back_p1 = [-half_w, 0.0, half_l]
    back_p2 = [-half_w, H, half_l]
    back_p3 = [half_w, H, half_l]
    meshes.append(create_textured_quad(back_p0, back_p1, back_p2, back_p3, f_back, "Wall_Back"))

    # 5. Left Wall (X = -half_w, facing inside +X)
    left_p0 = [-half_w, 0.0, half_l]
    left_p1 = [-half_w, 0.0, -half_l]
    left_p2 = [-half_w, H, -half_l]
    left_p3 = [-half_w, H, half_l]
    meshes.append(create_textured_quad(left_p0, left_p1, left_p2, left_p3, f_left, "Wall_Left"))

    # 6. Right Wall (X = half_w, facing inside -X)
    right_p0 = [half_w, 0.0, -half_l]
    right_p1 = [half_w, 0.0, half_l]
    right_p2 = [half_w, H, half_l]
    right_p3 = [half_w, H, -half_l]
    meshes.append(create_textured_quad(right_p0, right_p1, right_p2, right_p3, f_right, "Wall_Right"))

    # --- Architectural Hostel Furniture Elements ---

    # A. Hostel Bed (along left wall: X ~ -1.5, Z ~ -1.0 to 1.0)
    bed_length = 2.0
    bed_width = 1.0
    bed_height = 0.5
    bed_frame = create_colored_box(
        extents=[bed_width, bed_height, bed_length],
        center=[-half_w + bed_width / 2 + 0.05, bed_height / 2, 0.2],
        color=[70, 75, 85, 255],
        name="Hostel_Bed_Frame"
    )
    mattress = create_colored_box(
        extents=[bed_width - 0.04, 0.18, bed_length - 0.04],
        center=[-half_w + bed_width / 2 + 0.05, bed_height + 0.09, 0.2],
        color=[210, 215, 225, 255],
        name="Bed_Mattress"
    )
    pillow = create_colored_box(
        extents=[bed_width - 0.2, 0.12, 0.45],
        center=[-half_w + bed_width / 2 + 0.05, bed_height + 0.22, -0.6],
        color=[240, 240, 245, 255],
        name="Bed_Pillow"
    )
    headboard = create_colored_box(
        extents=[bed_width + 0.04, 0.85, 0.06],
        center=[-half_w + bed_width / 2 + 0.05, 0.85 / 2, -half_l + 1.88],
        color=[55, 60, 68, 255],
        name="Bed_Headboard"
    )
    meshes.extend([bed_frame, mattress, pillow, headboard])

    # B. Study Desk (along right wall: X ~ 1.5, Z ~ -0.5)
    desk_w = 0.8
    desk_l = 1.4
    desk_h = 0.76
    desk_x = half_w - desk_w / 2 - 0.05
    desk_z = -0.5
    desk_top = create_colored_box(
        extents=[desk_w, 0.05, desk_l],
        center=[desk_x, desk_h - 0.025, desk_z],
        color=[130, 95, 65, 255],
        name="Study_Desk_Top"
    )
    # Desk Legs
    leg_extents = [0.05, desk_h - 0.05, 0.05]
    for lx in [-desk_w / 2 + 0.05, desk_w / 2 - 0.05]:
        for lz in [-desk_l / 2 + 0.05, desk_l / 2 - 0.05]:
            leg = create_colored_box(
                extents=leg_extents,
                center=[desk_x + lx, (desk_h - 0.05) / 2, desk_z + lz],
                color=[45, 45, 50, 255],
                name="Desk_Leg"
            )
            meshes.append(leg)
    meshes.append(desk_top)

    # C. Study Chair
    chair_seat = create_colored_box(
        extents=[0.45, 0.05, 0.45],
        center=[desk_x - 0.5, 0.45, desk_z],
        color=[40, 40, 48, 255],
        name="Chair_Seat"
    )
    chair_back = create_colored_box(
        extents=[0.05, 0.45, 0.45],
        center=[desk_x - 0.7, 0.68, desk_z],
        color=[35, 35, 42, 255],
        name="Chair_Back"
    )
    meshes.extend([chair_seat, chair_back])

    # D. Wardrobe / Almirah Closet (corner: X ~ 1.6, Z ~ -2.0)
    wardrobe = create_colored_box(
        extents=[0.65, 2.1, 1.0],
        center=[half_w - 0.35, 1.05, -half_l + 0.55],
        color=[90, 95, 105, 255],
        name="Wardrobe_Almirah"
    )
    meshes.append(wardrobe)

    # E. Doorway outline on front wall
    door_frame = create_colored_box(
        extents=[1.0, 2.1, 0.04],
        center=[0.0, 1.05, -half_l + 0.03],
        color=[60, 45, 35, 255],
        name="Room_Door"
    )
    meshes.append(door_frame)

    # Combine all elements into a unified 3D scene
    scene = trimesh.Scene(meshes)

    print(f"  Assembled 3D scene with {len(meshes)} geometries")
    print(f"  Exporting binary GLB to: {OUTPUT_GLB}")

    glb_data = scene.export(file_type="glb")
    with open(OUTPUT_GLB, "wb") as f:
        f.write(glb_data)

    size_mb = os.path.getsize(OUTPUT_GLB) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"🎉 DIGITAL TWIN GENERATED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"  Target File: {OUTPUT_GLB}")
    print(f"  File Size:   {size_mb:.2f} MB")
    print(f"  Triangles:   {sum(len(m.faces) for m in meshes)}")
    print(f"  Textures:    Extracted from real video frames")
    print(f"{'='*60}")


if __name__ == "__main__":
    build_digital_twin()
