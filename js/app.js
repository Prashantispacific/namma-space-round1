// ============================================================
//  Namma Space – 3D Web Walkthrough Engine
//  Techfest IIT Bombay 2026-27 | Round 1 Submission
//
//  Three.js v0.160.0 | Desktop WASD + Mobile Touch Controls
//  Raycast Collision | GLB Hot-Swap | Procedural Fallback Room
// ============================================================

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';

// ─── Constants ───────────────────────────────────────────────
const WALK_SPEED = 3.0;       // m/s
const SPRINT_SPEED = 5.5;     // m/s
const EYE_HEIGHT = 1.6;       // meters
const COLLISION_DIST = 0.4;   // meters (player radius)
const DAMPING = 0.85;
const PITCH_MIN = -1.396;     // -80 degrees
const PITCH_MAX = 1.396;      // +80 degrees
const TOUCH_SENSITIVITY = 0.003;
const JOYSTICK_RADIUS = 50;   // px
const FPS_SAMPLES = 60;
const MODEL_PATH = './public/models/room.glb';

// ─── State ───────────────────────────────────────────────────
let scene, camera, renderer, clock;
let pointerControls;
let collidableMeshes = [];
let currentModel = null;
let noClipMode = false;
let isMobile = false;
let isPointerLocked = false;

// Movement state
const velocity = new THREE.Vector3();
const direction = new THREE.Vector3();
const keys = {
    forward: false, backward: false,
    left: false, right: false,
    sprint: false, up: false, down: false
};

// Mobile state
let joystickActive = false;
let joystickCenter = { x: 0, y: 0 };
let joystickDelta = { x: 0, y: 0 };
let joystickTouchId = null;
let lookTouchId = null;
let lastLookTouch = { x: 0, y: 0 };

// Raycasters
const horizontalRay = new THREE.Raycaster();
const downRay = new THREE.Raycaster();
horizontalRay.far = COLLISION_DIST;
downRay.far = 50;

// FPS tracking
const fpsSamples = [];
let lastFrameTime = 0;

// Spawn position
const spawnPosition = new THREE.Vector3(0, EYE_HEIGHT, 5);

// ─── DOM Elements ────────────────────────────────────────────
const canvas = document.getElementById('viewport');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingBar = document.getElementById('loading-bar');
const loadingStatus = document.getElementById('loading-status');
const startOverlay = document.getElementById('start-overlay');
const hudFps = document.getElementById('hud-fps');
const hudX = document.getElementById('hud-x');
const hudY = document.getElementById('hud-y');
const hudZ = document.getElementById('hud-z');
const hudMode = document.getElementById('hud-mode');
const hudCopyBtn = document.getElementById('hud-copy-btn');
const btnPointerToggle = document.getElementById('btn-pointer-toggle');
const btnControls = document.getElementById('btn-controls');
const btnLoadGlb = document.getElementById('btn-load-glb');
const btnSop = document.getElementById('btn-sop');
const btnNoclip = document.getElementById('btn-noclip');
const mobileNoclipBtn = document.getElementById('mobile-noclip-btn');
const fileInput = document.getElementById('file-input');
const exposureSlider = document.getElementById('exposure-slider');
const exposureValue = document.getElementById('exposure-value');
const sopBackdrop = document.getElementById('sop-modal-backdrop');
const sopClose = document.getElementById('sop-modal-close');
const controlsBackdrop = document.getElementById('controls-modal-backdrop');
const controlsClose = document.getElementById('controls-modal-close');
const controlsHud = document.getElementById('controls-hud');
const btnToggleControlsHud = document.getElementById('btn-toggle-controls-hud');
const startEnterBtn = document.getElementById('start-enter-action');
const toastEl = document.getElementById('toast');
const joystickZone = document.getElementById('joystick-zone');
const joystickKnob = document.getElementById('joystick-knob');

// ─── Initialization ──────────────────────────────────────────
function init() {
    detectMobile();
    createScene();
    createLighting();
    setupLoaders();
    loadModel(MODEL_PATH);
    setupDesktopControls();
    if (isMobile) setupMobileControls();
    setupHUD();
    setupExposure();
    setupGlbHotSwap();
    setupSOPModal();
    window.addEventListener('resize', onResize);
    clock = new THREE.Clock();
    animate();
}

function detectMobile() {
    isMobile = ('ontouchstart' in window) || (navigator.maxTouchPoints > 0);
    if (isMobile) {
        document.body.classList.add('mobile-device');
    }
}

// ─── Scene, Camera, Renderer ─────────────────────────────────
function createScene() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0c);

    camera = new THREE.PerspectiveCamera(
        75, window.innerWidth / window.innerHeight, 0.1, 1000
    );
    camera.position.copy(spawnPosition);
    camera.rotation.order = 'YXZ';

    renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: true,
        powerPreference: 'high-performance'
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
}

function createLighting() {
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.4);
    scene.add(ambientLight);

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x222233, 0.8);
    hemiLight.position.set(0, 10, 0);
    scene.add(hemiLight);
}

// ─── Model Loading ───────────────────────────────────────────
let gltfLoader, dracoLoader;

function setupLoaders() {
    dracoLoader = new DRACOLoader();
    dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/');

    gltfLoader = new GLTFLoader();
    gltfLoader.setDRACOLoader(dracoLoader);
}

function loadModel(url, isBlob = false) {
    updateLoading(10, 'Loading 3D model…');

    gltfLoader.load(
        url,
        (gltf) => {
            // Remove previous model
            if (currentModel) {
                scene.remove(currentModel);
                collidableMeshes = [];
            }

            const model = gltf.scene;

            // Auto-center and align floor to y=0
            const box = new THREE.Box3().setFromObject(model);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());

            model.position.x -= center.x;
            model.position.y -= box.min.y;
            model.position.z -= center.z;

            // Collect collidable meshes & configure materials
            model.traverse((child) => {
                if (child.isMesh) {
                    collidableMeshes.push(child);
                    child.castShadow = false;
                    child.receiveShadow = false;

                    // Ensure valid normals for proper PBR lighting
                    if (child.geometry) {
                        if (!child.geometry.attributes.normal) {
                            child.geometry.computeVertexNormals();
                        }
                    }

                    const hasTexture = child.material && child.material.map;
                    const hasVertexColors = child.geometry &&
                        child.geometry.attributes.color &&
                        child.geometry.attributes.color.count > 0;

                    if (hasTexture) {
                        // Textured mesh (from photogrammetry baked textures or manual UV):
                        // preserve existing material, just ensure double-sided
                        child.material.side = THREE.DoubleSide;

                    } else if (hasVertexColors) {
                        // Photogrammetry vertex-colored mesh (COLMAP/Poisson output):
                        // PRESERVE vertex colors — this is the primary visual data!
                        child.material = new THREE.MeshStandardMaterial({
                            vertexColors: true,
                            roughness: 0.75,
                            metalness: 0.05,
                            side: THREE.DoubleSide
                        });

                    } else {
                        // Untextured, no vertex colors (procedural/furniture geometry):
                        // assign semantic PBR material based on mesh name
                        const name = (child.name || '').toLowerCase();
                        let color = 0x555d6e;
                        let roughness = 0.6;
                        let metalness = 0.2;

                        if (name.includes('bed') || name.includes('frame')) {
                            color = 0x3a4050;
                            metalness = 0.4;
                        } else if (name.includes('mattress') || name.includes('pillow')) {
                            color = 0xd0d6e2;
                            roughness = 0.9;
                            metalness = 0.05;
                        } else if (name.includes('desk') || name.includes('table')) {
                            color = 0x8a5834; // teak wood
                            roughness = 0.45;
                            metalness = 0.08;
                        } else if (name.includes('wardrobe') || name.includes('almirah')) {
                            color = 0x424a56;
                            metalness = 0.5;
                            roughness = 0.4;
                        } else if (name.includes('door')) {
                            color = 0x2c3340;
                        }

                        child.material = new THREE.MeshStandardMaterial({
                            color: color,
                            roughness: roughness,
                            metalness: metalness,
                            side: THREE.DoubleSide
                        });
                    }
                }
            });

            scene.add(model);
            currentModel = model;

            // Set spawn position based on model size (inside the room facing forward)
            spawnPosition.set(0, EYE_HEIGHT, Math.min(size.z * 0.35, 1.8));
            camera.position.copy(spawnPosition);
            camera.lookAt(0, EYE_HEIGHT, -1.0);

            updateLoading(100, 'Model loaded');
            showToast(`Model loaded: ${collidableMeshes.length} meshes`);
            const urlParams = new URLSearchParams(window.location.search);
            if (urlParams.get('hide_overlay') === '1') {
                hideLoading();
            } else {
                setTimeout(hideLoading, 600);
            }

            if (isBlob) URL.revokeObjectURL(url);
        },
        (progress) => {
            if (progress.total > 0) {
                const pct = Math.round((progress.loaded / progress.total) * 90) + 10;
                updateLoading(pct, `Loading… ${Math.round(progress.loaded / 1024)}KB`);
            }
        },
        (error) => {
            console.warn('Model load failed, building fallback room:', error.message);
            updateLoading(50, 'Building demo room…');
            buildFallbackRoom();
            setTimeout(hideLoading, 600);
        }
    );
}

function updateLoading(pct, status) {
    if (loadingBar) loadingBar.style.width = pct + '%';
    if (loadingStatus) loadingStatus.textContent = status;
}

function hideLoading() {
    if (loadingOverlay) loadingOverlay.classList.add('hidden');
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('hide_overlay') === '1' && startOverlay) {
        startOverlay.classList.add('hidden');
    }
}

// ─── Procedural Fallback Room ────────────────────────────────
function buildFallbackRoom() {
    if (currentModel) {
        scene.remove(currentModel);
    }
    collidableMeshes = [];
    const roomGroup = new THREE.Group();
    roomGroup.name = 'FallbackRoom';

    const roomW = 20, roomD = 20, roomH = 3.5;
    const wallThick = 0.15;

    // Materials
    const floorMat = new THREE.MeshStandardMaterial({ color: 0x1a1a1e, roughness: 0.8, metalness: 0.1 });
    const wallMat = new THREE.MeshStandardMaterial({ color: 0x22222a, roughness: 0.7, metalness: 0.05 });
    const ceilingMat = new THREE.MeshStandardMaterial({ color: 0x18181c, roughness: 0.9 });
    const pillarMat = new THREE.MeshStandardMaterial({ color: 0x2a2a32, roughness: 0.5, metalness: 0.3 });
    const tableMat = new THREE.MeshStandardMaterial({ color: 0x3a3a42, roughness: 0.4, metalness: 0.2 });
    const accentMat = new THREE.MeshStandardMaterial({ color: 0x4fc3f7, roughness: 0.3, metalness: 0.6, emissive: 0x4fc3f7, emissiveIntensity: 0.3 });

    // --- Floor with grid pattern ---
    const floorGeo = new THREE.PlaneGeometry(roomW, roomD);
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0;
    roomGroup.add(floor);
    collidableMeshes.push(floor);

    // Grid lines on floor
    const gridHelper = new THREE.GridHelper(roomW, 20, 0x2a2a2e, 0x1e1e22);
    gridHelper.position.y = 0.005;
    roomGroup.add(gridHelper);

    // --- Ceiling ---
    const ceilingGeo = new THREE.PlaneGeometry(roomW, roomD);
    const ceiling = new THREE.Mesh(ceilingGeo, ceilingMat);
    ceiling.rotation.x = Math.PI / 2;
    ceiling.position.y = roomH;
    roomGroup.add(ceiling);
    collidableMeshes.push(ceiling);

    // --- Walls (with doorway cutouts) ---
    // Back wall (no door)
    const backWall = createWall(roomW, roomH, wallThick);
    backWall.position.set(0, roomH / 2, -roomD / 2);
    roomGroup.add(backWall);
    collidableMeshes.push(backWall);

    // Front wall (with doorway)
    const frontLeft = createWall(roomW / 2 - 1, roomH, wallThick);
    frontLeft.position.set(-roomW / 4 - 0.5, roomH / 2, roomD / 2);
    roomGroup.add(frontLeft);
    collidableMeshes.push(frontLeft);

    const frontRight = createWall(roomW / 2 - 1, roomH, wallThick);
    frontRight.position.set(roomW / 4 + 0.5, roomH / 2, roomD / 2);
    roomGroup.add(frontRight);
    collidableMeshes.push(frontRight);

    const frontTop = createWall(2, roomH - 2.2, wallThick);
    frontTop.position.set(0, roomH - (roomH - 2.2) / 2, roomD / 2);
    roomGroup.add(frontTop);
    collidableMeshes.push(frontTop);

    // Left wall (with doorway)
    const leftFront = createWall(wallThick, roomH, roomD / 2 - 1);
    leftFront.position.set(-roomW / 2, roomH / 2, roomD / 4 + 0.5);
    roomGroup.add(leftFront);
    collidableMeshes.push(leftFront);

    const leftBack = createWall(wallThick, roomH, roomD / 2 - 1);
    leftBack.position.set(-roomW / 2, roomH / 2, -roomD / 4 - 0.5);
    roomGroup.add(leftBack);
    collidableMeshes.push(leftBack);

    const leftTop = createWall(wallThick, roomH - 2.2, 2);
    leftTop.position.set(-roomW / 2, roomH - (roomH - 2.2) / 2, 0);
    roomGroup.add(leftTop);
    collidableMeshes.push(leftTop);

    // Right wall (no door)
    const rightWall = createWall(wallThick, roomH, roomD);
    rightWall.position.set(roomW / 2, roomH / 2, 0);
    roomGroup.add(rightWall);
    collidableMeshes.push(rightWall);

    // --- Pillars ---
    const pillarGeo = new THREE.CylinderGeometry(0.2, 0.2, roomH, 12);
    const pillarPositions = [
        [-5, 0, -5], [5, 0, -5], [-5, 0, 5], [5, 0, 5]
    ];
    pillarPositions.forEach(([x, , z]) => {
        const pillar = new THREE.Mesh(pillarGeo, pillarMat);
        pillar.position.set(x, roomH / 2, z);
        roomGroup.add(pillar);
        collidableMeshes.push(pillar);
    });

    // --- Conference Table ---
    const tableTopGeo = new THREE.BoxGeometry(4, 0.08, 1.5);
    const tableTop = new THREE.Mesh(tableTopGeo, tableMat);
    tableTop.position.set(0, 0.78, -3);
    roomGroup.add(tableTop);
    collidableMeshes.push(tableTop);

    // Table legs
    const legGeo = new THREE.BoxGeometry(0.08, 0.74, 0.08);
    const legPositions = [
        [-1.8, 0.37, -3.6], [1.8, 0.37, -3.6],
        [-1.8, 0.37, -2.4], [1.8, 0.37, -2.4]
    ];
    legPositions.forEach(([x, y, z]) => {
        const leg = new THREE.Mesh(legGeo, tableMat);
        leg.position.set(x, y, z);
        roomGroup.add(leg);
        collidableMeshes.push(leg);
    });

    // --- Decorative accent strips on walls ---
    const stripGeo = new THREE.BoxGeometry(roomW - 0.5, 0.02, 0.02);
    const stripBack = new THREE.Mesh(stripGeo, accentMat);
    stripBack.position.set(0, 1.2, -roomD / 2 + 0.1);
    roomGroup.add(stripBack);

    // --- Ceiling lights ---
    const lightGeo = new THREE.BoxGeometry(2, 0.05, 0.3);
    const lightMat = new THREE.MeshStandardMaterial({
        color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.5, roughness: 0.1
    });
    [[-3, 0], [3, 0], [0, -4], [0, 4]].forEach(([x, z]) => {
        const light = new THREE.Mesh(lightGeo, lightMat);
        light.position.set(x, roomH - 0.03, z);
        roomGroup.add(light);

        const pointLight = new THREE.PointLight(0xffffff, 0.5, 8);
        pointLight.position.set(x, roomH - 0.3, z);
        roomGroup.add(pointLight);
    });

    // --- Small accent furniture (chairs around table) ---
    const chairSeatGeo = new THREE.BoxGeometry(0.5, 0.05, 0.5);
    const chairBackGeo = new THREE.BoxGeometry(0.5, 0.5, 0.05);
    const chairLegGeo = new THREE.BoxGeometry(0.04, 0.45, 0.04);
    const chairMat = new THREE.MeshStandardMaterial({ color: 0x2e2e36, roughness: 0.6 });

    const chairPositions = [
        { x: -1.2, z: -3, rot: 0 }, { x: 0, z: -3, rot: 0 }, { x: 1.2, z: -3, rot: 0 },
        { x: -1.2, z: -3, rot: Math.PI }, { x: 0, z: -3, rot: Math.PI }, { x: 1.2, z: -3, rot: Math.PI }
    ];
    chairPositions.forEach(({ x, z, rot }, i) => {
        const chairGroup = new THREE.Group();
        const seat = new THREE.Mesh(chairSeatGeo, chairMat);
        seat.position.y = 0.45;
        chairGroup.add(seat);
        collidableMeshes.push(seat);

        const back = new THREE.Mesh(chairBackGeo, chairMat);
        back.position.set(0, 0.72, -0.22);
        chairGroup.add(back);

        const offsetZ = (i < 3) ? -1.2 : 1.2;
        chairGroup.position.set(x, 0, z + offsetZ);
        chairGroup.rotation.y = rot;
        roomGroup.add(chairGroup);
    });

    scene.add(roomGroup);
    currentModel = roomGroup;

    // Reset spawn
    spawnPosition.set(0, EYE_HEIGHT, 7);
    camera.position.copy(spawnPosition);

    updateLoading(100, 'Demo room ready');
    showToast('Fallback demo room loaded — drop a .glb to replace');
}

function createWall(w, h, d) {
    const geo = new THREE.BoxGeometry(w, h, d);
    const mat = new THREE.MeshStandardMaterial({ color: 0x22222a, roughness: 0.7, metalness: 0.05 });
    return new THREE.Mesh(geo, mat);
}

// ─── Desktop Controls ────────────────────────────────────────
function setupDesktopControls() {
    pointerControls = new PointerLockControls(camera, canvas);
    document.body.classList.add('cursor-free');

    let hasEnteredOnce = false;
    let isDraggingLook = false;
    let dragStartX = 0, dragStartY = 0;
    let lastDragX = 0, lastDragY = 0;
    let hasDraggedSignificant = false;

    // Click canvas to lock pointer (desktop only)
    if (!isMobile) {
        startOverlay.addEventListener('click', (e) => {
            pointerControls.lock();
        });

        // Mouse drag-to-look when cursor is free (not pointer locked)
        canvas.addEventListener('mousedown', (e) => {
            if (isPointerLocked || e.button !== 0) return;
            isDraggingLook = true;
            dragStartX = e.clientX;
            dragStartY = e.clientY;
            lastDragX = e.clientX;
            lastDragY = e.clientY;
            hasDraggedSignificant = false;
        });

        window.addEventListener('mousemove', (e) => {
            if (!isDraggingLook || isPointerLocked) return;
            const dx = e.clientX - lastDragX;
            const dy = e.clientY - lastDragY;
            if (Math.abs(e.clientX - dragStartX) > 4 || Math.abs(e.clientY - dragStartY) > 4) {
                hasDraggedSignificant = true;
            }
            camera.rotation.y -= dx * 0.0028;
            camera.rotation.x -= dy * 0.0028;
            camera.rotation.x = Math.max(PITCH_MIN, Math.min(PITCH_MAX, camera.rotation.x));
            lastDragX = e.clientX;
            lastDragY = e.clientY;
        });

        window.addEventListener('mouseup', () => {
            isDraggingLook = false;
        });

        // Right-click canvas releases pointer lock immediately
        canvas.addEventListener('contextmenu', (e) => {
            if (isPointerLocked) {
                e.preventDefault();
                pointerControls.unlock();
            }
        });

        // Click canvas locks pointer only if user didn't drag
        canvas.addEventListener('click', () => {
            if (!isPointerLocked && !hasDraggedSignificant) {
                pointerControls.lock();
            }
            hasDraggedSignificant = false;
        });
    }

    pointerControls.addEventListener('lock', () => {
        isPointerLocked = true;
        hasEnteredOnce = true;
        startOverlay.classList.add('hidden');
        document.body.classList.remove('cursor-free');
        document.body.classList.add('pointer-locked');
        if (btnPointerToggle) {
            btnPointerToggle.textContent = '🖱️ Free Cursor [Esc]';
            btnPointerToggle.classList.add('active');
        }
        showToast('Mouse Look active · Press [Esc] or [M] to free cursor', 2200);
    });

    pointerControls.addEventListener('unlock', () => {
        isPointerLocked = false;
        document.body.classList.remove('pointer-locked');
        document.body.classList.add('cursor-free');
        if (btnPointerToggle) {
            btnPointerToggle.textContent = '🎯 Lock Look';
            btnPointerToggle.classList.remove('active');
        }
        // Only show startOverlay on the very first landing before entering
        if (!hasEnteredOnce && !isMobile) {
            startOverlay.classList.remove('hidden');
        } else {
            showToast('Cursor free · Click HUD buttons or Drag 3D view to look', 2600);
        }
    });

    // Keyboard controls
    document.addEventListener('keydown', (e) => {
        switch (e.code) {
            case 'KeyW': case 'ArrowUp':    keys.forward = true; break;
            case 'KeyS': case 'ArrowDown':  keys.backward = true; break;
            case 'KeyA': case 'ArrowLeft':  keys.left = true; break;
            case 'KeyD': case 'ArrowRight': keys.right = true; break;
            case 'ShiftLeft': case 'ShiftRight': keys.sprint = true; break;
            case 'Space':    if (noClipMode) keys.up = true; e.preventDefault(); break;
            case 'ControlLeft': case 'ControlRight': if (noClipMode) keys.down = true; break;
            case 'KeyQ':     camera.position.y -= 0.1; break;
            case 'KeyE':     camera.position.y += 0.1; break;
            case 'KeyR':     resetPosition(); break;
            case 'KeyF':     toggleNoClip(); break;
            case 'KeyH':     toggleControlsHud(); break;
            case 'KeyM':
                if (isPointerLocked) {
                    pointerControls.unlock();
                } else {
                    pointerControls.lock();
                }
                break;
        }
    });

    document.addEventListener('keyup', (e) => {
        switch (e.code) {
            case 'KeyW': case 'ArrowUp':    keys.forward = false; break;
            case 'KeyS': case 'ArrowDown':  keys.backward = false; break;
            case 'KeyA': case 'ArrowLeft':  keys.left = false; break;
            case 'KeyD': case 'ArrowRight': keys.right = false; break;
            case 'ShiftLeft': case 'ShiftRight': keys.sprint = false; break;
            case 'Space':    keys.up = false; break;
            case 'ControlLeft': case 'ControlRight': keys.down = false; break;
        }
    });
}

// ─── Mobile Touch Controls ───────────────────────────────────
function setupMobileControls() {
    // Hide desktop start overlay on mobile
    startOverlay.classList.add('hidden');

    const joystickBase = document.getElementById('joystick-base');

    // --- Left Joystick (Movement) ---
    joystickZone.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const touch = e.changedTouches[0];
        joystickTouchId = touch.identifier;
        joystickActive = true;

        const rect = joystickBase.getBoundingClientRect();
        joystickCenter.x = rect.left + rect.width / 2;
        joystickCenter.y = rect.top + rect.height / 2;
    }, { passive: false });

    joystickZone.addEventListener('touchmove', (e) => {
        e.preventDefault();
        for (const touch of e.changedTouches) {
            if (touch.identifier === joystickTouchId) {
                let dx = touch.clientX - joystickCenter.x;
                let dy = touch.clientY - joystickCenter.y;

                // Clamp to joystick radius
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist > JOYSTICK_RADIUS) {
                    dx = (dx / dist) * JOYSTICK_RADIUS;
                    dy = (dy / dist) * JOYSTICK_RADIUS;
                }

                joystickDelta.x = dx / JOYSTICK_RADIUS;  // -1 to 1
                joystickDelta.y = dy / JOYSTICK_RADIUS;   // -1 to 1

                // Move knob visually
                joystickKnob.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
            }
        }
    }, { passive: false });

    const resetJoystick = (e) => {
        for (const touch of e.changedTouches) {
            if (touch.identifier === joystickTouchId) {
                joystickActive = false;
                joystickDelta.x = 0;
                joystickDelta.y = 0;
                joystickTouchId = null;
                joystickKnob.style.transform = 'translate(-50%, -50%)';
            }
        }
    };

    joystickZone.addEventListener('touchend', resetJoystick, { passive: false });
    joystickZone.addEventListener('touchcancel', resetJoystick, { passive: false });

    // --- Right-half Touch Look (Camera Rotation) ---
    canvas.addEventListener('touchstart', (e) => {
        for (const touch of e.changedTouches) {
            // Only track touches on the right half of screen (and not on joystick)
            if (touch.clientX > window.innerWidth * 0.35 && lookTouchId === null) {
                lookTouchId = touch.identifier;
                lastLookTouch.x = touch.clientX;
                lastLookTouch.y = touch.clientY;
            }
        }
    }, { passive: true });

    canvas.addEventListener('touchmove', (e) => {
        for (const touch of e.changedTouches) {
            if (touch.identifier === lookTouchId) {
                const dx = touch.clientX - lastLookTouch.x;
                const dy = touch.clientY - lastLookTouch.y;

                camera.rotation.y -= dx * TOUCH_SENSITIVITY;
                camera.rotation.x -= dy * TOUCH_SENSITIVITY;
                camera.rotation.x = Math.max(PITCH_MIN, Math.min(PITCH_MAX, camera.rotation.x));

                lastLookTouch.x = touch.clientX;
                lastLookTouch.y = touch.clientY;
            }
        }
    }, { passive: true });

    const resetLook = (e) => {
        for (const touch of e.changedTouches) {
            if (touch.identifier === lookTouchId) {
                lookTouchId = null;
            }
        }
    };

    canvas.addEventListener('touchend', resetLook, { passive: true });
    canvas.addEventListener('touchcancel', resetLook, { passive: true });
}

// ─── Collision System ────────────────────────────────────────
function checkWallCollision(pos, moveDir) {
    if (noClipMode || collidableMeshes.length === 0) return false;

    horizontalRay.set(pos, moveDir);
    const hits = horizontalRay.intersectObjects(collidableMeshes, true);
    return hits.length > 0 && hits[0].distance < COLLISION_DIST;
}

function getFloorHeight(pos) {
    if (noClipMode || collidableMeshes.length === 0) return null;

    // Cast downward from player's eye level towards feet
    const origin = new THREE.Vector3(pos.x, pos.y + 0.2, pos.z);
    downRay.set(origin, new THREE.Vector3(0, -1, 0));
    const hits = downRay.intersectObjects(collidableMeshes, true);

    for (let i = 0; i < hits.length; i++) {
        const hit = hits[i];
        // Ignore ceiling or downward-facing surface normals
        if (hit.object.name && hit.object.name.toLowerCase().includes('ceiling')) continue;
        if (hit.face && hit.face.normal.y < 0.2) continue;
        return hit.point.y + EYE_HEIGHT;
    }
    return null;
}

// ─── Movement Update ─────────────────────────────────────────
function updateMovement(dt) {
    const speed = keys.sprint ? SPRINT_SPEED : WALK_SPEED;

    // Get movement input
    let inputX = 0, inputZ = 0, inputY = 0;

    if (isMobile && joystickActive) {
        inputX = joystickDelta.x;
        inputZ = joystickDelta.y;
    } else {
        if (keys.forward) inputZ -= 1;
        if (keys.backward) inputZ += 1;
        if (keys.left) inputX -= 1;
        if (keys.right) inputX += 1;
    }

    if (noClipMode) {
        if (keys.up) inputY += 1;
        if (keys.down) inputY -= 1;
    }

    // Compute world-space movement direction from camera orientation
    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);

    if (!noClipMode) {
        forward.y = 0;
        forward.normalize();
    }

    const right = new THREE.Vector3();
    right.crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();

    // Target velocity
    const targetVelocity = new THREE.Vector3();
    targetVelocity.addScaledVector(right, inputX * speed);
    targetVelocity.addScaledVector(forward, -inputZ * speed);
    if (noClipMode) targetVelocity.y += inputY * speed;

    // Apply damping
    velocity.lerp(targetVelocity, 1 - DAMPING);

    // Apply movement with collision detection
    const movement = velocity.clone().multiplyScalar(dt);

    if (!noClipMode && collidableMeshes.length > 0) {
        const pos = camera.position.clone();

        // Check X movement
        if (Math.abs(movement.x) > 0.001) {
            const dirX = new THREE.Vector3(Math.sign(movement.x), 0, 0);
            if (checkWallCollision(pos, dirX)) {
                movement.x = 0;
                velocity.x = 0;
            }
        }

        // Check Z movement
        if (Math.abs(movement.z) > 0.001) {
            const dirZ = new THREE.Vector3(0, 0, Math.sign(movement.z));
            if (checkWallCollision(pos, dirZ)) {
                movement.z = 0;
                velocity.z = 0;
            }
        }
    }

    camera.position.add(movement);

    // Floor snapping
    if (!noClipMode) {
        const floorY = getFloorHeight(camera.position);
        if (floorY !== null) {
            camera.position.y += (floorY - camera.position.y) * 0.2;
        }
    }
}

// ─── HUD ─────────────────────────────────────────────────────
function setupHUD() {
    hudCopyBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const coords = `${camera.position.x.toFixed(2)}, ${camera.position.y.toFixed(2)}, ${camera.position.z.toFixed(2)}`;
        navigator.clipboard.writeText(coords).then(() => {
            showToast(`Coords copied: ${coords}`);
        }).catch(() => {
            showToast(`Coords: ${coords}`);
        });
    });

    btnNoclip.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleNoClip();
    });

    mobileNoclipBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleNoClip();
    });

    if (btnPointerToggle) {
        btnPointerToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            if (isPointerLocked) {
                pointerControls.unlock();
            } else {
                pointerControls.lock();
            }
        });
    }

    if (btnControls) {
        btnControls.addEventListener('click', (e) => {
            e.stopPropagation();
            if (controlsBackdrop) controlsBackdrop.classList.add('visible');
            if (isPointerLocked) pointerControls.unlock();
        });
    }

    if (controlsClose) {
        controlsClose.addEventListener('click', (e) => {
            e.stopPropagation();
            if (controlsBackdrop) controlsBackdrop.classList.remove('visible');
        });
    }

    if (controlsBackdrop) {
        controlsBackdrop.addEventListener('click', (e) => {
            if (e.target === controlsBackdrop) {
                controlsBackdrop.classList.remove('visible');
            }
        });
    }

    if (btnToggleControlsHud) {
        btnToggleControlsHud.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleControlsHud();
        });
    }

    if (startEnterBtn) {
        startEnterBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (!isPointerLocked) {
                pointerControls.lock();
            }
        });
    }
}

function toggleControlsHud() {
    if (controlsHud) {
        controlsHud.classList.toggle('minimized');
        const isMin = controlsHud.classList.contains('minimized');
        showToast(isMin ? 'Bottom HUD minimized (Press H to restore)' : 'Bottom HUD visible');
    }
}

function updateHUD(dt) {
    // FPS
    fpsSamples.push(dt);
    if (fpsSamples.length > FPS_SAMPLES) fpsSamples.shift();
    const avgDt = fpsSamples.reduce((a, b) => a + b, 0) / fpsSamples.length;
    const fps = Math.round(1 / avgDt);
    hudFps.textContent = fps;
    hudFps.style.color = fps >= 50 ? '#66bb6a' : fps >= 30 ? '#ffa726' : '#ef5350';

    // Coordinates
    hudX.textContent = camera.position.x.toFixed(2);
    hudY.textContent = camera.position.y.toFixed(2);
    hudZ.textContent = camera.position.z.toFixed(2);

    // Mode
    if (noClipMode) {
        hudMode.textContent = 'FREE-FLY';
        hudMode.className = 'hud-value collision-off';
    } else {
        hudMode.textContent = 'COLLISION';
        hudMode.className = 'hud-value collision-on';
    }
}

// ─── Utilities ───────────────────────────────────────────────
function toggleNoClip() {
    noClipMode = !noClipMode;
    btnNoclip.textContent = noClipMode ? '👻 No-Clip: ON' : '👻 No-Clip: OFF';
    btnNoclip.classList.toggle('active', noClipMode);
    mobileNoclipBtn.classList.toggle('active', noClipMode);
    showToast(noClipMode ? 'Free-Fly mode enabled' : 'Collision mode enabled');
}

function resetPosition() {
    camera.position.copy(spawnPosition);
    camera.rotation.set(0, 0, 0);
    velocity.set(0, 0, 0);
    showToast('Position reset');
}

function showToast(message, duration = 2500) {
    toastEl.textContent = message;
    toastEl.classList.add('visible');
    clearTimeout(toastEl._timeout);
    toastEl._timeout = setTimeout(() => {
        toastEl.classList.remove('visible');
    }, duration);
}

// ─── Exposure Control ────────────────────────────────────────
function setupExposure() {
    exposureSlider.addEventListener('input', (e) => {
        e.stopPropagation();
        const val = parseFloat(e.target.value);
        renderer.toneMappingExposure = val;
        exposureValue.textContent = val.toFixed(2);
    });

    // Prevent pointer lock when interacting with slider
    exposureSlider.addEventListener('click', (e) => e.stopPropagation());
}

// ─── GLB Hot-Swap ────────────────────────────────────────────
function setupGlbHotSwap() {
    btnLoadGlb.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        showToast(`Loading: ${file.name}`);
        const url = URL.createObjectURL(file);

        // Show loading overlay
        loadingOverlay.classList.remove('hidden');
        updateLoading(5, `Loading ${file.name}…`);

        loadModel(url, true);

        // Reset file input so the same file can be re-loaded
        fileInput.value = '';
    });
}

// ─── SOP Modal ───────────────────────────────────────────────
function setupSOPModal() {
    btnSop.addEventListener('click', (e) => {
        e.stopPropagation();
        sopBackdrop.classList.add('visible');
        if (isPointerLocked) pointerControls.unlock();
    });

    sopClose.addEventListener('click', (e) => {
        e.stopPropagation();
        sopBackdrop.classList.remove('visible');
    });

    sopBackdrop.addEventListener('click', (e) => {
        if (e.target === sopBackdrop) {
            sopBackdrop.classList.remove('visible');
        }
    });
}

// ─── Resize Handler ──────────────────────────────────────────
function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// ─── Animation Loop ──────────────────────────────────────────
function animate() {
    requestAnimationFrame(animate);

    const dt = clock.getDelta();
    const clampedDt = Math.min(dt, 0.1); // Prevent large jumps

    updateMovement(clampedDt);
    updateHUD(clampedDt > 0 ? clampedDt : 0.016);

    renderer.render(scene, camera);
}

// ─── Start ───────────────────────────────────────────────────
init();
