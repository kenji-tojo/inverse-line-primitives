// web/main.js - three.js + WebGL2 viewer for baked line bundles.
//
// Renders a 1-2 M vertex line-strip soup (one mega-draw via WebGL2 primitive
// restart at 0xFFFFFFFF).  The bundle's SH is evaluated per vertex against the
// world-space view direction d = normalize(pos - cameraPosition), matching the
// convention used in training and in the desktop viewer.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);

const els = {
  canvas: $("canvas"),
  sceneSel: $("scene"),
  stats: $("stats"),
  err: $("err"),
};

function fail(msg) {
  els.err.textContent = msg;
  els.err.style.display = "block";
  console.error(msg);
}

// -- Binary bundle parser ----------------------------------------------------
// Header layout (all little-endian, 56 bytes total before buffers):
//   u32 tag            0x454E494C, the four bytes "LINE"
//   u32 version
//   u32 n_verts
//   u32 n_idx
//   u32 n_strips
//   u32 sh_max
//   f32 opacity_thresh
//   u32 pad
//   f32[6] bbox_min(3), bbox_max(3)
// Then: positions[n_verts*3] f32, sh[n_verts*(sh_max+1)^2*3] f32,
//       indices[n_idx] u32.
const LINE_BINARY = 0x454e494c;   // the four bytes "LINE"

async function loadBundle(url, onProgress) {
  const resp = await fetch(url, { cache: "force-cache" });
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
  // Streamed so the loading percentage can be reported for big files.
  const total = Number(resp.headers.get("content-length") || 0);
  const reader = resp.body.getReader();
  const chunks = []; let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); received += value.byteLength;
    if (onProgress) onProgress(received, total);
  }
  const buf = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.byteLength; }

  const dv = new DataView(buf.buffer);
  if (dv.getUint32(0, true) !== LINE_BINARY)
    throw new Error(`${url} is not a line bundle (unexpected file header)`);
  const version  = dv.getUint32(4, true);
  if (version !== 1) throw new Error(`unsupported version ${version}`);
  const nVerts   = dv.getUint32(8, true);
  const nIdx     = dv.getUint32(12, true);
  const nStrips  = dv.getUint32(16, true);
  const shMax    = dv.getUint32(20, true);
  const opacity  = dv.getFloat32(24, true);
  // bytes 28..31 are padding, keeping the bbox 4-byte aligned
  const bbox = new Float32Array(buf.buffer, 32, 6);
  const bboxMin = new THREE.Vector3(bbox[0], bbox[1], bbox[2]);
  const bboxMax = new THREE.Vector3(bbox[3], bbox[4], bbox[5]);

  const shCoeffs = (shMax + 1) ** 2;   // SH coefficients per colour channel
  let cursor = 56;   // 8 + 24 + 24 = 56-byte header
  const positions = new Float32Array(buf.buffer, cursor, nVerts * 3);
  cursor += positions.byteLength;
  const sh = new Float32Array(buf.buffer, cursor, nVerts * shCoeffs * 3);
  cursor += sh.byteLength;
  const indices = new Uint32Array(buf.buffer, cursor, nIdx);

  return {
    nVerts, nIdx, nStrips, shMax, opacity,
    bboxMin, bboxMax, positions, sh, indices, shCoeffs,
  };
}

// -- Query-string flags ------------------------------------------------------
// ?dpr=<n>       - clamp window.devicePixelRatio (default 2).  Try ?dpr=1
//                  on phones for a big fill-rate win.
// ?aa=0          - disable canvas hardware MSAA.
// ?ui=0          - hide every overlay, including the scene picker.  Intended
//                  for screen capture; pair it with ?scene= to choose what
//                  loads, since the picker is unavailable.
// ?scene=<file>  - load this bundle from data/ instead of the manifest's
//                  first entry.
const query = new URLSearchParams(window.location.search);
const DPR_CAP    = Math.max(0.5, parseFloat(query.get("dpr") || "2"));
const USE_AA     = query.get("aa") !== "0";
const HIDE_UI    = query.get("ui")  === "0";
const SCENE_FILE = query.get("scene") || "";

if (HIDE_UI) {
  for (const id of ["ui", "stats"]) {
    const el = document.getElementById(id);
    if (el) el.style.display = "none";
  }
}

// -- Shaders -----------------------------------------------------------------
// shCoeffs is (sh_max + 1)^2: 1 coefficient per channel for a DC-only bundle,
// 4 for degree 1, up to 16 for degree 3.  One shader is compiled per count so
// the vertex shader evaluates only the basis functions the bundle carries.
// Same basis and sign convention as the desktop viewer's SH compute shader
// (external/fuzzydr/viewer/shaders/viewer_sh_eval.comp).
function buildMaterial(shCoeffs) {
  const shAttrs = [];
  for (let k = 0; k < shCoeffs; k++) shAttrs.push(`in vec3 sh${k};`);

  // Basis functions, in the order the bundle stores its coefficients.
  const basisLines = [
    "b0  =  0.282095;",
    "b1  = -0.488603 * d.y;",
    "b2  =  0.488603 * d.z;",
    "b3  = -0.488603 * d.x;",
    "b4  =  1.092548 * d.x * d.y;",
    "b5  = -1.092548 * d.y * d.z;",
    "b6  =  0.315392 * (2.0*d.z*d.z - d.x*d.x - d.y*d.y);",
    "b7  = -1.092548 * d.x * d.z;",
    "b8  =  0.546274 * (d.x*d.x - d.y*d.y);",
    "b9  = -0.590044 * d.y * (3.0*d.x*d.x - d.y*d.y);",
    "b10 =  2.890611 * d.x * d.y * d.z;",
    "b11 = -0.457046 * d.y * (4.0*d.z*d.z - d.x*d.x - d.y*d.y);",
    "b12 =  0.373176 * d.z * (2.0*d.z*d.z - 3.0*d.x*d.x - 3.0*d.y*d.y);",
    "b13 = -0.457046 * d.x * (4.0*d.z*d.z - d.x*d.x - d.y*d.y);",
    "b14 =  1.445306 * d.z * (d.x*d.x - d.y*d.y);",
    "b15 = -0.590044 * d.x * (d.x*d.x - 3.0*d.y*d.y);",
  ];
  const basisDecls = Array.from({ length: shCoeffs }, (_, k) => `float b${k};`).join(" ");
  const basisAssign = basisLines.slice(0, shCoeffs).join("\n  ");
  const accum = Array.from({ length: shCoeffs }, (_, k) => `sh${k} * b${k}`).join(" + ");

  // NB: do NOT prepend `#version 300 es` here - three.js injects it (and a
  // SHADER_TYPE / SHADER_NAME preamble) when glslVersion: THREE.GLSL3.
  const vertex = `precision highp float;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 uCameraPos;
in vec3 position;
${shAttrs.join("\n")}
out vec3 vColor;
void main() {
  vec3 d = normalize(position - uCameraPos);
  ${basisDecls}
  ${basisAssign}
  vec3 acc = ${accum};
  vColor = max(0.5 + acc, vec3(0.0));
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

  const fragment = `precision mediump float;
in vec3 vColor;
out vec4 fragColor;
void main() { fragColor = vec4(vColor, 1.0); }`;

  return new THREE.RawShaderMaterial({
    vertexShader: vertex,
    fragmentShader: fragment,
    glslVersion: THREE.GLSL3,
    uniforms: { uCameraPos: { value: new THREE.Vector3() } },
    transparent: false,
    depthTest: true,
    depthWrite: true,
  });
}

// -- Build a THREE.Line from a parsed bundle ---------------------------------
function buildLineObject(bundle) {
  const geom = new THREE.BufferGeometry();

  // Position attribute (own VBO).
  geom.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(bundle.positions, 3),
  );

  // One vec3 attribute per SH coefficient, all views into one interleaved VBO.
  const stride = bundle.shCoeffs * 3;   // floats per vertex
  const interleaved = new THREE.InterleavedBuffer(bundle.sh, stride);
  for (let k = 0; k < bundle.shCoeffs; k++) {
    geom.setAttribute(
      `sh${k}`,
      new THREE.InterleavedBufferAttribute(interleaved, 3, k * 3, false),
    );
  }

  // Indexed line-strip with primitive-restart sentinels.
  const idxAttr = new THREE.Uint32BufferAttribute(bundle.indices, 1);
  geom.setIndex(idxAttr);

  // Bypass three.js's bounding-sphere computation: it walks indices linearly
  // and would treat 0xFFFFFFFF as a real vertex index.  The header's bbox
  // gives the same answer directly.
  const center = new THREE.Vector3()
    .addVectors(bundle.bboxMin, bundle.bboxMax).multiplyScalar(0.5);
  const radius = bundle.bboxMin.distanceTo(bundle.bboxMax) * 0.5;
  geom.boundingSphere = new THREE.Sphere(center, radius);
  geom.boundingBox    = new THREE.Box3(bundle.bboxMin.clone(), bundle.bboxMax.clone());

  const mat = buildMaterial(bundle.shCoeffs);
  const line = new THREE.Line(geom, mat);
  line.frustumCulled = false;   // bbox + line-strip can be brittle; just draw.
  return line;
}

// -- Scene management --------------------------------------------------------
const renderer = new THREE.WebGLRenderer({
  canvas: els.canvas,
  antialias: USE_AA,
  powerPreference: "high-performance",
  alpha: false,
  premultipliedAlpha: false,
});
renderer.setClearColor(0x000000, 1.0);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, DPR_CAP));
console.log(`[lines] dpr cap=${DPR_CAP}  effective=${renderer.getPixelRatio()}  aa=${USE_AA}`);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  50, window.innerWidth / window.innerHeight, 0.001, 100.0,
);
// Z-up, matching the training convention.
camera.up.set(0, 0, 1);
camera.position.set(0.0, 1.0, 0.0);

const controls = new OrbitControls(camera, els.canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.rotateSpeed = 0.6;
controls.zoomSpeed = 1.0;
controls.panSpeed = 0.8;
controls.minDistance = 0.01;
controls.maxDistance = 50.0;
controls.target.set(0, 0, 0);

let currentLine = null;
let currentBundle = null;

function disposeCurrent() {
  if (!currentLine) return;
  scene.remove(currentLine);
  currentLine.geometry.dispose();
  currentLine.material.dispose();
  currentLine = null;
  currentBundle = null;
}

function frameToBundle(b) {
  controls.target.copy(
    new THREE.Vector3().addVectors(b.bboxMin, b.bboxMax).multiplyScalar(0.5),
  );
  const diag = b.bboxMin.distanceTo(b.bboxMax);
  const dist = diag * 1.4;
  // Pull camera off the +Y axis from the centre (arbitrary; OrbitControls
  // will let the user reorient immediately).
  const center = controls.target.clone();
  camera.position.set(center.x, center.y + dist, center.z);
  camera.near = Math.max(diag * 0.0005, 1e-4);
  camera.far  = Math.max(diag * 20.0, 10.0);
  camera.updateProjectionMatrix();
  controls.update();
}

async function loadScene(url, label) {
  els.stats.textContent = "loading...";
  disposeCurrent();
  let bundle;
  try {
    bundle = await loadBundle(url, (recv, total) => {
      const pct = total ? Math.round((recv / total) * 100) : null;
      els.stats.textContent = pct !== null ? `${pct}%` : "loading...";
    });
  } catch (e) {
    fail(`load failed: ${e.message}`);
    return;
  }
  const line = buildLineObject(bundle);
  scene.add(line);
  currentLine = line;
  currentBundle = bundle;
  frameToBundle(bundle);
  refreshStats();

  console.log(
    `[lines] loaded ${label}\n`
    + `  verts=${bundle.nVerts}  strips=${bundle.nStrips}  idx=${bundle.nIdx}  sh<=${bundle.shMax}\n`
    + `  bbox  min=${bundle.bboxMin.toArray().map(v => v.toFixed(3))}  `
    +        `max=${bundle.bboxMax.toArray().map(v => v.toFixed(3))}\n`
    + `  cam   pos=${camera.position.toArray().map(v => v.toFixed(3))}  `
    +        `target=${controls.target.toArray().map(v => v.toFixed(3))}\n`
    + `  near=${camera.near}  far=${camera.far}`
  );
}

// -- FPS overlay (only thing visible while viewing) --------------------------
let frameCount = 0, fpsAccum = 0, lastFpsT = performance.now(), fps = 0;
function refreshStats() {
  els.stats.textContent = currentBundle ? `${fps.toFixed(0)} fps` : "loading...";
}
function tickFps(now) {
  frameCount++;
  fpsAccum += 1;
  if (now - lastFpsT >= 500) {
    fps = (fpsAccum * 1000) / (now - lastFpsT);
    fpsAccum = 0;
    lastFpsT = now;
    refreshStats();
  }
}

// -- Render loop -------------------------------------------------------------
const tmpVec = new THREE.Vector3();
function tick() {
  requestAnimationFrame(tick);
  controls.update();
  if (currentLine) {
    // Update the per-vertex view direction's reference camera position.
    camera.getWorldPosition(tmpVec);
    currentLine.material.uniforms.uCameraPos.value.copy(tmpVec);
  }
  renderer.render(scene, camera);
  tickFps(performance.now());
}

function onResize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", onResize);

// Phones only: one-shot request for browser fullscreen on the first touch,
// to escape the URL bar.  Browsers require a user gesture, and OrbitControls
// leaves the initial pointerdown alone.  Mouse and pen events are ignored so
// a desktop click does not hijack the window.
let fsTriedAlready = false;
function tryEnterFullscreen(ev) {
  if (ev && ev.pointerType !== "touch") return;
  if (fsTriedAlready) return;
  fsTriedAlready = true;
  const el = document.documentElement;
  const req = el.requestFullscreen
           || el.webkitRequestFullscreen
           || el.webkitEnterFullscreen;
  if (req) req.call(el).catch(() => {});
}
els.canvas.addEventListener("pointerdown", tryEnterFullscreen);

// -- Manifest + scene picker -------------------------------------------------
async function init() {
  // WebGL2 sanity check.
  const gl = renderer.getContext();
  if (!(gl instanceof WebGL2RenderingContext)) {
    fail("WebGL2 not available - needed for 32-bit indices + primitive restart.");
    return;
  }
  onResize();
  tick();

  let manifest;
  try {
    manifest = await fetch("data/manifest.json").then((r) => r.json());
  } catch (e) {
    fail("manifest.json missing under data/ - run web/precompute.py first");
    return;
  }
  if (!manifest.scenes || !manifest.scenes.length) {
    fail("manifest has no scenes"); return;
  }

  for (const s of manifest.scenes) {
    const opt = document.createElement("option");
    opt.value = s.file;
    opt.textContent = s.id;
    els.sceneSel.appendChild(opt);
  }
  els.sceneSel.addEventListener("change", () => {
    const opt = els.sceneSel.selectedOptions[0];
    loadScene(`data/${opt.value}`, opt.textContent);
  });

  // ?scene= names a bundle directly, which is the only way to choose one
  // when ?ui=0 has hidden the picker.  Anything else falls back to the
  // manifest's first entry.
  let initial = manifest.scenes[0];
  if (SCENE_FILE) {
    const match = manifest.scenes.find((s) => s.file === SCENE_FILE);
    if (match) initial = match;
    else fail(`scene "${SCENE_FILE}" is not in the manifest; loading ${initial.file}`);
  }
  els.sceneSel.value = initial.file;
  loadScene(`data/${initial.file}`, initial.id);
}

init().catch((e) => fail(e.message));
