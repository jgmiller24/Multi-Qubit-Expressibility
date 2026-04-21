from __future__ import annotations

import gc
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import math
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from urllib.parse import quote

# =============================================================================
# Dynamic import of the user's actual QNN script
# =============================================================================
def load_user_module(script_path: str):
    script_file = Path(script_path).expanduser().resolve()
    if not script_file.exists():
        raise FileNotFoundError(f"Script not found: {script_file}")

    spec = importlib.util.spec_from_file_location("user_hybrid_module", str(script_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {script_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_resource(show_spinner=False)
def cached_load_user_module(script_path: str):
    return load_user_module(script_path)


# =============================================================================
# Helpers
# =============================================================================
def finalize_stop_state(message: str = "Training stopped. Change parameters and run again."):
    st.session_state.hide_controls = False
    st.session_state.start_training = False
    st.session_state.training_active = False
    st.session_state.stop_requested = False
    st.session_state.training_complete = False
    st.session_state.stop_message = message
    st.session_state.live_snapshot_history = []
    st.session_state.pop("last_training_results", None)


def reset_after_stop(message: str = "Training stopped. Change parameters and run again."):
    finalize_stop_state(message)
    st.rerun()


def finish_training_and_restore_controls(select_summary: bool = True):
    st.session_state.hide_controls = False
    st.session_state.start_training = False
    st.session_state.training_active = False
    st.session_state.training_complete = True
    st.session_state.post_run_default_view = "Summary" if select_summary else "Training"


def format_duration_compact(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def render_svg_data_uri(svg: str, max_width_px: int | None = 1180, align: str = "center"):
    justify = {
        "center": "center",
        "left": "flex-start",
        "right": "flex-end",
    }.get(align, "center")

    if max_width_px is None:
        img_style = "width:auto; max-width:100%; height:auto; display:block;"
    else:
        img_style = f"width:100%; max-width:{max_width_px}px; height:auto; display:block;"

    uri = "data:image/svg+xml;utf8," + quote(svg)
    st.markdown(
        f"""
        <div style="width:100%; display:flex; justify-content:{justify}; margin:0; padding:0;">
            <img src="{uri}" style="{img_style}" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def _pil_resample():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def _load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates += [
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        ]
    candidates += [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _bezier_points(p0, p1, p2, p3, n=28):
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1.0 - t
        x = (
            mt**3 * p0[0]
            + 3 * mt**2 * t * p1[0]
            + 3 * mt * t**2 * p2[0]
            + t**3 * p3[0]
        )
        y = (
            mt**3 * p0[1]
            + 3 * mt**2 * t * p1[1]
            + 3 * mt * t**2 * p2[1]
            + t**3 * p3[1]
        )
        pts.append((x, y))
    return pts


def _sparsify_edges(mat: np.ndarray, keep_per_row: int = 2, rel_thresh: float = 0.35) -> np.ndarray:
    out = np.zeros_like(mat, dtype=float)
    if mat.size == 0:
        return out

    for i in range(mat.shape[0]):
        row = mat[i]
        if not np.any(row > 0):
            continue
        idx = np.argsort(-row)[:keep_per_row]
        peak = row[idx[0]]
        for j in idx:
            if row[j] >= rel_thresh * peak:
                out[i, j] = row[j]
    return out


def _clustered_y_positions(n: int, center: float = 195.0, spacing: float = 42.0):
    if n <= 0:
        return []
    start = center - spacing * (n - 1) / 2.0
    return [start + i * spacing for i in range(n)]


def _fit_cluster_spacing(n: int, preferred: float, available_height: float, min_spacing: float = 30.0) -> float:
    if n <= 1:
        return preferred
    max_fit = available_height / max(n - 1, 1)
    return max(min_spacing, min(preferred, max_fit))


def build_visual_signal_flow_svg(sample_item, cdbg, qdbg, rdbg, class_names: List[str]) -> str:
    input_vals = _patch_strengths_from_input(sample_item["x"], grid=2)
    _, conv_vals = _topk_channel_strengths(cdbg["conv1"], k=5)
    _, latent_vals, _ = _topk_vector_strengths(cdbg["fc1"], k=6)
    probs = rdbg["probs"].detach().cpu().numpy()

    qcount = int(qdbg["qubit_count"])
    qubit_strength = np.zeros(qcount, dtype=np.float32)

    labels = list(qdbg["expectation_labels"])
    feature_vals = np.abs(qdbg["expectations"].detach().cpu().numpy())

    for feat_idx, label in enumerate(labels):
        matched_any = False
        for q in range(qcount):
            if str(q) in label:
                qubit_strength[q] += float(feature_vals[feat_idx])
                matched_any = True
        if not matched_any:
            qubit_strength += float(feature_vals[feat_idx]) / max(qcount, 1)

    q_to_out = aggregate_qubit_output_strengths(qdbg, rdbg, class_names)

    input_strength = _normalize_strength(input_vals, floor=0.18)
    conv_strength = _normalize_strength(conv_vals, floor=0.18)
    latent_strength = _normalize_strength(latent_vals, floor=0.18)
    qubit_strength = _normalize_strength(qubit_strength, floor=0.22)
    out_strength = _normalize_strength(probs, floor=0.24)

    layer_count_for_height = max(
        4,
        len(conv_strength),
        len(latent_strength),
        len(qubit_strength),
        len(out_strength),
    )
    panel_top = 40
    extra_output_room = max(0, len(out_strength) - 6) * 22
    panel_h = max(340, 230 + 18 * layer_count_for_height + extra_output_room)
    panel_bottom = panel_top + panel_h
    svg_h = panel_bottom + 34
    center_y = panel_top + panel_h / 2.0
    usable_h = max(150.0, panel_h - 54.0)

    input_cx = 140
    cnn_cx = 355
    latent_cx = 605
    qubit_cx = 845
    output_cx = 1065

    input_xs = [112, 172, 112, 172]
    input_offset = 43
    input_ys = [center_y - input_offset, center_y - input_offset, center_y + input_offset, center_y + input_offset]
    input_rs = [8 + 9 * s for s in input_strength]

    conv_spacing = _fit_cluster_spacing(len(conv_strength), preferred=46, available_height=usable_h, min_spacing=34)
    latent_spacing = _fit_cluster_spacing(len(latent_strength), preferred=40, available_height=usable_h, min_spacing=30)
    quantum_spacing = _fit_cluster_spacing(len(qubit_strength), preferred=64, available_height=usable_h, min_spacing=46)
    output_spacing = _fit_cluster_spacing(len(out_strength), preferred=52, available_height=usable_h, min_spacing=40)

    conv_ys = _clustered_y_positions(len(conv_strength), center=center_y, spacing=conv_spacing)
    latent_ys = _clustered_y_positions(len(latent_strength), center=center_y, spacing=latent_spacing)
    quantum_ys = _clustered_y_positions(len(qubit_strength), center=center_y, spacing=quantum_spacing)
    output_ys = _clustered_y_positions(len(out_strength), center=center_y, spacing=output_spacing)

    edge_in_conv = np.outer(input_strength, conv_strength)
    edge_conv_latent = np.outer(conv_strength, latent_strength)
    edge_latent_q = np.outer(latent_strength, qubit_strength)
    edge_q_out = q_to_out.copy()

    pred_idx = int(np.argmax(probs))
    edge_q_out[:, pred_idx] = np.maximum(edge_q_out[:, pred_idx], q_to_out[:, pred_idx])

    def curve(x1, y1, x2, y2):
        c1 = x1 + 0.38 * (x2 - x1)
        c2 = x2 - 0.38 * (x2 - x1)
        return f"M {x1:.1f},{y1:.1f} C {c1:.1f},{y1:.1f} {c2:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"

    def layer_edges(x1, ys1, x2, ys2, strengths, rgb):
        strengths = np.asarray(strengths, dtype=float)
        if strengths.size == 0:
            return ""

        smax = float(np.max(strengths)) + 1e-8
        strengths = strengths / smax

        out = []
        bins = [
            (0.72, 4.0, 0.46),
            (0.48, 2.8, 0.25),
            (0.22, 1.6, 0.12),
        ]
        for thresh, width, alpha in bins:
            stroke = _hex(rgb)
            for i in range(len(ys1)):
                for j in range(len(ys2)):
                    s = strengths[i, j]
                    if s >= thresh:
                        out.append(
                            f'<path d="{curve(x1, ys1[i], x2, ys2[j])}" '
                            f'stroke="{stroke}" stroke-opacity="{alpha:.3f}" '
                            f'stroke-width="{width:.2f}" fill="none" stroke-linecap="round"/>'
                        )
        return "".join(out)

    svg_parts = [
        f'<svg viewBox="0 0 1180 {svg_h:.0f}" width="100%" height="{svg_h:.0f}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">',
        """
        <defs>
          <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="3.5" result="blur"/>
            <feMerge>
              <feMergeNode in="blur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        """,
        f'<rect x="0" y="0" width="1180" height="{svg_h:.0f}" fill="#070b14"/>',
        f'<rect x="40" y="{panel_top}" width="200" height="{panel_h}" rx="22" fill="rgba(70,130,180,0.10)" stroke="rgba(180,220,255,0.16)"/>',
        f'<rect x="270" y="{panel_top}" width="170" height="{panel_h}" rx="22" fill="rgba(46,204,113,0.10)" stroke="rgba(210,255,225,0.14)"/>',
        f'<rect x="520" y="{panel_top}" width="170" height="{panel_h}" rx="22" fill="rgba(241,196,15,0.10)" stroke="rgba(255,245,180,0.14)"/>',
        f'<rect x="760" y="{panel_top}" width="170" height="{panel_h}" rx="22" fill="rgba(155,89,182,0.10)" stroke="rgba(240,215,255,0.14)"/>',
        f'<rect x="945" y="{panel_top}" width="220" height="{panel_h}" rx="22" fill="rgba(231,76,60,0.10)" stroke="rgba(255,215,210,0.14)"/>',
        f'<text x="{input_cx}" y="24" text-anchor="middle" font-size="16" font-weight="700" fill="#dbe7ff">Input</text>',
        f'<text x="{cnn_cx}" y="24" text-anchor="middle" font-size="16" font-weight="700" fill="#dff7ea">CNN</text>',
        f'<text x="{latent_cx}" y="24" text-anchor="middle" font-size="16" font-weight="700" fill="#fff1c9">Latent</text>',
        f'<text x="{qubit_cx}" y="24" text-anchor="middle" font-size="16" font-weight="700" fill="#f1ddff">Qubits</text>',
        f'<text x="{output_cx}" y="24" text-anchor="middle" font-size="16" font-weight="700" fill="#ffd9d3">Output</text>',
        layer_edges(175, input_ys, cnn_cx, conv_ys, edge_in_conv, (120, 170, 235)),
        layer_edges(cnn_cx, conv_ys, latent_cx, latent_ys, edge_conv_latent, (90, 210, 140)),
        layer_edges(latent_cx, latent_ys, qubit_cx, quantum_ys, edge_latent_q, (235, 192, 75)),
        layer_edges(qubit_cx, quantum_ys, output_cx, output_ys, edge_q_out, (210, 120, 185)),
    ]

    horizontal_pairs = [(0, 1), (2, 3)]
    for left_idx, right_idx in horizontal_pairs:
        x1 = input_xs[left_idx] + input_rs[left_idx]
        y1 = input_ys[left_idx]
        x2 = input_xs[right_idx] - input_rs[right_idx]
        y2 = input_ys[right_idx]

        svg_parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="rgba(140,190,255,0.34)" stroke-width="2.4" stroke-linecap="round"/>'
        )

    for x, y, s, r in zip(input_xs, input_ys, input_strength, input_rs):
        fill = _rgba_str((95, 155, 220), 0.28 + 0.70 * s)
        svg_parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r:.1f}" fill="{fill}" stroke="white" stroke-width="1.2"/>'
        )

    for y, s in zip(conv_ys, conv_strength):
        r = 10 + 11 * s
        fill = _rgba_str((61, 214, 126), 0.28 + 0.70 * s)
        svg_parts.append(f'<circle cx="{cnn_cx}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="white" stroke-width="1.2"/>')

    for y, s in zip(latent_ys, latent_strength):
        r = 9 + 10 * s
        fill = _rgba_str((241, 196, 15), 0.28 + 0.70 * s)
        svg_parts.append(f'<circle cx="{latent_cx}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="white" stroke-width="1.2"/>')

    for qi, (y, s) in enumerate(zip(quantum_ys, qubit_strength)):
        r = 12 + 12 * s
        fill = _rgba_str((155, 89, 182), 0.30 + 0.68 * s)
        svg_parts.append(
            f'<circle cx="{qubit_cx}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="white" stroke-width="1.4" filter="url(#glow)"/>'
        )
        svg_parts.append(
            f'<text x="{qubit_cx}" y="{y + 4:.1f}" text-anchor="middle" font-size="11" fill="white">q{qi}</text>'
        )

    for i, (digit, y, s, p) in enumerate(zip(class_names, output_ys, out_strength, probs)):
        is_pred = i == pred_idx
        r = (13 + 9 * s) if not is_pred else (17 + 10 * s)
        fill = _rgba_str((231, 76, 60), 0.30 + 0.58 * s if not is_pred else 0.96)
        stroke_w = 1.2 if not is_pred else 2.5
        filt = ' filter="url(#glow)"' if is_pred else ""
        svg_parts.append(
            f'<circle cx="{output_cx}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="white" stroke-width="{stroke_w}"{filt}/>'
        )
        svg_parts.append(
            f'<text x="{output_cx}" y="{y + 5:.1f}" text-anchor="middle" font-size="15" font-weight="700" fill="#f4f7ff">{digit}</text>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)

def _rgba_str(rgb, alpha: float) -> str:
    r, g, b = rgb
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _hex(rgb) -> str:
    r, g, b = rgb
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _normalize_strength(values, floor: float = 0.14):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    vmax = float(np.max(arr))
    if vmax <= 1e-8:
        return np.full_like(arr, floor)
    return floor + (1.0 - floor) * (arr / vmax)


def _patch_strengths_from_input(x: torch.Tensor, grid: int = 3) -> np.ndarray:
    img = denorm_mnist_image(x[0, 0])
    h, w = img.shape
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)

    vals = []
    for gy in range(grid):
        for gx in range(grid):
            patch = img[ys[gy]:ys[gy + 1], xs[gx]:xs[gx + 1]]
            vals.append(float(np.mean(patch)))
    return np.asarray(vals, dtype=float)


def _topk_channel_strengths(feature_tensor: torch.Tensor, k: int):
    scores = feature_tensor.abs().mean(dim=(1, 2)).detach().cpu().numpy()
    k = min(k, len(scores))
    idx = np.argsort(-scores)[:k]
    vals = scores[idx]
    return idx, vals


def _topk_vector_strengths(vec: torch.Tensor, k: int):
    arr = vec.detach().cpu().numpy().flatten()
    scores = np.abs(arr)
    k = min(k, len(scores))
    idx = np.argsort(-scores)[:k]
    vals = scores[idx]
    signed_vals = arr[idx]
    return idx, vals, signed_vals


def _layer_y_positions(n: int, y_min: float = 36, y_max: float = 244):
    if n <= 1:
        return [140.0]
    return np.linspace(y_min, y_max, n).tolist()


def build_latent_landscape_image(vec: torch.Tensor, out_h: int = 120, out_w: int = 220) -> np.ndarray:
    arr = vec.detach().cpu().numpy().flatten()
    if arr.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    cols = max(4, min(8, int(np.ceil(np.sqrt(arr.size)))))
    rows = int(np.ceil(arr.size / cols))
    padded = np.zeros(rows * cols, dtype=np.float32)
    padded[: arr.size] = arr
    grid = padded.reshape(rows, cols)

    grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(grid_t, size=(out_h, out_w), mode="bicubic", align_corners=False)[0, 0].cpu().numpy()

    norm = float(np.max(np.abs(up))) + 1e-8
    z = up / norm

    pos = np.clip(z, 0.0, 1.0)
    neg = np.clip(-z, 0.0, 1.0)
    mag = np.clip(np.abs(z), 0.0, 1.0)

    # warm = positive, cool = negative, brightness = magnitude
    r = 0.18 + 0.82 * pos + 0.18 * mag
    g = 0.12 + 0.45 * (1.0 - mag)
    b = 0.18 + 0.82 * neg + 0.10 * (1.0 - pos)

    rgb = np.stack([np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)], axis=-1)

    # subtle contour bands so it reads more like a landscape
    contour = 0.08 * np.sin(18.0 * z)
    rgb = np.clip(rgb + contour[..., None], 0.0, 1.0)

    return (255 * rgb).astype(np.uint8)


def build_quantum_prep_svg(theta: torch.Tensor) -> str:
    arr = theta.detach().cpu().numpy().flatten()
    if arr.size == 0:
        return '<svg viewBox="0 0 120 60" xmlns="http://www.w3.org/2000/svg"></svg>'

    q = max(1, arr.size // 2)
    ry = arr[:q]
    rx = arr[q:q * 2] if arr.size >= 2 * q else np.zeros_like(ry)

    mat = np.vstack([ry, rx]).astype(np.float32)
    norm = float(np.max(np.abs(mat))) + 1e-8
    z = mat / norm

    pos = np.clip(z, 0.0, 1.0)
    neg = np.clip(-z, 0.0, 1.0)
    mag = np.clip(np.abs(z), 0.0, 1.0)

    cell_w = 108
    cell_h = 58
    pad = 10
    width = q * cell_w + (q - 1) * pad
    height = 2 * cell_h + pad

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    ]

    for row in range(2):
        for col in range(q):
            val = mat[row, col]
            p = pos[row, col]
            n = neg[row, col]
            m = mag[row, col]

            r = int(255 * np.clip(0.16 + 0.84 * p, 0, 1))
            g = int(255 * np.clip(0.10 + 0.22 * (1.0 - m), 0, 1))
            b = int(255 * np.clip(0.16 + 0.84 * n, 0, 1))

            x = col * (cell_w + pad)
            y = row * (cell_h + pad)
            gate = "RY" if row == 0 else "RX"
            cx = x + cell_w / 2

            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" rx="16" '
                f'fill="rgb({r},{g},{b})" stroke="rgba(255,255,255,0.92)" stroke-width="1.5"/>'
            )
            parts.append(
                f'<text x="{cx}" y="{y + 23}" text-anchor="middle" font-size="14" font-weight="700" fill="white">{gate}{col}</text>'
            )
            parts.append(
                f'<text x="{cx}" y="{y + 43}" text-anchor="middle" font-size="15" fill="white">{val:+.2f}</text>'
            )

    parts.append("</svg>")
    return "".join(parts)


def aggregate_qubit_output_strengths(qdbg, rdbg, class_names: List[str]) -> np.ndarray:
    """
    Aggregate feature-to-output contributions by actual qubit index.
    Result shape: [num_qubits, num_classes]
    """
    qcount = int(qdbg["qubit_count"])
    num_classes = len(class_names)
    contribs = rdbg["contribs"].detach().cpu().numpy()  # [classes, features]
    labels = list(qdbg["expectation_labels"])

    out = np.zeros((qcount, num_classes), dtype=np.float32)

    for feat_idx, label in enumerate(labels):
        matched_any = False
        for q in range(qcount):
            if str(q) in label:
                out[q, :] += np.abs(contribs[:, feat_idx])
                matched_any = True

        if not matched_any:
            # fallback: distribute evenly if feature name doesn't map cleanly
            out += np.abs(contribs[:, feat_idx])[None, :] / max(qcount, 1)

    return out


def render_deeper_representation_panel(cdbg):
    st.markdown("**Deeper CNN focus**")
    st.image(
        build_focus_projection_image(cdbg["conv2"], top_k=12, out_size=120),
        caption="Composite of the strongest deeper feature maps",
        width=220,
    )

    st.markdown("**Latent gradient landscape**")
    st.image(
        build_latent_landscape_image(cdbg["fc1"], out_h=120, out_w=220),
        caption="Warm = positive, cool = negative, brightness = strength",
        width=220,
    )

    st.markdown("**Quantum prep code**")
    render_svg_data_uri(build_quantum_prep_svg(cdbg["theta"]), max_width_px=None, align="left")


def get_loader_iterable(loader):
    return loader.loader if hasattr(loader, "loader") else loader


def get_loader_dataset(loader):
    if hasattr(loader, "dataset"):
        return loader.dataset
    if hasattr(loader, "loader") and hasattr(loader.loader, "dataset"):
        return loader.loader.dataset
    return None


def build_observable_labels(qubit_count: int, max_features: Optional[int] = None) -> List[str]:
    if max_features is None:
        max_features = 2 ** qubit_count

    labels: List[str] = []

    def append(label: str) -> bool:
        labels.append(label)
        return len(labels) >= max_features

    for i in range(qubit_count):
        if append(f"Z{i}"):
            return labels
    for i in range(qubit_count):
        if append(f"X{i}"):
            return labels
    for i in range(qubit_count - 1):
        if append(f"Z{i}Z{i+1}"):
            return labels
    for i in range(qubit_count - 1):
        if append(f"X{i}X{i+1}"):
            return labels
    if append("Z_all"):
        return labels
    if append("X_all"):
        return labels
    return labels


def denorm_mnist_image(x: torch.Tensor) -> np.ndarray:
    img = x.detach().cpu().numpy()
    img = np.clip((img * 0.3081) + 0.1307, 0.0, 1.0)
    return img


def feature_map_to_rgb(feature_map: torch.Tensor) -> np.ndarray:
    arr = feature_map.detach().float().cpu().numpy()
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)

    # simple blue -> cyan -> white style mapping, lightweight and readable
    r = np.clip(0.15 + 0.85 * arr, 0.0, 1.0)
    g = np.clip(0.25 + 0.75 * arr, 0.0, 1.0)
    b = np.clip(0.40 + 0.60 * arr, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (255 * rgb).astype(np.uint8)


def pick_current_batch_sample(x: torch.Tensor, y: torch.Tensor, class_names: List[str], batch_idx: int):
    if x.size(0) == 0:
        raise ValueError("Empty batch encountered.")

    sample_i = (batch_idx - 1) % x.size(0)
    class_idx = int(y[sample_i].detach().cpu().item())
    digit_label = str(class_names[class_idx])

    return {
        "x": x[sample_i : sample_i + 1].detach().cpu(),
        "class_index": class_idx,
        "digit": digit_label,
        "batch_index": batch_idx,
        "sample_index": sample_i,
    }


def maybe_empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def maybe_sync_cuda():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def release_training_resources(*objects):
    for obj in objects:
        if obj is None:
            continue
        try:
            if isinstance(obj, nn.Module):
                obj.zero_grad(set_to_none=True)
                if any(p.is_cuda for p in obj.parameters()):
                    obj.to('cpu')
        except Exception:
            pass
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    maybe_sync_cuda()
    maybe_empty_cuda_cache()


@dataclass
class Args:
    num_qubits: int
    samples: int
    data_dir: str
    device: str
    q_device: str
    batch_size: int
    num_workers: int
    epochs: int
    optimizer: str
    lr: float
    weight_decay: float
    dropout: float
    early_stop_metric: str
    early_stop_patience: int
    scheduler_patience: int
    live_every_batches: int
    conv_maps_to_show: int


# =============================================================================
# Single-pass visualization extraction
# =============================================================================
def collect_live_viz_data(model, x: torch.Tensor):
    with torch.no_grad():
        conv1 = model.classical[0](x)
        relu1 = model.classical[1](conv1)
        pool1 = model.classical[2](relu1)

        conv2 = model.classical[3](pool1)
        relu2 = model.classical[4](conv2)
        pool2 = model.classical[5](relu2)

        flat = model.classical[6](pool2)
        fc1 = model.classical[7](flat)
        fc1_relu = model.classical[8](fc1)
        dropped = model.classical[9](fc1_relu)
        theta = model.classical[10](dropped)

        q_features = model.quantum(theta)
        logits = model.readout(q_features)[0]
        probs = F.softmax(logits, dim=0)

    q = theta.shape[1] // 2
    angle_labels = [f"ry[{i}]" for i in range(q)] + [f"rx[{i}]" for i in range(q)]
    feature_labels = getattr(
        model.quantum.runner,
        "hamiltonian_names",
        build_observable_labels(model.quantum.runner.qubit_count, q_features.shape[1]),
    )

    cdbg = {
        "conv1": conv1[0],
        "conv2": conv2[0],
        "fc1": fc1[0],
        "theta": theta[0],
    }
    qdbg = {
        "angles": theta[0],
        "angle_labels": angle_labels,
        "expectations": q_features[0],
        "expectation_labels": feature_labels,
        "qubit_count": int(model.quantum.runner.qubit_count),
    }
    rdbg = {
        "feature_values": q_features[0],
        "feature_labels": feature_labels,
        "contribs": model.readout.weight.detach() * q_features[0].detach().unsqueeze(0),
        "bias": model.readout.bias.detach(),
        "logits": logits.detach(),
        "probs": probs.detach(),
    }
    return cdbg, qdbg, rdbg


# =============================================================================
# Figures
# =============================================================================
def build_loss_figure(history) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history["train_loss"]) + 1)),
            y=history["train_loss"],
            mode="lines+markers",
            name="train loss",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history["eval_epochs"],
            y=history["test_loss"],
            mode="lines+markers",
            name="test loss",
        )
    )
    max_epoch = max(1, len(history.get("train_loss", [])))
    fig.update_layout(
        title="Loss",
        height=340,
        margin={"l": 10, "r": 10, "t": 42, "b": 10},
        xaxis={
            "title": "epoch",
            "tickmode": "linear",
            "tick0": 1,
            "dtick": 1,
            "range": [0.8, max_epoch + 0.2],
        },
        yaxis_title="cross-entropy loss",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def build_accuracy_figure(history) -> go.Figure:
    train_acc = [100.0 * x for x in history["train_acc"]]
    test_acc = [100.0 * x for x in history["test_acc"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(train_acc) + 1)),
            y=train_acc,
            mode="lines+markers",
            name="train accuracy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history["eval_epochs"],
            y=test_acc,
            mode="lines+markers",
            name="test accuracy",
        )
    )
    max_epoch = max(1, len(train_acc))
    fig.update_layout(
        title="Accuracy",
        height=340,
        margin={"l": 10, "r": 10, "t": 42, "b": 10},
        xaxis={
            "title": "epoch",
            "tickmode": "linear",
            "tick0": 1,
            "dtick": 1,
            "range": [0.8, max_epoch + 0.2],
        },
        yaxis_title="accuracy (%)",
        yaxis={"range": [0, 100]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


def build_human_flow_figure(sample_item, cdbg, qdbg, rdbg, class_names: List[str]) -> go.Figure:
    probs = rdbg["probs"].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_digit = class_names[pred_idx]
    confidence = float(probs[pred_idx]) * 100.0

    cnn_strength = float(
        np.mean(
            [
                cdbg["conv1"].abs().mean().item(),
                cdbg["conv2"].abs().mean().item(),
                cdbg["theta"].abs().mean().item(),
            ]
        )
    )
    quantum_strength = float(qdbg["expectations"].abs().mean().item())

    fig = go.Figure()

    box_specs = [
        {
            "x0": 0.02,
            "x1": 0.22,
            "title": "Input image",
            "body": f"True digit: {sample_item['digit']}<br>Current batch sample",
        },
        {
            "x0": 0.28,
            "x1": 0.48,
            "title": "CNN feature extraction",
            "body": f"Edges → shapes → digit features<br>Activation strength: {cnn_strength:.3f}",
        },
        {
            "x0": 0.54,
            "x1": 0.74,
            "title": "Quantum state",
            "body": f"{qdbg['qubit_count']} qubit(s)<br>Expectation signal: {quantum_strength:.3f}",
        },
        {
            "x0": 0.80,
            "x1": 0.98,
            "title": "Classifier output",
            "body": f"Predicted digit: {pred_digit}<br>Confidence: {confidence:.1f}%",
        },
    ]

    fills = ["rgba(70,130,180,0.10)", "rgba(46,204,113,0.10)", "rgba(155,89,182,0.10)", "rgba(231,76,60,0.10)"]
    borders = ["rgba(70,130,180,0.75)", "rgba(46,204,113,0.75)", "rgba(155,89,182,0.75)", "rgba(231,76,60,0.75)"]

    for spec, fill, border in zip(box_specs, fills, borders):
        fig.add_shape(
            type="rect",
            x0=spec["x0"],
            x1=spec["x1"],
            y0=0.18,
            y1=0.82,
            line={"color": border, "width": 2},
            fillcolor=fill,
            layer="below",
        )
        fig.add_annotation(
            x=(spec["x0"] + spec["x1"]) / 2,
            y=0.62,
            text=f"<b>{spec['title']}</b>",
            showarrow=False,
            font={"size": 16},
        )
        fig.add_annotation(
            x=(spec["x0"] + spec["x1"]) / 2,
            y=0.40,
            text=spec["body"],
            showarrow=False,
            font={"size": 13},
            align="center",
        )

    for start_x, end_x in [(0.22, 0.28), (0.48, 0.54), (0.74, 0.80)]:
        fig.add_annotation(
            x=end_x - 0.01,
            y=0.50,
            ax=start_x + 0.01,
            ay=0.50,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2.5,
        )

    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(
        title="Live end-to-end signal flow",
        height=230,
        margin={"l": 10, "r": 10, "t": 46, "b": 10},
    )
    return fig


def build_probability_figure(class_names: List[str], probs: torch.Tensor) -> go.Figure:
    probs_np = probs.detach().cpu().numpy() * 100.0
    order = np.argsort(-probs_np)

    ordered_classes = [class_names[i] for i in order]
    ordered_probs = [float(probs_np[i]) for i in order]

    df = pd.DataFrame(
        {
            "class": ordered_classes,
            "probability_pct": ordered_probs,
        }
    )

    fig = px.bar(
        df,
        x="probability_pct",
        y="class",
        orientation="h",
        text=df["probability_pct"].map(lambda v: f"{v:.1f}%"),
        title="Output classification",
    )
    max_prob = float(np.max(probs_np)) if len(probs_np) > 0 else 100.0
    x_max = max(100.0, max_prob * 1.35 + 3.0)
    fig.update_traces(textposition="outside", cliponaxis=False, textfont={"size": 12})
    fig.update_layout(
        height=max(340, 38 * len(class_names) + 86),
        margin={"l": 28, "r": 90, "t": 42, "b": 10},
        xaxis_title="confidence (%)",
        yaxis_title="digit",
        yaxis={
            "type": "category",
            "categoryorder": "array",
            "categoryarray": ordered_classes[::-1],
            "tickmode": "array",
            "tickvals": ordered_classes,
            "ticktext": ordered_classes,
            "automargin": True,
        },
        xaxis={"range": [0, x_max], "automargin": True},
    )
    return fig


def build_qubit_state_figure(qdbg) -> go.Figure:
    values = qdbg["expectations"].detach().cpu().numpy().tolist()
    labels = list(qdbg["expectation_labels"])
    lookup = {name: float(val) for name, val in zip(labels, values)}

    qcount = qdbg["qubit_count"]
    spacing = 3.0

    fig = go.Figure()

    for i in range(qcount):
        cx = i * spacing
        x_val = lookup.get(f"X{i}", 0.0)
        z_val = lookup.get(f"Z{i}", 0.0)

        fig.add_shape(
            type="circle",
            x0=cx - 1.0,
            x1=cx + 1.0,
            y0=-1.0,
            y1=1.0,
            line={"color": "rgba(120,120,120,0.85)", "width": 2},
        )
        fig.add_shape(
            type="line",
            x0=cx - 1.1,
            x1=cx + 1.1,
            y0=0,
            y1=0,
            line={"color": "rgba(160,160,160,0.55)", "width": 1},
        )
        fig.add_shape(
            type="line",
            x0=cx,
            x1=cx,
            y0=-1.1,
            y1=1.1,
            line={"color": "rgba(160,160,160,0.55)", "width": 1},
        )
        fig.add_annotation(
            x=cx + x_val,
            y=z_val,
            ax=cx,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.1,
            arrowwidth=2.4,
        )
        fig.add_trace(
            go.Scatter(
                x=[cx + x_val],
                y=[z_val],
                mode="markers+text",
                text=[f"q{i}"],
                textposition="top center",
                marker={"size": 10},
                showlegend=False,
                hovertemplate=f"q{i}<br>X={x_val:.3f}<br>Z={z_val:.3f}<extra></extra>",
            )
        )
        fig.add_annotation(
            x=cx,
            y=-1.34,
            text=f"X={x_val:.2f} | Z={z_val:.2f}",
            showarrow=False,
            font={"size": 12},
        )

    fig.update_xaxes(
        visible=False,
        range=[-1.5, max(1.5, spacing * (qcount - 1) + 1.5)],
    )
    fig.update_yaxes(
        visible=False,
        range=[-1.55, 1.45],
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        title="Bloch-style qubit state view",
        height=260,
        margin={"l": 10, "r": 10, "t": 42, "b": 10},
    )
    return fig

def build_focus_projection_image(feature_tensor: torch.Tensor, top_k: int = 12, out_size: int = 120) -> np.ndarray:
    scores = feature_tensor.abs().mean(dim=(1, 2))
    k = min(top_k, feature_tensor.shape[0])
    idx = scores.topk(k).indices

    maps = feature_tensor[idx].abs()
    maps = maps / (maps.amax(dim=(1, 2), keepdim=True) + 1e-8)
    proj = maps.mean(dim=0, keepdim=True).unsqueeze(0)
    proj = F.interpolate(proj, size=(out_size, out_size), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

    proj = np.clip(proj, 0.0, 1.0)
    r = np.clip(0.20 + 0.90 * proj, 0.0, 1.0)
    g = np.clip(0.12 + 0.65 * (proj ** 0.8), 0.0, 1.0)
    b = np.clip(0.20 + 0.35 * (1.0 - proj), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (255 * rgb).astype(np.uint8)


def build_signed_tile_image(vec: torch.Tensor, max_items: int = 24, cols: int = 6, tile_px: int = 18) -> np.ndarray:
    arr = vec.detach().cpu().numpy().flatten()
    if arr.size == 0:
        return np.zeros((tile_px, tile_px, 3), dtype=np.uint8)

    idx = np.argsort(-np.abs(arr))[: min(max_items, arr.size)]
    vals = arr[idx]

    rows = int(np.ceil(len(vals) / cols))
    padded = np.zeros(rows * cols, dtype=float)
    padded[: len(vals)] = vals
    grid = padded.reshape(rows, cols)

    norm = float(np.max(np.abs(grid))) + 1e-8
    pos = np.clip(grid / norm, 0.0, 1.0)
    neg = np.clip(-grid / norm, 0.0, 1.0)
    mag = np.clip(np.abs(grid) / norm, 0.0, 1.0)

    r = 0.15 + 0.85 * pos
    g = 0.10 + 0.45 * (1.0 - mag)
    b = 0.15 + 0.85 * neg
    rgb = np.stack([r, g, b], axis=-1)

    rgb = np.kron(rgb, np.ones((tile_px, tile_px, 1)))
    return (255 * rgb).astype(np.uint8)


# =============================================================================
# Rendering helpers
# =============================================================================
def render_feature_map_grid(feature_tensor: torch.Tensor, title: str, max_maps: int = 6):
    st.markdown(f"**{title}**")

    scores = feature_tensor.abs().mean(dim=(1, 2))
    count = min(max_maps, feature_tensor.shape[0])
    top_idx = scores.topk(count).indices.detach().cpu().tolist()

    rows = [top_idx[i : i + 2] for i in range(0, len(top_idx), 2)]
    for row in rows:
        cols = st.columns(2)
        for col, idx in zip(cols, row):
            with col:
                img = feature_map_to_rgb(feature_tensor[idx])
                st.image(img, caption=f"feature {idx}", width=120)


def _cpu_clone_debug_dict(payload: Dict):
    cloned = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        elif isinstance(value, (list, tuple)):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def capture_live_snapshot(model, sample_item, class_names: List[str], header_text: str, tab_label: str):
    model_device = next(model.parameters()).device
    x_vis = sample_item["x"].to(model_device, non_blocking=True)

    was_training = model.training
    model.eval()

    try:
        cdbg, qdbg, rdbg = collect_live_viz_data(model, x_vis)
        probs = rdbg["probs"].detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_digit = class_names[pred_idx]
        confidence = float(probs[pred_idx]) * 100.0

        snapshot_id = f"{tab_label}-{sample_item['batch_index']}-{sample_item['sample_index']}"

        return {
            "snapshot_id": snapshot_id,
            "tab_label": tab_label,
            "header_text": header_text,
            "pred_digit": pred_digit,
            "confidence": confidence,
            "sample_item": {
                "x": sample_item["x"].detach().cpu().clone(),
                "class_index": sample_item["class_index"],
                "digit": sample_item["digit"],
                "batch_index": sample_item["batch_index"],
                "sample_index": sample_item["sample_index"],
            },
            "cdbg": _cpu_clone_debug_dict(cdbg),
            "qdbg": _cpu_clone_debug_dict(qdbg),
            "rdbg": _cpu_clone_debug_dict(rdbg),
        }
    finally:
        if was_training:
            model.train()
        del x_vis
        gc.collect()
        maybe_empty_cuda_cache()


def render_live_snapshot_from_snapshot(snapshot, class_names: List[str], conv_maps_to_show: int):
    sample_item = snapshot["sample_item"]
    cdbg = snapshot["cdbg"]
    qdbg = snapshot["qdbg"]
    rdbg = snapshot["rdbg"]
    chart_key_prefix = snapshot.get("snapshot_id", snapshot.get("tab_label", "snapshot"))

    st.markdown(f"### {snapshot['header_text']}")
    st.caption(
        f"Live sample is pulled from the current batch "
        f"(batch {sample_item['batch_index']}, item {sample_item['sample_index']}) • "
        f"true digit {sample_item['digit']} • predicted {snapshot['pred_digit']} ({snapshot['confidence']:.1f}%)"
    )

    svg = build_visual_signal_flow_svg(sample_item, cdbg, qdbg, rdbg, class_names)
    render_svg_data_uri(svg, max_width_px=1180, align="center")

    col1, col2, col3, col4 = st.columns([0.92, 1.02, 1.08, 1.28])

    with col1:
        st.subheader("1) Input")
        st.image(
            denorm_mnist_image(sample_item["x"][0, 0]),
            caption="Current image entering the network",
            width=220,
        )

        a, b = st.columns(2)
        with a:
            st.metric("True digit", sample_item["digit"])
        with b:
            st.metric("Predicted", snapshot["pred_digit"])

        st.metric("Confidence", f"{snapshot['confidence']:.1f}%")

    with col2:
        st.subheader("2) CNN early features")
        render_feature_map_grid(cdbg["conv1"], "Top conv1 feature maps", max_maps=conv_maps_to_show)

    with col3:
        st.subheader("3) Deep representation")
        render_deeper_representation_panel(cdbg)

    with col4:
        st.subheader("4) Quantum → output")
        st.plotly_chart(
            build_qubit_state_figure(qdbg),
            width="stretch",
            key=f"{chart_key_prefix}-qubit-state",
        )
        st.plotly_chart(
            build_probability_figure(class_names, rdbg["probs"]),
            width="stretch",
            key=f"{chart_key_prefix}-probability",
        )


def render_live_snapshot_history(history: List[Dict], class_names: List[str], conv_maps_to_show: int):
    if not history:
        return

    snapshots = list(reversed(history))
    for idx, snapshot in enumerate(snapshots):
        label = snapshot.get("tab_label", f"Capture {idx + 1}")
        expanded = idx == 0
        with st.expander(label, expanded=expanded):
            render_live_snapshot_from_snapshot(snapshot, class_names, conv_maps_to_show)


def render_live_snapshot(model, sample_item, class_names: List[str], header_text: str, conv_maps_to_show: int):
    snapshot = capture_live_snapshot(
        model=model,
        sample_item=sample_item,
        class_names=class_names,
        header_text=header_text,
        tab_label="Current",
    )
    render_live_snapshot_from_snapshot(snapshot, class_names, conv_maps_to_show)


# =============================================================================
# Training + summary rendering helpers
# =============================================================================
def build_metrics_dataframe(history) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "epoch": history["eval_epochs"],
            "train_loss": history["train_loss"],
            "train_acc_pct": [100.0 * x for x in history["train_acc"]],
            "test_loss": history["test_loss"],
            "test_acc_pct": [100.0 * x for x in history["test_acc"]],
        }
    )


def render_training_history_panel(history, chart_prefix: str, metrics_rows_visible: int):
    if not history or len(history.get("eval_epochs", [])) == 0:
        st.info("No completed training history is available yet.")
        return

    loss_col, acc_col = st.columns(2)
    with loss_col:
        st.plotly_chart(
            build_loss_figure(history),
            width="stretch",
            key=f"{chart_prefix}-loss",
        )
    with acc_col:
        st.plotly_chart(
            build_accuracy_figure(history),
            width="stretch",
            key=f"{chart_prefix}-accuracy",
        )

    metrics_df = build_metrics_dataframe(history)
    table_height = max(140, 36 + 35 * max(1, int(metrics_rows_visible)))
    st.markdown("### Epoch metrics")
    st.dataframe(metrics_df, width="stretch", height=table_height)


def render_summary_panel(results, user_module):
    st.success(
        f"Training complete. Final test loss={results['final_loss']:.4f}, "
        f"final test accuracy={100.0 * results['final_acc']:.2f}%, "
        f"best epoch={results['best_epoch']}"
    )

    summary_cols = st.columns(3)
    summary_cols[0].metric("Final test loss", f"{results['final_loss']:.4f}")
    summary_cols[1].metric("Final test accuracy", f"{100.0 * results['final_acc']:.2f}%")
    summary_cols[2].metric("Best epoch", f"{results['best_epoch']}")

    if hasattr(user_module, "per_class_accuracy") and len(results["y_true"]) > 0:
        per_class = user_module.per_class_accuracy(
            results["y_true"],
            results["y_pred"],
            num_classes=len(results["target_digits"]),
        )
        per_class_df = pd.DataFrame(
            {
                "class_index": list(per_class.keys()),
                "digit": [results["target_digits"][i] for i in per_class.keys()],
                "accuracy_pct": [100.0 * v for v in per_class.values()],
            }
        )

        st.markdown("### Final digit accuracy")
        st.dataframe(
            per_class_df,
            width="stretch",
            height=min(60 + 35 * len(per_class_df), 420),
        )


# =============================================================================
# Evaluation
# =============================================================================
@torch.no_grad()
def evaluate_model(model, loader, loss_fn) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model_device = next(model.parameters()).device
    iterable = get_loader_iterable(loader)

    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    y_true_chunks = []
    y_pred_chunks = []

    try:
        for x, y in iterable:
            x = x.to(model_device, non_blocking=True)
            y = y.to(model_device, non_blocking=True)

            logits = model(x)
            loss = loss_fn(logits, y)

            preds = logits.argmax(dim=1)
            batch_size = y.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (preds == y).sum().item()
            total_examples += batch_size

            y_true_chunks.append(y.detach().cpu())
            y_pred_chunks.append(preds.detach().cpu())
    finally:
        if was_training:
            model.train()

    if total_examples == 0:
        return 0.0, 0.0, np.array([]), np.array([])

    y_true = torch.cat(y_true_chunks).numpy() if y_true_chunks else np.array([])
    y_pred = torch.cat(y_pred_chunks).numpy() if y_pred_chunks else np.array([])

    return (
        total_loss / total_examples,
        total_correct / total_examples,
        y_true,
        y_pred,
    )


# =============================================================================
# Training
# =============================================================================
def run_training(
    user_module,
    model,
    train_loader,
    test_loader,
    args: Args,
    class_names: List[str],
    metrics_rows_visible: int,
):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = user_module.build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        threshold=1e-3,
        min_lr=1e-5,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "eval_epochs": [],
    }

    best_metric = float("inf") if args.early_stop_metric == "loss" else -float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    model_device = next(model.parameters()).device
    train_iterable = get_loader_iterable(train_loader)

    try:
        train_len = len(train_iterable)
    except Exception:
        train_len = 1

    total_steps = max(1, args.epochs * train_len)
    global_step = 0
    run_start_time = time.perf_counter()

    top_progress = st.progress(0.0, text="Preparing training...")
    epoch_status_box = st.empty()
    batch_status_box = st.empty()
    live_placeholder = st.empty()
    training_history_placeholder = st.empty()
    st.session_state.setdefault("live_snapshot_history", [])

    try:
        for epoch in range(args.epochs):
            if st.session_state.get("stop_requested", False):
                return history, None, None, None, None, best_epoch, True

            model.train()
            running_loss = 0.0
            running_correct = 0
            running_examples = 0

            train_iterable = get_loader_iterable(train_loader)

            for batch_idx, (x, y) in enumerate(train_iterable, start=1):
                logits = None
                loss = None
                preds = None

                if st.session_state.get("stop_requested", False):
                    return history, None, None, None, None, best_epoch, True

                try:
                    x = x.to(model_device, non_blocking=True)
                    y = y.to(model_device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)

                    if st.session_state.get("stop_requested", False):
                        optimizer.zero_grad(set_to_none=True)
                        return history, None, None, None, None, best_epoch, True

                    loss = loss_fn(logits, y)
                    loss.backward()

                    if st.session_state.get("stop_requested", False):
                        optimizer.zero_grad(set_to_none=True)
                        return history, None, None, None, None, best_epoch, True

                    optimizer.step()
                    preds = logits.argmax(dim=1)
                    batch_size = y.size(0)

                    running_loss += loss.item() * batch_size
                    running_correct += (preds == y).sum().item()
                    running_examples += batch_size

                    global_step += 1
                    step_progress = global_step / total_steps

                    avg_loss = running_loss / max(running_examples, 1)
                    avg_acc = running_correct / max(running_examples, 1)

                    elapsed = time.perf_counter() - run_start_time
                    avg_step_seconds = elapsed / max(global_step, 1)
                    remaining_steps = max(0, total_steps - global_step)
                    eta_text = format_duration_compact(avg_step_seconds * remaining_steps)

                    top_progress.progress(
                        step_progress,
                        text=(
                            f"Epoch {epoch + 1}/{args.epochs} • batch {batch_idx}/{train_len} "
                            f"• ETA {eta_text}"
                        ),
                    )
                    batch_status_box.info(
                        f"Epoch {epoch + 1}/{args.epochs} | "
                        f"batch {batch_idx}/{train_len} | "
                        f"train_loss={avg_loss:.4f} | "
                        f"train_acc={100.0 * avg_acc:.2f}%"
                    )

                    should_refresh_live = (
                        batch_idx == 1
                        or batch_idx == train_len
                        or (args.live_every_batches > 0 and batch_idx % args.live_every_batches == 0)
                    )

                    if should_refresh_live and not st.session_state.get("stop_requested", False):
                        current_sample = pick_current_batch_sample(x, y, class_names, batch_idx)
                        snapshot = capture_live_snapshot(
                            model=model,
                            sample_item=current_sample,
                            class_names=class_names,
                            header_text=f"Live network walkthrough — epoch {epoch + 1}, batch {batch_idx}",
                            tab_label=f"E{epoch + 1} · B{batch_idx}",
                        )
                        history_snapshots = st.session_state.setdefault("live_snapshot_history", [])
                        history_snapshots.append(snapshot)
                        st.session_state.live_snapshot_history = history_snapshots[-6:]

                        with live_placeholder.container():
                            render_live_snapshot_from_snapshot(
                                snapshot,
                                class_names=class_names,
                                conv_maps_to_show=args.conv_maps_to_show,
                            )
                finally:
                    try:
                        del x, y, logits, loss, preds
                    except Exception:
                        pass
                    if batch_idx % 4 == 0:
                        gc.collect()
                        maybe_empty_cuda_cache()

            train_loss = running_loss / max(running_examples, 1)
            train_acc = running_correct / max(running_examples, 1)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if st.session_state.get("stop_requested", False):
                return history, None, None, None, None, best_epoch, True

            test_loss, test_acc, _, _ = evaluate_model(model, test_loader, loss_fn)
            scheduler.step(test_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)
            history["eval_epochs"].append(epoch + 1)

            if args.early_stop_metric == "loss":
                improved = test_loss < best_metric
                if improved:
                    best_metric = test_loss
            else:
                improved = test_acc > best_metric
                if improved:
                    best_metric = test_acc

            if improved:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            epoch_status_box.success(
                f"Epoch {epoch + 1}/{args.epochs} complete | "
                f"train_loss={train_loss:.4f} | "
                f"train_acc={100.0 * train_acc:.2f}% | "
                f"test_loss={test_loss:.4f} | "
                f"test_acc={100.0 * test_acc:.2f}% | "
                f"lr={current_lr:.2e} | "
                f"patience={patience_counter}/{args.early_stop_patience}"
            )

            with training_history_placeholder.container():
                live_chart_prefix = f"training-live-run-{st.session_state.get('current_run_id', 0)}-epoch-{epoch + 1}"
                render_training_history_panel(history, chart_prefix=live_chart_prefix, metrics_rows_visible=metrics_rows_visible)

            if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                epoch_status_box.warning(f"Early stopping triggered at epoch {epoch + 1}.")
                break

            gc.collect()
            maybe_empty_cuda_cache()

        if best_state is not None:
            model.load_state_dict(best_state)

        top_progress.progress(1.0, text="Training complete • final evaluation finished")
        final_loss, final_acc, y_true, y_pred = evaluate_model(model, test_loader, loss_fn)
        return history, final_loss, final_acc, y_true, y_pred, best_epoch, False
    finally:
        release_training_resources(optimizer, scheduler)


# =============================================================================
# Control panel rendering
# =============================================================================
def render_controls_panel(container, widget_prefix: str = "control"):
    def wkey(name: str) -> str:
        return f"{widget_prefix}_{name}"

    def persist_control_state(values: Dict[str, object]):
        st.session_state["control_script_path"] = values["script_path"]
        st.session_state["control_num_qubits"] = int(values["num_qubits"])
        st.session_state["control_samples"] = int(values["samples"])
        st.session_state["control_data_dir"] = values["data_dir"]
        st.session_state["control_device"] = values["device_choice"]
        st.session_state["control_q_device"] = values["q_device"]
        st.session_state["control_batch_size"] = int(values["batch_size"])
        st.session_state["control_num_workers"] = int(values["num_workers"])
        st.session_state["control_epochs"] = int(values["epochs"])
        st.session_state["control_optimizer"] = values["optimizer"]
        st.session_state["control_lr"] = float(values["lr"])
        st.session_state["control_weight_decay"] = float(values["weight_decay"])
        st.session_state["control_dropout"] = float(values["dropout"])
        st.session_state["control_early_stop_metric"] = values["early_stop_metric"]
        st.session_state["control_early_stop_patience"] = int(values["early_stop_patience"])
        st.session_state["control_scheduler_patience"] = int(values["scheduler_patience"])
        st.session_state["control_live_every_batches"] = int(values["live_every_batches"])
        st.session_state["control_conv_maps_to_show"] = int(values["conv_maps_to_show"])
        st.session_state["control_metrics_rows_visible"] = int(values["metrics_rows_visible"])
        st.session_state["control_custom_digits_csv"] = values["custom_digits_csv"]

    with container:
        training_active = bool(st.session_state.get("training_active", False))
        stop_pending = bool(st.session_state.get("stop_requested", False))

        run_label = "Run training"
        run_training_clicked = st.button(run_label, type="primary", width="stretch", key=wkey("run_training"))

        if training_active:
            st.caption(
                "A run is currently active. Request stop interrupts the current run and returns you to the controls. "
                "Press Run training to interrupt the current run and launch the configuration below."
            )
        else:
            st.caption("Edit the configuration below, then launch a run from here.")

        st.header("Script")
        script_path = st.text_input(
            "Path to your QNN script",
            value=st.session_state.get("control_script_path", "Experiment_CNN/cnn_qnn.py"),
            key=wkey("script_path"),
        )

        st.header("Parameters")

        qubit_options = [1, 2, 3, 4]
        current_num_qubits = int(st.session_state.get("control_num_qubits", 1))
        qubit_index = qubit_options.index(current_num_qubits) if current_num_qubits in qubit_options else 0
        num_qubits = st.selectbox("num_qubits", qubit_options, index=qubit_index, key=wkey("num_qubits"))

        samples = st.number_input("samples", min_value=100, value=int(st.session_state.get("control_samples", 4000)), step=100, key=wkey("samples"))
        data_dir = st.text_input("data_dir", value=st.session_state.get("control_data_dir", "./data"), key=wkey("data_dir"))

        device_options = ["cpu", "cuda"]
        current_device = st.session_state.get("control_device", "cpu")
        device_index = device_options.index(current_device) if current_device in device_options else 0
        device_choice = st.selectbox("device", device_options, index=device_index, key=wkey("device"))

        q_device_options = ["qpp-cpu", "cuda"]
        current_q_device = st.session_state.get("control_q_device", "qpp-cpu")
        q_device_index = q_device_options.index(current_q_device) if current_q_device in q_device_options else 0
        q_device = st.selectbox("q_device", q_device_options, index=q_device_index, key=wkey("q_device"))

        batch_size = st.number_input("batch_size", min_value=1, value=int(st.session_state.get("control_batch_size", 16)), step=1, key=wkey("batch_size"))
        num_workers = st.number_input("num_workers", min_value=0, value=int(st.session_state.get("control_num_workers", 0)), step=1, key=wkey("num_workers"))
        epochs = st.number_input("epochs", min_value=1, value=int(st.session_state.get("control_epochs", 10)), step=1, key=wkey("epochs"))

        optimizer_options = ["AdamW", "Adam", "RAdam", "NAdam"]
        current_optimizer = st.session_state.get("control_optimizer", "AdamW")
        optimizer_index = optimizer_options.index(current_optimizer) if current_optimizer in optimizer_options else 0
        optimizer = st.selectbox("optimizer", optimizer_options, index=optimizer_index, key=wkey("optimizer"))

        lr = st.number_input("lr", min_value=1e-6, value=float(st.session_state.get("control_lr", 1e-3)), format="%.6f", key=wkey("lr"))
        weight_decay = st.number_input("weight_decay", min_value=0.0, value=float(st.session_state.get("control_weight_decay", 1e-4)), format="%.6f", key=wkey("weight_decay"))
        dropout = st.number_input("dropout", min_value=0.0, max_value=0.95, value=float(st.session_state.get("control_dropout", 0.15)), step=0.01, key=wkey("dropout"))

        early_stop_options = ["loss", "accuracy"]
        current_early_stop_metric = st.session_state.get("control_early_stop_metric", "loss")
        early_stop_index = early_stop_options.index(current_early_stop_metric) if current_early_stop_metric in early_stop_options else 0
        early_stop_metric = st.selectbox("early_stop_metric", early_stop_options, index=early_stop_index, key=wkey("early_stop_metric"))

        early_stop_patience = st.number_input("early_stop_patience (0 = off)", min_value=0, value=int(st.session_state.get("control_early_stop_patience", 2)), step=1, key=wkey("early_stop_patience"))
        scheduler_patience = st.number_input("scheduler_patience", min_value=0, value=int(st.session_state.get("control_scheduler_patience", 0)), step=1, key=wkey("scheduler_patience"))
        live_every_batches = st.number_input("Update live visualization every N batches", min_value=1, value=int(st.session_state.get("control_live_every_batches", 25)), step=1, key=wkey("live_every_batches"))
        conv_maps_to_show = st.number_input("Feature maps per CNN stage", min_value=2, max_value=8, value=int(st.session_state.get("control_conv_maps_to_show", 6)), step=1, key=wkey("conv_maps_to_show"))
        metrics_rows_visible = st.number_input("Epoch metric rows visible before scroll", min_value=3, max_value=40, value=int(st.session_state.get("control_metrics_rows_visible", 8)), step=1, key=wkey("metrics_rows_visible"))
        custom_digits_csv = st.text_input(
            "Digit list override (comma separated, blank = use preset)",
            value=st.session_state.get("control_custom_digits_csv", ""),
            key=wkey("custom_digits_csv"),
        )

    values = {
        "script_path": script_path,
        "num_qubits": int(num_qubits),
        "samples": int(samples),
        "data_dir": data_dir,
        "device_choice": device_choice,
        "q_device": q_device,
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "epochs": int(epochs),
        "optimizer": optimizer,
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "dropout": float(dropout),
        "early_stop_metric": early_stop_metric,
        "early_stop_patience": int(early_stop_patience),
        "scheduler_patience": int(scheduler_patience),
        "live_every_batches": int(live_every_batches),
        "conv_maps_to_show": int(conv_maps_to_show),
        "metrics_rows_visible": int(metrics_rows_visible),
        "custom_digits_csv": custom_digits_csv,
        "run_training_clicked": bool(run_training_clicked),
    }
    persist_control_state(values)
    return values


# =============================================================================
# App
# =============================================================================
def app():
    st.set_page_config(page_title="Hybrid QNN Training + Human-Friendly Live Visualization", layout="wide", initial_sidebar_state="collapsed")

    if "hide_controls" not in st.session_state:
        st.session_state.hide_controls = False
    if "start_training" not in st.session_state:
        st.session_state.start_training = False
    if "training_complete" not in st.session_state:
        st.session_state.training_complete = False
    if "training_active" not in st.session_state:
        st.session_state.training_active = False
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False
    if "stop_message" not in st.session_state:
        st.session_state.stop_message = None
    if "live_snapshot_history" not in st.session_state:
        st.session_state.live_snapshot_history = []
    if "run_counter" not in st.session_state:
        st.session_state.run_counter = 0
    if "current_run_id" not in st.session_state:
        st.session_state.current_run_id = 0
    if "post_run_default_view" not in st.session_state:
        st.session_state.post_run_default_view = "Training"

    st.title("Hybrid QNN Training Visualization")
    st.caption(
        "This is a visualization of the hybrid QNN training process."
    )
    st.caption(
        "Batch sample → CNN feature maps → Qubit state view → Output classification"
    )

    if st.session_state.training_complete:
        st.success("Training complete.")
        st.session_state.training_complete = False
        

    if st.session_state.stop_message:
        st.warning(st.session_state.stop_message)
        st.session_state.stop_message = None

    results = st.session_state.get("last_training_results")
    default_view = st.session_state.get("post_run_default_view", "Training")

    if results is not None and not st.session_state.get("training_active", False) and default_view == "Summary":
        summary_tab, controls_tab, training_tab = st.tabs(["Summary", "Controls", "Training"])
    else:
        controls_tab, training_tab, summary_tab = st.tabs(["Controls", "Training", "Summary"])

    with controls_tab:
        control_values = render_controls_panel(st.container(), widget_prefix="controltab")

    was_training_active = bool(st.session_state.get("training_active", False))


    if control_values["run_training_clicked"]:
        st.session_state.run_counter += 1
        st.session_state.current_run_id = st.session_state.run_counter
        st.session_state.hide_controls = False
        st.session_state.start_training = True
        st.session_state.training_active = False
        st.session_state.stop_requested = False
        st.session_state.training_complete = False
        st.session_state.post_run_default_view = "Training"
        st.session_state.live_snapshot_history = []
        st.session_state.pop("last_training_results", None)
        st.rerun()

    script_path = control_values["script_path"]

    try:
        user_module = cached_load_user_module(script_path)
    except Exception as exc:
        with training_tab:
            st.error(f"Could not import the QNN script: {exc}")
        st.stop()

    required_names = [
        "HybridQNN",
        "build_dataloaders",
        "configure_backend",
        "build_observables",
        "build_optimizer",
        "PRESETS",
        "SHIFT",
        "TEST_SIZE_PCT",
    ]
    missing = [name for name in required_names if not hasattr(user_module, name)]
    if missing:
        with training_tab:
            st.error(f"Imported script is missing required names: {missing}")
        st.stop()

    num_qubits = control_values["num_qubits"]
    samples = control_values["samples"]
    data_dir = control_values["data_dir"]
    device_choice = control_values["device_choice"]
    q_device = control_values["q_device"]
    batch_size = control_values["batch_size"]
    num_workers = control_values["num_workers"]
    epochs = control_values["epochs"]
    optimizer = control_values["optimizer"]
    lr = control_values["lr"]
    weight_decay = control_values["weight_decay"]
    dropout = control_values["dropout"]
    early_stop_metric = control_values["early_stop_metric"]
    early_stop_patience = control_values["early_stop_patience"]
    scheduler_patience = control_values["scheduler_patience"]
    live_every_batches = control_values["live_every_batches"]
    conv_maps_to_show = control_values["conv_maps_to_show"]
    metrics_rows_visible = control_values["metrics_rows_visible"]
    custom_digits_csv = control_values["custom_digits_csv"]

    args = Args(
        num_qubits=int(num_qubits),
        samples=int(samples),
        data_dir=data_dir,
        device=device_choice,
        q_device=q_device,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        epochs=int(epochs),
        optimizer=optimizer,
        lr=float(lr),
        weight_decay=float(weight_decay),
        dropout=float(dropout),
        early_stop_metric=early_stop_metric,
        early_stop_patience=int(early_stop_patience),
        scheduler_patience=int(scheduler_patience),
        live_every_batches=int(live_every_batches),
        conv_maps_to_show=int(conv_maps_to_show),
    )

    default_digits = list(user_module.PRESETS[args.num_qubits])
    if custom_digits_csv.strip():
        try:
            target_digits = [int(x.strip()) for x in custom_digits_csv.split(",") if x.strip()]
        except ValueError:
            with training_tab:
                st.error("Digit list override must be comma-separated integers.")
            st.stop()
    else:
        target_digits = default_digits

    if len(target_digits) < 2:
        with training_tab:
            st.error("Need at least 2 digits.")
        st.stop()
    if len(set(target_digits)) != len(target_digits):
        with training_tab:
            st.error("Digit list contains duplicates.")
        st.stop()
    if any(d < 0 or d > 9 for d in target_digits):
        with training_tab:
            st.error("Digits must be between 0 and 9.")
        st.stop()

    class_names = [str(d) for d in target_digits]

    with controls_tab:
        st.write("Selected digit subset:", ", ".join(map(str, target_digits)))

    run_training_clicked = st.session_state.start_training
    if run_training_clicked:
        st.session_state.start_training = False

    with summary_tab:
        if results is None:
            st.info("Summary will appear here after a completed run.")
        else:
            render_summary_panel(results, user_module)

    if not run_training_clicked:
        with training_tab:
            st.write("Selected digit subset:", ", ".join(map(str, target_digits)))
            if results is None:
                st.info(
                    "Ready. Use the Controls tab to launch a run. Training progress, live walkthroughs, and epoch history will appear here."
                )
            else:
                final_chart_prefix = f"training-final-run-{results.get('run_id', st.session_state.get('current_run_id', 0))}"
                render_training_history_panel(
                    results["history"],
                    chart_prefix=final_chart_prefix,
                    metrics_rows_visible=int(metrics_rows_visible),
                )

                if st.session_state.live_snapshot_history:
                    st.markdown("### Recent live walkthroughs")
                    st.caption("The most recent captures remain available here after training finishes.")
                    render_live_snapshot_history(
                        st.session_state.live_snapshot_history,
                        class_names=class_names,
                        conv_maps_to_show=args.conv_maps_to_show,
                    )
        st.stop()

    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
        device = user_module.configure_backend(args.device, args.q_device)
    except Exception as exc:
        with training_tab:
            st.error(f"Backend configuration failed: {exc}")
        st.stop()

    if hasattr(user_module, "SET_DETERMINISTIC") and getattr(user_module, "SET_DETERMINISTIC"):
        if hasattr(user_module, "seed_everything"):
            user_module.seed_everything(getattr(user_module, "SEED", 22))

    observables = user_module.build_observables(args.num_qubits)
    if len(observables) < len(target_digits):
        with training_tab:
            st.error(
                f"Need at least as many quantum features as classes. "
                f"Got {len(observables)} observables for {len(target_digits)} classes."
            )
        st.stop()

    loader_device = torch.device("cpu")
    try:
        train_loader, test_loader = user_module.build_dataloaders(
            target_digits=target_digits,
            test_size_pct=user_module.TEST_SIZE_PCT,
            device=loader_device,
            args=args,
        )
    except Exception as exc:
        with training_tab:
            st.error(f"Could not build dataloaders: {exc}")
        st.stop()

    _ = get_loader_dataset(test_loader)

    model = user_module.HybridQNN(
        args.num_qubits,
        user_module.SHIFT,
        args.dropout,
        len(target_digits),
    ).to(device)

    try:
        model.quantum.runner.hamiltonian_names = build_observable_labels(
            args.num_qubits,
            len(model.quantum.runner.hamiltonians),
        )
    except Exception:
        pass

    try:
        st.session_state.training_active = True
        st.session_state.stop_requested = False
        with training_tab:
            st.write("Selected digit subset:", ", ".join(map(str, target_digits)))
            history, final_loss, final_acc, y_true, y_pred, best_epoch, stopped = run_training(
                user_module=user_module,
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                args=args,
                class_names=class_names,
                metrics_rows_visible=int(metrics_rows_visible),
            )

        st.session_state.training_active = False

        if stopped:
            release_training_resources(model)
            finalize_stop_state("Training stopped cleanly. Change parameters and run again.")
            st.session_state.post_run_default_view = "Controls"
            st.rerun()

    except KeyboardInterrupt:
        release_training_resources(model)
        finalize_stop_state("Training stopped. Change parameters and run again.")
        st.session_state.post_run_default_view = "Controls"
        st.rerun()

    release_training_resources(model)

    st.session_state.last_training_results = {
        "history": history,
        "final_loss": final_loss,
        "final_acc": final_acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "best_epoch": best_epoch,
        "target_digits": target_digits,
        "run_id": st.session_state.get("current_run_id", 0),
    }

    finish_training_and_restore_controls(select_summary=True)
    st.rerun()


if __name__ == "__main__":
    app()
