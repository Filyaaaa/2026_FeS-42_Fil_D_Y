import os
import time
import csv
import threading
from datetime import datetime

import numpy as np
import cv2
from flask import Flask, Response, jsonify, send_from_directory
from picamera2 import Picamera2
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

CAP_W, CAP_H = 640, 480
PROC_W = 240
PANEL_W = 1000

MAX_DRAW_PTS = 220

N_FRAMES = 100
GT_WARMUP_SEC = 2.0
GT_MIN_AREA = 180
GT_MIN_PIXELS = 250

JPEG_QUALITY = 85
STREAM_FPS_LIMIT = 12
PREVIEW_EVERY_N = 5

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FONT_SIZE = 20

# -------------------------------
# Параметри методів
# -------------------------------

# ---- Контури
SOBEL_KSIZE = 3
SOBEL_THRESH_SCALE = 0.47

ROBERTS_THRESH_SCALE = 0.52
PREWITT_THRESH_SCALE = 0.47
SCHARR_THRESH_SCALE = 0.43

EDGE_DILATE_ITER = 1

# ---- Кути
FAST_THRESHOLD = 16
FAST_NONMAX = True

HARRIS_BLOCK_SIZE = 2
HARRIS_KSIZE = 3
HARRIS_K = 0.04
HARRIS_THRESH_REL = 0.010

SHI_MAX_CORNERS = 260
SHI_QUALITY = 0.006
SHI_MIN_DISTANCE = 5

# ---- Плями
LOG_SIGMA = 1.10
LOG_THRESH_REL = 0.27

DOG_SIGMA1 = 0.8
DOG_SIGMA2 = 1.7
DOG_THRESH_REL = 0.20

HESS_BLOB_SIGMA = 1.2
HESS_BLOB_THRESH_REL = 0.16

BLOB_MIN_AREA = 14
BLOB_MAX_AREA = 10000

# ---- Хребти
RIDGE_SIGMA = 1.4
RIDGE_THRESH_REL = 0.14

FRANGI_BETA = 0.60
FRANGI_C = 0.45
FRANGI_THRESH_REL = 0.085

SATO_ALPHA = 0.62
SATO_GAMMA = 0.50
SATO_THRESH_REL = 0.11

RIDGE_SIGMAS = [0.9, 1.4, 2.0, 2.8, 3.6]

# ---- HOG
HOG_DET_W = 384
HOG_SCALE = 1.02
HOG_WIN_STRIDE = (4, 4)
HOG_PADDING = (16, 16)
HOG_HIT_THRESHOLD = -0.2
HOG_NMS_THRESHOLD = 0.42
HOG_BOX_PAD = 0.28
HOG_MIN_W = 16
HOG_MIN_H = 32

# ---- Fusion
HOG_GATE_DILATE_K = 13
HOG_GATE_DILATE_ITERS = 2
HOG_BLEND_ALPHA = 0.38
HOG_MIN_PIXELS_TO_GATE = 60


def resize_w(img, width):
    h, w = img.shape[:2]
    if w == width:
        return img
    s = width / w
    return cv2.resize(img, (width, int(h * s)), interpolation=cv2.INTER_AREA)


def ensure_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def stack_2x2(tl, tr, bl, br):
    h = min(tl.shape[0], tr.shape[0], bl.shape[0], br.shape[0])
    w = min(tl.shape[1], tr.shape[1], bl.shape[1], br.shape[1])

    def fit(x):
        return cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)

    top = np.hstack([fit(tl), fit(tr)])
    bot = np.hstack([fit(bl), fit(br)])
    return np.vstack([top, bot])


def load_font():
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, FONT_SIZE)
        except Exception:
            pass
    return ImageFont.load_default()


FONT = load_font()

HOG = cv2.HOGDescriptor()
HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def draw_ua_text(panel_bgr, lines):
    rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)

    for (x, y, text) in lines:
        bbox = d.textbbox((x, y), text, font=FONT)
        pad = 6
        d.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill=(0, 0, 0)
        )
        d.text((x, y), text, font=FONT, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def compute_metrics(tp, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    acc = safe_div(tp, tp + fp + fn)
    return precision, recall, f1, acc


def motion_gt(bg, frame_bgr):
    fg = bg.apply(frame_bgr)
    fg = cv2.medianBlur(fg, 5)
    fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    return fg


def pixel_match(gt_mask, pred_mask):
    gt = gt_mask > 0
    pr = pred_mask > 0
    tp = int(np.logical_and(gt, pr).sum())
    fp = int(np.logical_and(~gt, pr).sum())
    fn = int(np.logical_and(gt, ~pr).sum())
    return tp, fp, fn


def points_to_mask(shape, pts, radius=3):
    mask = np.zeros(shape, dtype=np.uint8)
    for x, y in pts:
        cv2.circle(mask, (int(x), int(y)), radius, 255, -1)
    return mask


def boxes_to_mask(shape, boxes, pad_ratio=0.0):
    mask = np.zeros(shape, dtype=np.uint8)
    h_img, w_img = shape[:2]

    for (x, y, w, h) in boxes:
        pad_w = int(w * pad_ratio)
        pad_h = int(h * pad_ratio)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w_img, x + w + pad_w)
        y2 = min(h_img, y + h + pad_h)

        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask


def normalize_to_u8(arr):
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - mn) / (mx - mn)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def threshold_rel(resp_u8, rel=0.5):
    thr = int(np.clip(resp_u8.max() * rel, 0, 255))
    _, mask = cv2.threshold(resp_u8, thr, 255, cv2.THRESH_BINARY)
    return mask


def clean_binary(mask, open_iter=1, close_iter=1, ksize=3):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    out = mask.copy()
    if open_iter > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=open_iter)
    if close_iter > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=close_iter)
    return out


def draw_points(frame_bgr, pts, color=(0, 0, 255), radius=2):
    vis = frame_bgr.copy()
    for x, y in pts:
        cv2.circle(vis, (int(x), int(y)), radius, color, -1)
    return vis


def mask_to_components_vis(mask, frame_bgr, color=(255, 0, 0), area_min=5, area_max=100000):
    vis = frame_bgr.copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_min or area > area_max:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
        cv2.circle(vis, (int(cx), int(cy)), 2, color, -1)
        kept += 1
    return vis, kept


def hessian_second_derivatives(gray_f, sigma=1.0):
    k = int(max(3, round(sigma * 6)))
    if k % 2 == 0:
        k += 1
    blur = cv2.GaussianBlur(gray_f, (k, k), sigmaX=sigma, sigmaY=sigma)

    dxx = cv2.Sobel(blur, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(blur, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(blur, cv2.CV_32F, 1, 1, ksize=3)
    return dxx, dyy, dxy


def hessian_eigenvalues(dxx, dyy, dxy):
    trace = dxx + dyy
    det = dxx * dyy - dxy * dxy
    tmp = np.sqrt(np.maximum(trace * trace - 4.0 * det, 0.0))

    la = 0.5 * (trace + tmp)
    lb = 0.5 * (trace - tmp)

    l_small = la.copy()
    l_large = lb.copy()

    swap = np.abs(la) > np.abs(lb)
    l_small[swap] = lb[swap]
    l_large[swap] = la[swap]

    return l_small, l_large


def top_response_points(resp, max_pts=MAX_DRAW_PTS, thr_rel=0.01):
    resp = resp.astype(np.float32)
    mx = float(resp.max())
    if mx <= 0:
        return []

    thr = mx * thr_rel
    ys, xs = np.where(resp > thr)
    if len(xs) == 0:
        return []

    vals = resp[ys, xs]
    idx = np.argsort(vals)[::-1][:max_pts]
    pts = [(int(xs[i]), int(ys[i])) for i in idx]
    return pts


def build_hog_boxes_and_mask(frame_bgr, out_shape):
    scaled = resize_w(frame_bgr, HOG_DET_W)
    sh, sw = scaled.shape[:2]

    rects, weights = HOG.detectMultiScale(
        scaled,
        hitThreshold=HOG_HIT_THRESHOLD,
        winStride=HOG_WIN_STRIDE,
        padding=HOG_PADDING,
        scale=HOG_SCALE
    )

    filtered = []
    if len(rects) > 0:
        for (x, y, w, h), score in zip(rects, weights):
            if w < HOG_MIN_W or h < HOG_MIN_H:
                continue
            filtered.append((x, y, w, h, float(score)))

    boxes_xyxy = []
    scores = []
    for x, y, w, h, score in filtered:
        boxes_xyxy.append([x, y, x + w, y + h])
        scores.append(score)

    keep = []
    if len(boxes_xyxy) > 0:
        idxs = cv2.dnn.NMSBoxes(
            bboxes=[[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes_xyxy],
            scores=scores,
            score_threshold=0.0,
            nms_threshold=HOG_NMS_THRESHOLD
        )
        if len(idxs) > 0:
            idxs = np.array(idxs).reshape(-1)
            keep = [filtered[i] for i in idxs]

    scale_back = frame_bgr.shape[1] / float(sw)
    final_boxes = []

    for x, y, w, h, score in keep:
        rx = int(x * scale_back)
        ry = int(y * scale_back)
        rw = int(w * scale_back)
        rh = int(h * scale_back)
        final_boxes.append((rx, ry, rw, rh))

    hog_mask = boxes_to_mask(out_shape, final_boxes, pad_ratio=HOG_BOX_PAD)

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (HOG_GATE_DILATE_K, HOG_GATE_DILATE_K)
    )
    hog_mask = cv2.dilate(hog_mask, k, iterations=HOG_GATE_DILATE_ITERS)

    return final_boxes, hog_mask


def draw_hog_boxes(frame_bgr, boxes, color=(0, 255, 255), label="HOG"):
    vis = frame_bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            vis, label,
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )
    return vis


def fuse_with_hog(base_mask, frame_bgr, method_name="base", color=(0, 255, 255)):
    boxes, hog_mask = build_hog_boxes_and_mask(frame_bgr, base_mask.shape)

    if int((hog_mask > 0).sum()) < HOG_MIN_PIXELS_TO_GATE:
        vis = frame_bgr.copy()
        vis[base_mask > 0] = color
        return base_mask, vis, f"{method_name}+HOG | no_hog_boxes"

    base_u8 = (base_mask > 0).astype(np.uint8) * 255
    hog_u8 = (hog_mask > 0).astype(np.uint8) * 255

    inter = cv2.bitwise_and(base_u8, hog_u8)
    union_soft = cv2.addWeighted(base_u8, 1.0 - HOG_BLEND_ALPHA, hog_u8, HOG_BLEND_ALPHA, 0)

    fused = cv2.bitwise_or(inter, cv2.threshold(union_soft, 145, 255, cv2.THRESH_BINARY)[1])
    fused = clean_binary(fused, open_iter=1, close_iter=2, ksize=3)

    vis = frame_bgr.copy()
    vis = draw_hog_boxes(vis, boxes, color=(0, 255, 255), label="HOG")
    vis[fused > 0] = color

    return fused, vis, f"{method_name}+HOG | boxes={len(boxes)}"


# -------------------------------
# Базові методи
# -------------------------------

def sobel_method(gray, frame_bgr):
    g = cv2.GaussianBlur(gray, (3, 3), 0.8)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=SOBEL_KSIZE)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=SOBEL_KSIZE)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = normalize_to_u8(mag)
    mask = threshold_rel(mag_u8, SOBEL_THRESH_SCALE)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=EDGE_DILATE_ITER)
    vis = ensure_bgr(mask)
    return mask, vis, f"Sobel | k={SOBEL_KSIZE} thr_rel={SOBEL_THRESH_SCALE}"


def roberts_method(gray, frame_bgr):
    g = cv2.GaussianBlur(gray, (3, 3), 0.8).astype(np.float32)

    kx = np.array([[1, 0],
                   [0, -1]], dtype=np.float32)
    ky = np.array([[0, 1],
                   [-1, 0]], dtype=np.float32)

    rx = cv2.filter2D(g, cv2.CV_32F, kx)
    ry = cv2.filter2D(g, cv2.CV_32F, ky)
    mag = cv2.magnitude(rx, ry)
    mag_u8 = normalize_to_u8(mag)

    mask = threshold_rel(mag_u8, ROBERTS_THRESH_SCALE)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=EDGE_DILATE_ITER)
    vis = ensure_bgr(mask)
    return mask, vis, f"Roberts | thr_rel={ROBERTS_THRESH_SCALE}"


def prewitt_method(gray, frame_bgr):
    g = cv2.GaussianBlur(gray, (3, 3), 0.8).astype(np.float32)

    kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]], dtype=np.float32)

    px = cv2.filter2D(g, cv2.CV_32F, kx)
    py = cv2.filter2D(g, cv2.CV_32F, ky)
    mag = cv2.magnitude(px, py)
    mag_u8 = normalize_to_u8(mag)

    mask = threshold_rel(mag_u8, PREWITT_THRESH_SCALE)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=EDGE_DILATE_ITER)
    vis = ensure_bgr(mask)
    return mask, vis, f"Prewitt | thr_rel={PREWITT_THRESH_SCALE}"


def scharr_method(gray, frame_bgr):
    g = cv2.GaussianBlur(gray, (3, 3), 0.8)
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = normalize_to_u8(mag)

    mask = threshold_rel(mag_u8, SCHARR_THRESH_SCALE)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=EDGE_DILATE_ITER)
    vis = ensure_bgr(mask)
    return mask, vis, f"Scharr | thr_rel={SCHARR_THRESH_SCALE}"


def fast_method(gray, frame_bgr):
    fast = cv2.FastFeatureDetector_create(
        threshold=FAST_THRESHOLD,
        nonmaxSuppression=FAST_NONMAX
    )
    kps = fast.detect(gray, None)
    pts = []
    if kps:
        kps = sorted(kps, key=lambda k: k.response, reverse=True)[:MAX_DRAW_PTS]
        pts = [(int(k.pt[0]), int(k.pt[1])) for k in kps]

    mask = points_to_mask(gray.shape, pts, radius=3)
    vis = draw_points(frame_bgr, pts, color=(0, 0, 255), radius=1)

    return mask, vis, f"FAST | thr={FAST_THRESHOLD} nonmax={FAST_NONMAX} pts={len(pts)}"


def harris_method(gray, frame_bgr):
    g = np.float32(gray)
    resp = cv2.cornerHarris(g, HARRIS_BLOCK_SIZE, HARRIS_KSIZE, HARRIS_K)
    resp = cv2.dilate(resp, None)

    pts = top_response_points(resp, max_pts=MAX_DRAW_PTS, thr_rel=HARRIS_THRESH_REL)
    mask = points_to_mask(gray.shape, pts, radius=3)
    vis = draw_points(frame_bgr, pts, color=(0, 255, 255), radius=1)

    return mask, vis, (
        f"Harris | block={HARRIS_BLOCK_SIZE} ksize={HARRIS_KSIZE} "
        f"k={HARRIS_K} thr_rel={HARRIS_THRESH_REL} pts={len(pts)}"
    )


def shi_tomasi_method(gray, frame_bgr):
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=SHI_MAX_CORNERS,
        qualityLevel=SHI_QUALITY,
        minDistance=SHI_MIN_DISTANCE
    )

    pts = []
    if corners is not None:
        for c in corners:
            x, y = c.ravel()
            pts.append((int(x), int(y)))

    mask = points_to_mask(gray.shape, pts, radius=3)
    vis = draw_points(frame_bgr, pts, color=(255, 255, 0), radius=1)

    return mask, vis, (
        f"Shi-Tomasi | maxC={SHI_MAX_CORNERS} q={SHI_QUALITY} "
        f"minDist={SHI_MIN_DISTANCE} pts={len(pts)}"
    )


def log_blob_method(gray, frame_bgr):
    gray_f = gray.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray_f, (0, 0), LOG_SIGMA)
    log_resp = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    log_abs = np.abs(log_resp)
    resp_u8 = normalize_to_u8(log_abs)

    mask = threshold_rel(resp_u8, LOG_THRESH_REL)
    mask = clean_binary(mask, open_iter=1, close_iter=1, ksize=3)

    vis, kept = mask_to_components_vis(
        mask, frame_bgr, color=(255, 0, 0),
        area_min=BLOB_MIN_AREA, area_max=BLOB_MAX_AREA
    )

    return mask, vis, f"LoG Blob | sigma={LOG_SIGMA} thr_rel={LOG_THRESH_REL} blobs={kept}"


def dog_blob_method(gray, frame_bgr):
    gray_f = gray.astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(gray_f, (0, 0), DOG_SIGMA1)
    g2 = cv2.GaussianBlur(gray_f, (0, 0), DOG_SIGMA2)
    dog = g1 - g2
    dog_abs = np.abs(dog)
    resp_u8 = normalize_to_u8(dog_abs)

    mask = threshold_rel(resp_u8, DOG_THRESH_REL)
    mask = clean_binary(mask, open_iter=1, close_iter=1, ksize=3)

    vis, kept = mask_to_components_vis(
        mask, frame_bgr, color=(0, 255, 0),
        area_min=BLOB_MIN_AREA, area_max=BLOB_MAX_AREA
    )

    return mask, vis, (
        f"DoG Blob | s1={DOG_SIGMA1} s2={DOG_SIGMA2} "
        f"thr_rel={DOG_THRESH_REL} blobs={kept}"
    )


def hessian_blob_method(gray, frame_bgr):
    gray_f = gray.astype(np.float32) / 255.0
    dxx, dyy, dxy = hessian_second_derivatives(gray_f, sigma=HESS_BLOB_SIGMA)
    det_h = dxx * dyy - dxy * dxy
    det_abs = np.abs(det_h)
    resp_u8 = normalize_to_u8(det_abs)

    mask = threshold_rel(resp_u8, HESS_BLOB_THRESH_REL)
    mask = clean_binary(mask, open_iter=1, close_iter=1, ksize=3)

    vis, kept = mask_to_components_vis(
        mask, frame_bgr, color=(0, 128, 255),
        area_min=BLOB_MIN_AREA, area_max=BLOB_MAX_AREA
    )

    return mask, vis, (
        f"Hessian Blob | sigma={HESS_BLOB_SIGMA} "
        f"thr_rel={HESS_BLOB_THRESH_REL} blobs={kept}"
    )


def hessian_ridge_method(gray, frame_bgr):
    gray_f = gray.astype(np.float32) / 255.0
    dxx, dyy, dxy = hessian_second_derivatives(gray_f, sigma=RIDGE_SIGMA)
    l_small, l_large = hessian_eigenvalues(dxx, dyy, dxy)

    ridge = np.abs(l_large)
    resp_u8 = normalize_to_u8(ridge)

    mask = threshold_rel(resp_u8, RIDGE_THRESH_REL)
    mask = clean_binary(mask, open_iter=1, close_iter=1, ksize=3)

    vis, kept = mask_to_components_vis(mask, frame_bgr, color=(255, 0, 255), area_min=10, area_max=100000)
    return mask, vis, f"Hessian Ridge | sigma={RIDGE_SIGMA} thr_rel={RIDGE_THRESH_REL} comps={kept}"


def frangi_like_method(gray, frame_bgr):
    gray_f = gray.astype(np.float32) / 255.0
    best = np.zeros_like(gray_f, dtype=np.float32)

    eps = 1e-9

    for sigma in RIDGE_SIGMAS:
        dxx, dyy, dxy = hessian_second_derivatives(gray_f, sigma=sigma)
        l_small, l_large = hessian_eigenvalues(dxx, dyy, dxy)

        abs_small = np.abs(l_small)
        abs_large = np.abs(l_large)

        rb = abs_small / (abs_large + eps)
        s2 = l_small * l_small + l_large * l_large

        vessel = np.exp(-(rb * rb) / (2.0 * FRANGI_BETA * FRANGI_BETA)) * \
                 (1.0 - np.exp(-s2 / (2.0 * FRANGI_C * FRANGI_C)))

        vessel[abs_large < eps] = 0.0
        vessel[l_large > 0] = 0.0

        best = np.maximum(best, vessel)

    best = cv2.GaussianBlur(best, (0, 0), 1.0)
    best = np.power(np.clip(best, 0.0, None), 0.7)

    resp_u8 = normalize_to_u8(best)
    mask = threshold_rel(resp_u8, FRANGI_THRESH_REL)
    mask = clean_binary(mask, open_iter=1, close_iter=2, ksize=3)

    vis, kept = mask_to_components_vis(
        mask,
        frame_bgr,
        color=(0, 255, 255),
        area_min=10,
        area_max=100000
    )

    return mask, vis, (
        f"Frangi-like MS | sigmas={RIDGE_SIGMAS} "
        f"beta={FRANGI_BETA} c={FRANGI_C} "
        f"thr_rel={FRANGI_THRESH_REL} comps={kept}"
    )


def sato_like_method(gray, frame_bgr):
    gray_f = gray.astype(np.float32) / 255.0
    best = np.zeros_like(gray_f, dtype=np.float32)

    for sigma in RIDGE_SIGMAS:
        dxx, dyy, dxy = hessian_second_derivatives(gray_f, sigma=sigma)
        l_small, l_large = hessian_eigenvalues(dxx, dyy, dxy)

        abs_small = np.abs(l_small)
        abs_large = np.abs(l_large)

        line_resp = np.maximum(0.0, abs_large - SATO_ALPHA * abs_small)
        line_resp = line_resp * np.exp(-(abs_small * abs_small + abs_large * abs_large) / (2.0 * SATO_GAMMA * SATO_GAMMA))

        line_resp[l_large > 0] = 0.0

        best = np.maximum(best, line_resp)

    best = cv2.GaussianBlur(best, (0, 0), 1.0)
    best = np.power(np.clip(best, 0.0, None), 0.7)

    resp_u8 = normalize_to_u8(best)
    mask = threshold_rel(resp_u8, SATO_THRESH_REL)
    mask = clean_binary(mask, open_iter=1, close_iter=2, ksize=3)

    vis, kept = mask_to_components_vis(
        mask,
        frame_bgr,
        color=(0, 200, 120),
        area_min=10,
        area_max=100000
    )

    return mask, vis, (
        f"Sato-like MS | sigmas={RIDGE_SIGMAS} "
        f"alpha={SATO_ALPHA} gamma={SATO_GAMMA} "
        f"thr_rel={SATO_THRESH_REL} comps={kept}"
    )


# -------------------------------
# HOG окремо
# -------------------------------

def hog_people_method(gray, frame_bgr):
    boxes, hog_mask = build_hog_boxes_and_mask(frame_bgr, gray.shape)
    vis = draw_hog_boxes(frame_bgr, boxes, color=(0, 255, 255), label="HOG")
    return hog_mask, vis, f"HOG People | boxes={len(boxes)}"


# -------------------------------
# Fusion HOG + всі 12 методів
# -------------------------------

def sobel_hog_method(gray, frame_bgr):
    base_mask, _, _ = sobel_method(gray, frame_bgr)
    return fuse_with_hog(base_mask, frame_bgr, "Sobel", (255, 255, 255))


def roberts_hog_method(gray, frame_bgr):
    base_mask, _, _ = roberts_method(gray, frame_bgr)
    return fuse_with_hog(base_mask, frame_bgr, "Roberts", (220, 220, 220))


def prewitt_hog_method(gray, frame_bgr):
    base_mask, _, _ = prewitt_method(gray, frame_bgr)
    return fuse_with_hog(base_mask, frame_bgr, "Prewitt", (200, 255, 200))


def scharr_hog_method(gray, frame_bgr):
    base_mask, _, _ = scharr_method(gray, frame_bgr)
    return fuse_with_hog(base_mask, frame_bgr, "Scharr", (255, 200, 200))


def fast_hog_method(gray, frame_bgr):
    base_mask, _, _ = fast_method(gray, frame_bgr)
    return fuse_with_hog(base_mask, frame_bgr, "FAST", (0, 0, 255))


def harris_hog_method(gray, frame_bgr):
    base_mask, _, _ = harris_method(gray, frame_bgr)
    return fuse_with_hog(base_mask, frame_bgr, "Harris", (0, 255, 255))


def shi_tomasi_hog_method(gray, frame_bgr):
    base_mask, _, _ = shi_tomasi_method(gray, frame_bgr)
    return fuse_with_hog(base_mask, frame_bgr, "Shi-Tomasi", (255, 255, 0))


def log_hog_method(gray, frame_bgr):
    base_mask, _, _ = log_blob_method(gray, frame_bgr)
    return fuse_with_hog(base_mask, frame_bgr, "LoG", (255, 0, 0))


def dog_hog_method(gray, frame_bgr):
    base_mask, _, _ = dog_blob_method(gray, frame_bgr)
    return fuse_with_hog(base_mask, frame_bgr, "DoG", (0, 255, 0))


def hessian_blob_hog_method(gray, frame_bgr):
    base_mask, _, _ = hessian_blob_method(gray, frame_bgr)
    return fuse_with_hog(base_mask, frame_bgr, "Hessian Blob", (0, 128, 255))


def hessian_ridge_hog_method(gray, frame_bgr):
    base_mask, _, _ = hessian_ridge_method(gray, frame_bgr)
    return fuse_with_hog(base_mask, frame_bgr, "Hessian Ridge", (255, 0, 255))


def frangi_hog_method(gray, frame_bgr):
    base_mask, _, _ = frangi_like_method(gray, frame_bgr)
    return fuse_with_hog(base_mask, frame_bgr, "Frangi-like", (0, 255, 255))


def sato_hog_method(gray, frame_bgr):
    base_mask, _, _ = sato_like_method(gray, frame_bgr)
    return fuse_with_hog(base_mask, frame_bgr, "Sato-like", (0, 200, 120))


GROUPS = [
    {
        "group_name": "Контури",
        "methods": [
            ("Sobel+HOG", sobel_hog_method),
            ("Roberts+HOG", roberts_hog_method),
            ("Prewitt+HOG", prewitt_hog_method),
            ("Scharr+HOG", scharr_hog_method),
        ],
    },
    {
        "group_name": "Кути",
        "methods": [
            ("FAST+HOG", fast_hog_method),
            ("Harris+HOG", harris_hog_method),
            ("Shi-Tomasi+HOG", shi_tomasi_hog_method),
            ("HOG", hog_people_method),
        ],
    },
    {
        "group_name": "Плями",
        "methods": [
            ("LoG+HOG", log_hog_method),
            ("DoG+HOG", dog_hog_method),
            ("Hessian Blob+HOG", hessian_blob_hog_method),
            ("HOG", hog_people_method),
        ],
    },
    {
        "group_name": "Хребти",
        "methods": [
            ("Hessian Ridge+HOG", hessian_ridge_hog_method),
            ("Frangi+HOG", frangi_hog_method),
            ("Sato+HOG", sato_hog_method),
            ("HOG", hog_people_method),
        ],
    },
]


def save_plots_and_csv(summary_rows, run_dir):
    names = [r["name"] for r in summary_rows]
    P = [r["P"] for r in summary_rows]
    R = [r["R"] for r in summary_rows]
    F1 = [r["F1"] for r in summary_rows]
    ACC = [r["ACC"] for r in summary_rows]
    MS = [r["mean_ms"] for r in summary_rows]

    x = np.arange(len(names))
    w = 0.22

    plt.figure(figsize=(16, 7))
    plt.bar(x - w, P, width=w, label="Precision")
    plt.bar(x, R, width=w, label="Recall")
    plt.bar(x + w, F1, width=w, label="F1")
    plt.xticks(x, names, rotation=15, ha="right", fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(0, 1.0)
    plt.title("Precision / Recall / F1", fontsize=16)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "metrics_prf.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(16, 7))
    plt.bar(names, MS)
    plt.xticks(rotation=15, ha="right", fontsize=11)
    plt.yticks(fontsize=11)
    plt.title("Середній час методу (мс)", fontsize=16)
    plt.ylabel("ms", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "metrics_time_ms.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(16, 7))
    plt.bar(names, ACC)
    plt.xticks(rotation=15, ha="right", fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(0, 1.0)
    plt.title("Object-Accuracy = TP / (TP + FP + FN)", fontsize=16)
    plt.ylabel("Acc", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "metrics_accuracy.png"), dpi=220)
    plt.close()

    csv_path = os.path.join(run_dir, "metrics_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["Group", "Method", "FullName", "Precision", "Recall", "F1", "Acc", "Mean_ms", "TP", "FP", "FN"])
        for r in summary_rows:
            wri.writerow([
                r["group"], r["method"], r["name"], r["P"], r["R"], r["F1"], r["ACC"],
                r["mean_ms"], r["tp"], r["fp"], r["fn"]
            ])


def compose_group_panel(group_name, previews, gt_mask, frame_bgr):
    if len(previews) >= 4:
        tl = previews[0]["vis"]
        tr = previews[1]["vis"]
        bl = previews[2]["vis"]
        br = previews[3]["vis"]
        legend = "TL/TR/BL/BR: " + " | ".join([p["name"] for p in previews[:4]])
    elif len(previews) == 3:
        tl = previews[0]["vis"]
        tr = previews[1]["vis"]
        bl = previews[2]["vis"]
        br = ensure_bgr(gt_mask)
        legend = f"TL/TR/BL: {previews[0]['name']} | {previews[1]['name']} | {previews[2]['name']} | BR: GT mask"
    elif len(previews) == 2:
        tl = previews[0]["vis"]
        tr = previews[1]["vis"]
        bl = frame_bgr
        br = ensure_bgr(gt_mask)
        legend = f"TL/TR: {previews[0]['name']} | {previews[1]['name']} | BL: Original | BR: GT mask"
    else:
        tl = frame_bgr
        tr = ensure_bgr(gt_mask)
        bl = previews[0]["vis"]
        br = previews[0]["vis"]
        legend = f"TL: Original | TR: GT mask | BL/BR: {previews[0]['name']}"

    panel = stack_2x2(tl, tr, bl, br)
    panel = resize_w(panel, PANEL_W)
    return panel, legend


class Runner:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.last_panel_path = os.path.join(SCRIPT_DIR, "last_panel.png")

        self.running = False
        self.done = False
        self.error = None

        self.current_group = "-"
        self.group_index = 0
        self.group_total = len(GROUPS)

        self.current_method = "-"
        self.method_index = 0
        self.method_total = 0

        self.collected = 0
        self.target = N_FRAMES
        self.warmup_done = False
        self.gt_pixels = 0
        self.fps_s = 0.0

        self.summary_rows = None
        self.run_dir = None
        self.run_name = None

        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.running = True
        self.done = False
        self.error = None

        self.current_group = "-"
        self.group_index = 0
        self.current_method = "-"
        self.method_index = 0
        self.method_total = 0

        self.collected = 0
        self.gt_pixels = 0
        self.fps_s = 0.0
        self.summary_rows = None

        self.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(RUNS_DIR, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _set_latest(self, panel_bgr):
        ok, buf = cv2.imencode(
            ".jpg", panel_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            return
        with self.lock:
            self.latest_jpeg = buf.tobytes()

    def _run(self):
        picam2 = None
        try:
            picam2 = Picamera2()
            cfg = picam2.create_video_configuration(
                main={"size": (CAP_W, CAP_H), "format": "RGB888"}
            )
            picam2.configure(cfg)
            picam2.start()

            bg = cv2.createBackgroundSubtractorMOG2(
                history=200,
                varThreshold=24,
                detectShadows=False
            )
            warmup_start = time.perf_counter()

            summary_rows = []
            prev = time.perf_counter()
            frame_id = 0

            for gidx, group in enumerate(GROUPS, start=1):
                if self.stop_event.is_set():
                    break

                group_name = group["group_name"]
                methods = group["methods"]

                self.current_group = group_name
                self.group_index = gidx
                self.method_total = len(methods)
                self.collected = 0

                method_stats = {}
                for midx, (mname, _) in enumerate(methods, start=1):
                    full_name = f"{group_name}: {mname}"
                    method_stats[full_name] = {
                        "group": group_name,
                        "method": mname,
                        "name": full_name,
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "ms": [],
                    }

                while not self.stop_event.is_set():
                    frame = picam2.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    frame = cv2.flip(frame, 1)
                    frame = resize_w(frame, PROC_W)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    now = time.perf_counter()
                    dt = now - prev
                    prev = now
                    fps = (1.0 / dt) if dt > 0 else 0.0
                    self.fps_s = 0.9 * self.fps_s + 0.1 * fps

                    self.warmup_done = (time.perf_counter() - warmup_start) >= GT_WARMUP_SEC

                    gt_mask = motion_gt(bg, frame)
                    self.gt_pixels = int((gt_mask > 0).sum())

                    previews = []

                    for midx, (mname, mfunc) in enumerate(methods, start=1):
                        self.current_method = mname
                        self.method_index = midx

                        t0 = time.perf_counter()
                        pred_mask, vis, method_info = mfunc(gray, frame)
                        t1 = time.perf_counter()

                        full_name = f"{group_name}: {mname}"

                        if self.warmup_done and self.gt_pixels >= GT_MIN_PIXELS:
                            tp, fp, fn = pixel_match(gt_mask, pred_mask)
                            method_stats[full_name]["tp"] += tp
                            method_stats[full_name]["fp"] += fp
                            method_stats[full_name]["fn"] += fn
                            method_stats[full_name]["ms"].append((t1 - t0) * 1000.0)

                        previews.append({
                            "name": mname,
                            "vis": vis,
                            "mask": pred_mask,
                            "info": method_info
                        })

                    if self.warmup_done and self.gt_pixels >= GT_MIN_PIXELS:
                        self.collected += 1

                    panel, legend = compose_group_panel(group_name, previews, gt_mask, frame)

                    info_line = " | ".join([p["name"] for p in previews])
                    if len(info_line) > 120:
                        info_line = info_line[:117] + "..."

                    lines = [
                        (15, 12, f"Група {self.group_index}/{self.group_total}: {group_name}"),
                        (15, 42, f"Кадр {self.collected}/{self.target} | warmup={'OK' if self.warmup_done else '...'} | GT pixels={self.gt_pixels} | FPS~{self.fps_s:.1f}"),
                        (15, 72, f"Методи групи: {info_line}"),
                        (15, 102, legend),
                    ]
                    panel = draw_ua_text(panel, lines)
                    self._set_latest(panel)

                    if frame_id % PREVIEW_EVERY_N == 0:
                        try:
                            safe_name = group_name.replace(" ", "_").replace("/", "_")
                            cv2.imwrite(self.last_panel_path, panel)
                            cv2.imwrite(os.path.join(self.run_dir, f"preview_{safe_name}.png"), panel)
                        except Exception:
                            pass

                    if self.collected >= self.target:
                        break

                    frame_id += 1

                    if STREAM_FPS_LIMIT > 0:
                        time.sleep(max(0.0, (1.0 / STREAM_FPS_LIMIT) - 0.001))

                for _, stats in method_stats.items():
                    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
                    P, R, F1, ACC = compute_metrics(tp, fp, fn)
                    mean_ms = float(np.mean(stats["ms"])) if stats["ms"] else 0.0

                    summary_rows.append({
                        "group": stats["group"],
                        "method": stats["method"],
                        "name": stats["name"],
                        "P": P,
                        "R": R,
                        "F1": F1,
                        "ACC": ACC,
                        "mean_ms": mean_ms,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn
                    })

            self.summary_rows = summary_rows
            save_plots_and_csv(summary_rows, self.run_dir)

            self.done = not self.stop_event.is_set()
            self.running = False

        except Exception as e:
            self.error = str(e)
            self.running = False
            self.done = False
        finally:
            try:
                if picam2 is not None:
                    picam2.stop()
            except Exception:
                pass


app = Flask(__name__)
runner = Runner()


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Тестування методів по черзі</title>
  <style>
    body{margin:0;background:#0b0d12;color:#fff;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial}
    .wrap{max-width:1180px;margin:0 auto;padding:28px 18px}
    h1{font-size:40px;margin:0 0 8px;text-align:center}
    .sub{opacity:.85;text-align:center;margin-bottom:18px}
    .status{display:flex;gap:10px;align-items:center;justify-content:center;margin:12px 0 18px}
    .pill{display:inline-flex;gap:10px;align-items:center;background:#111827;border:1px solid #243047;border-radius:999px;padding:10px 14px}
    .dot{width:10px;height:10px;border-radius:999px;background:#22c55e}
    .dot.wait{background:#f59e0b}
    .dot.err{background:#ef4444}
    .btns{display:flex;gap:12px;justify-content:center;margin:14px 0 22px;flex-wrap:wrap}
    button{background:#b00000;border:0;color:#fff;font-weight:700;border-radius:12px;padding:14px 26px;font-size:18px;cursor:pointer}
    button.secondary{background:#1f2937}
    .panelWrap{display:flex;justify-content:center}
    .panel{background:#0f172a;border:1px solid #23314a;border-radius:18px;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.35)}
    img{display:block;max-width:100%;height:auto}
    a{color:#9ca3af}
    .hint{max-width:1050px;margin:0 auto 18px;background:#111827;border:1px solid #243047;border-radius:16px;padding:14px}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Тестування методів по черзі</h1>
  <div class="sub">Один запуск = усі групи по черзі. Для кожної групи збирається 100 валідних кадрів.</div>

  <div class="hint">
    GT рахується по всій картині через маску руху. Методи всередині однієї групи тестуються на тих самих кадрах.
  </div>

  <div class="status">
    <div class="pill">
      <div id="dot" class="dot wait"></div>
      <div id="st">Очікування запуску…</div>
    </div>
  </div>

  <div class="btns">
    <button onclick="startRun()">Запустити</button>
    <button onclick="stopRun()">Зупинити</button>
    <button class="secondary" id="resBtn" onclick="openResults()" style="display:none">Відкрити результати</button>
  </div>

  <div class="panelWrap">
    <div class="panel">
      <img id="panel" src="/panel_feed" />
    </div>
  </div>

  <div class="sub" style="margin-top:14px">
    Останній preview: <a href="/file/last_panel.png">last_panel.png</a> |
    Результати: <a href="/results">/results</a>
  </div>
</div>

<script>
async function poll(){
  try{
    const r = await fetch('/api/status', {cache:'no-store'});
    const j = await r.json();
    const dot = document.getElementById('dot');
    const st = document.getElementById('st');
    const resBtn = document.getElementById('resBtn');

    if(j.error){
      dot.className='dot err';
      st.textContent = 'Помилка: ' + j.error;
      resBtn.style.display='none';
      return;
    }

    if(!j.running && !j.done){
      dot.className='dot wait';
      st.textContent = 'Очікування запуску…';
      resBtn.style.display='none';
      return;
    }

    if(j.done){
      dot.className='dot';
      st.textContent = '✅ Завершено | run=' + j.run_name;
      resBtn.style.display='inline-block';
      return;
    }

    dot.className = 'dot wait';
    st.textContent =
      'Група ' + j.group_index + '/' + j.group_total +
      ' | ' + j.current_group +
      ' | кадрів: ' + j.collected + '/' + j.target +
      ' | GT pixels=' + j.gt_pixels +
      ' | FPS~' + j.fps.toFixed(1);

    resBtn.style.display='none';
  }catch(e){}
  setTimeout(poll, 700);
}
poll();

async function startRun(){
  const r = await fetch('/api/start', {method:'POST'});
  const j = await r.json();
  if(!j.ok){
    alert(j.error || 'Не вдалося запустити');
    return;
  }
  document.getElementById('panel').src = '/panel_feed?ts=' + Date.now();
}

async function stopRun(){
  await fetch('/api/stop', {method:'POST'});
  setTimeout(()=>location.reload(), 500);
}

function openResults(){
  window.location.href='/results';
}
</script>
</body>
</html>
"""


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/")
def index():
    return HTML


@app.route("/api/start", methods=["POST"])
def api_start():
    if runner.running:
        return jsonify({"ok": False, "error": "Запуск уже виконується"})
    runner.start()
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    return jsonify({
        "running": runner.running,
        "done": runner.done,
        "error": runner.error,
        "collected": runner.collected,
        "target": runner.target,
        "warmup_done": runner.warmup_done,
        "gt_pixels": runner.gt_pixels,
        "fps": runner.fps_s,
        "current_group": runner.current_group,
        "group_index": runner.group_index,
        "group_total": runner.group_total,
        "current_method": runner.current_method,
        "method_index": runner.method_index,
        "method_total": runner.method_total,
        "run_name": runner.run_name,
    })


@app.route("/api/stop", methods=["POST"])
def api_stop():
    runner.stop()
    return jsonify({"ok": True})


def mjpeg_stream():
    last = None
    while True:
        if runner.error is not None:
            break
        with runner.lock:
            frame = runner.latest_jpeg
        if frame and frame != last:
            last = frame
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        if runner.done and not runner.running:
            time.sleep(0.3)
        else:
            time.sleep(0.03)


@app.route("/panel_feed")
def panel_feed():
    return Response(
        mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/results")
def results():
    rows = runner.summary_rows
    if not rows:
        return """
        <html><body style="font-family:system-ui;background:#0b0d12;color:#fff;padding:24px">
        <h2>Результати ще не готові.</h2>
        <p><a style="color:#9ca3af" href="/">Назад</a></p>
        </body></html>
        """

    run_name = runner.run_name

    def fmt(x):
        return f"{x:.3f}"

    tr = ""
    for r in rows:
        tr += (
            f"<tr>"
            f"<td>{r['group']}</td>"
            f"<td>{r['method']}</td>"
            f"<td>{fmt(r['P'])}</td>"
            f"<td>{fmt(r['R'])}</td>"
            f"<td>{fmt(r['F1'])}</td>"
            f"<td>{fmt(r['ACC'])}</td>"
            f"<td>{r['mean_ms']:.2f}</td>"
            f"<td>{r['tp']}</td>"
            f"<td>{r['fp']}</td>"
            f"<td>{r['fn']}</td>"
            f"</tr>"
        )

    return f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Results</title>
      <style>
        body{{font-family:system-ui;background:#0b0d12;color:#fff;padding:24px}}
        table{{border-collapse:collapse;width:100%;max-width:1100px;font-size:15px}}
        th,td{{border:1px solid #23314a;padding:8px 10px;text-align:center}}
        th{{background:#111827}}
        a{{color:#9ca3af}}
        .imgs{{display:flex;flex-direction:column;gap:18px;max-width:1100px}}
        .imgs img{{width:100%;border-radius:14px;border:1px solid #23314a}}
        .top{{margin-bottom:18px}}
      </style>
    </head>
    <body>
      <div class="top">
        <h2>✅ Результати запуску: {run_name}</h2>
        <p><a href="/">← Назад</a></p>
      </div>

      <table>
        <tr>
          <th>Group</th>
          <th>Method</th>
          <th>P</th>
          <th>R</th>
          <th>F1</th>
          <th>Acc</th>
          <th>Mean ms</th>
          <th>TP</th>
          <th>FP</th>
          <th>FN</th>
        </tr>
        {tr}
      </table>

      <h3 style="margin-top:24px">Графіки</h3>
      <div class="imgs">
        <img src="/runfile/{run_name}/metrics_prf.png" />
        <img src="/runfile/{run_name}/metrics_time_ms.png" />
        <img src="/runfile/{run_name}/metrics_accuracy.png" />
      </div>

      <p style="margin-top:18px">
        CSV:
        <a href="/runfile/{run_name}/metrics_summary.csv">metrics_summary.csv</a>
      </p>
    </body>
    </html>
    """


@app.route("/file/<path:fname>")
def file_get(fname):
    return send_from_directory(SCRIPT_DIR, fname, as_attachment=False)


@app.route("/runfile/<run_name>/<path:fname>")
def run_file_get(run_name, fname):
    run_dir = os.path.join(RUNS_DIR, run_name)
    return send_from_directory(run_dir, fname, as_attachment=False)


def main():
    print("Запуск Flask...")
    print("Відкрий у браузері:")
    print("  http://<IP_RPI>:5000")
    print(f"last_panel.png: {os.path.join(SCRIPT_DIR, 'last_panel.png')}")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()