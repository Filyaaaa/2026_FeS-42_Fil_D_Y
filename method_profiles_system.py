import os
import time
import csv
import threading
import numpy as np
import cv2
from flask import Flask, Response, jsonify, send_from_directory, request
from picamera2 import Picamera2
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CAP_W, CAP_H = 640, 480
PROC_W = 240
PANEL_W = 900

HEAVY_PERIOD = 8
MAX_DRAW_PTS = 220

N_FRAMES = 100
MATCH_DIST_PX = 18

GT_WARMUP_SEC = 2.0
GT_MIN_AREA = 180
GT_TOPK = 3

MIN_OBJ_AREA = 80

JPEG_QUALITY = 85
STREAM_FPS_LIMIT = 12
PREVIEW_EVERY_N = 5

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FONT_SIZE = 20


# =========================
# ПРОФІЛІ ПАРАМЕТРІВ МЕТОДІВ
# =========================
METHOD_PROFILES = {
    "balanced": {
        "CANNY_T1": 70,
        "CANNY_T2": 140,

        "FAST_THRESHOLD": 22,
        "FAST_NONMAX": True,

        "BLOB_MIN_AREA": 80,
        "BLOB_MAX_AREA": 25000,
        "BLOB_MIN_THRESHOLD": 50,

        "BH_KERNEL": 9,
        "BH_ITERATIONS": 1,
    },

    "strict_clean": {
        "CANNY_T1": 110,
        "CANNY_T2": 220,

        "FAST_THRESHOLD": 35,
        "FAST_NONMAX": True,

        "BLOB_MIN_AREA": 160,
        "BLOB_MAX_AREA": 22000,
        "BLOB_MIN_THRESHOLD": 80,

        "BH_KERNEL": 13,
        "BH_ITERATIONS": 2,
    },

    "sensitive_recall": {
        "CANNY_T1": 45,
        "CANNY_T2": 110,

        "FAST_THRESHOLD": 14,
        "FAST_NONMAX": True,

        "BLOB_MIN_AREA": 40,
        "BLOB_MAX_AREA": 35000,
        "BLOB_MIN_THRESHOLD": 20,

        "BH_KERNEL": 7,
        "BH_ITERATIONS": 1,
    },

    "blob_focus": {
        "CANNY_T1": 90,
        "CANNY_T2": 190,

        "FAST_THRESHOLD": 28,
        "FAST_NONMAX": True,

        "BLOB_MIN_AREA": 120,
        "BLOB_MAX_AREA": 12000,
        "BLOB_MIN_THRESHOLD": 40,

        "BH_KERNEL": 11,
        "BH_ITERATIONS": 1,
    },

    "fast_dense": {
        "CANNY_T1": 70,
        "CANNY_T2": 150,

        "FAST_THRESHOLD": 10,
        "FAST_NONMAX": False,

        "BLOB_MIN_AREA": 80,
        "BLOB_MAX_AREA": 25000,
        "BLOB_MIN_THRESHOLD": 50,

        "BH_KERNEL": 9,
        "BH_ITERATIONS": 1,
    },

    "blackhat_strong": {
        "CANNY_T1": 100,
        "CANNY_T2": 200,

        "FAST_THRESHOLD": 26,
        "FAST_NONMAX": True,

        "BLOB_MIN_AREA": 100,
        "BLOB_MAX_AREA": 25000,
        "BLOB_MIN_THRESHOLD": 60,

        "BH_KERNEL": 15,
        "BH_ITERATIONS": 2,
    },
}


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


def draw_ua_text(panel_bgr, lines):
    rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)

    for (x, y, text) in lines:
        bbox = d.textbbox((x, y), text, font=FONT)
        pad = 6
        d.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=(0, 0, 0))
        d.text((x, y), text, font=FONT, fill=(255, 255, 255))

    out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return out


def centers_from_mask(mask, top_k=3, min_area=MIN_OBJ_AREA):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] <= 1e-6:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        objs.append((cx, cy, area))
    objs.sort(key=lambda t: t[2], reverse=True)
    return [(x, y) for x, y, _ in objs[:top_k]]


def canny_edges(gray, profile):
    g = cv2.GaussianBlur(gray, (3, 3), 0.8)
    return cv2.Canny(g, profile["CANNY_T1"], profile["CANNY_T2"])


def sobel_edges(gray):
    g = cv2.GaussianBlur(gray, (3, 3), 0.8)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(mag, 60, 255, cv2.THRESH_BINARY)
    return th


def laplacian_edges(gray):
    g = cv2.GaussianBlur(gray, (3, 3), 0.8)
    lap = cv2.Laplacian(g, cv2.CV_16S, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    _, th = cv2.threshold(lap, 25, 255, cv2.THRESH_BINARY)
    return th


def fast_points(gray, profile, max_pts=MAX_DRAW_PTS):
    fast = cv2.FastFeatureDetector_create(
        threshold=profile["FAST_THRESHOLD"],
        nonmaxSuppression=profile["FAST_NONMAX"]
    )
    kps = fast.detect(gray, None)
    if not kps:
        return np.empty((0, 2), np.int32)
    kps = sorted(kps, key=lambda k: k.response, reverse=True)[:max_pts]
    pts = np.array([kp.pt for kp in kps], dtype=np.float32).astype(np.int32)
    return pts


def shi_tomasi_points(gray, max_pts=MAX_DRAW_PTS):
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_pts, qualityLevel=0.02, minDistance=8)
    if pts is None:
        return np.empty((0, 2), np.int32)
    return np.int32(pts).reshape(-1, 2)


def harris_points(gray, max_pts=MAX_DRAW_PTS):
    g = np.float32(gray) / 255.0
    dst = cv2.cornerHarris(g, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    th = 0.01 * dst.max() if dst.size else 0.0
    ys, xs = np.where(dst > th)
    if len(xs) == 0:
        return np.empty((0, 2), np.int32)
    resp = dst[ys, xs]
    idx = np.argsort(resp)[::-1][:max_pts]
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.int32)
    return pts


def kmeans_centers_3(pts_xy):
    if pts_xy is None or len(pts_xy) < 3:
        return []
    data = np.float32(pts_xy)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centers = cv2.kmeans(data, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.int32)
    return [(int(c[0]), int(c[1])) for c in centers]


def make_blob_detector(profile):
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = profile["BLOB_MIN_THRESHOLD"]
    p.maxThreshold = 255
    p.filterByArea = True
    p.minArea = profile["BLOB_MIN_AREA"]
    p.maxArea = profile["BLOB_MAX_AREA"]
    p.filterByCircularity = False
    p.filterByConvexity = False
    p.filterByInertia = False
    p.filterByColor = False
    return cv2.SimpleBlobDetector_create(p)


def dog_blobs_centers(gray):
    g1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(gray, (0, 0), 2.0)
    dog = cv2.absdiff(g1, g2)
    _, th = cv2.threshold(dog, 12, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(
        th,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(th)
    comps = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 60:
            continue
        cx, cy = centroids[i]
        comps.append((int(cx), int(cy), area))
    comps.sort(key=lambda t: t[2], reverse=True)
    centers = [(x, y) for x, y, _ in comps[:3]]
    return th, centers


def blackhat_otsu_ridges(gray, profile):
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    kernel_size = profile["BH_KERNEL"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
    _, th = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(
        th,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=profile["BH_ITERATIONS"],
    )
    return th


def hessian_like_ridges(gray):
    g = cv2.GaussianBlur(gray, (0, 0), 1.2)
    g = np.float32(g) / 255.0
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)

    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    sqrt_disc = np.sqrt(disc)

    l1 = 0.5 * (tr + sqrt_disc)
    l2 = 0.5 * (tr - sqrt_disc)

    resp = np.minimum(l1, l2)
    resp = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX)
    resp_u8 = np.uint8(np.clip(resp, 0, 255))
    _, mask = cv2.threshold(resp_u8, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return mask


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def compute_metrics(tp, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    acc = safe_div(tp, tp + fp + fn)
    return precision, recall, f1, acc


def match_points(gt_pts, pred_pts, max_dist):
    used_pred = set()
    tp = 0
    for (gx, gy) in gt_pts:
        best_j = None
        best_d = None
        for j, (px, py) in enumerate(pred_pts):
            if j in used_pred:
                continue
            d = (gx - px) ** 2 + (gy - py) ** 2
            if best_d is None or d < best_d:
                best_d = d
                best_j = j
        if best_d is not None and best_d <= (max_dist * max_dist):
            tp += 1
            used_pred.add(best_j)
    fn = len(gt_pts) - tp
    fp = len(pred_pts) - len(used_pred)
    return tp, fp, fn


def motion_gt(bg, frame_bgr):
    fg = bg.apply(frame_bgr)
    fg = cv2.medianBlur(fg, 5)
    fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    gt_pts = centers_from_mask(fg, top_k=GT_TOPK, min_area=GT_MIN_AREA)
    return fg, gt_pts


def save_plots_and_csv(summary_rows, profile_name):
    names = [r["name"] for r in summary_rows]
    P = [r["P"] for r in summary_rows]
    R = [r["R"] for r in summary_rows]
    F1 = [r["F1"] for r in summary_rows]
    ACC = [r["ACC"] for r in summary_rows]
    MS = [r["mean_ms"] for r in summary_rows]

    x = np.arange(len(names))
    w = 0.22

    plt.figure(figsize=(13, 6))
    plt.bar(x - w, P, width=w, label="Precision")
    plt.bar(x, R, width=w, label="Recall")
    plt.bar(x + w, F1, width=w, label="F1")
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylim(0, 1.0)
    plt.title(f"Precision / Recall / F1 (100 кадрів, GT по руху) | profile={profile_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "metrics_prf.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(13, 5))
    plt.bar(names, MS)
    plt.xticks(rotation=25, ha="right")
    plt.title(f"Середній час методу (мс) | profile={profile_name}")
    plt.ylabel("ms")
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "metrics_time_ms.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(13, 5))
    plt.bar(names, ACC)
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1.0)
    plt.title(f"Object-Accuracy = TP/(TP+FP+FN) | profile={profile_name}")
    plt.ylabel("Acc")
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "metrics_accuracy.png"), dpi=200)
    plt.close()

    csv_path = os.path.join(SCRIPT_DIR, "metrics_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["Profile", "Method", "Precision", "Recall", "F1", "Acc", "Mean_ms", "TP", "FP", "FN"])
        for r in summary_rows:
            wri.writerow([profile_name, r["name"], r["P"], r["R"], r["F1"], r["ACC"], r["mean_ms"], r["tp"], r["fp"], r["fn"]])


class Runner:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.last_panel_path = os.path.join(SCRIPT_DIR, "last_panel.png")

        self.running = False
        self.done = False
        self.error = None

        self.collected = 0
        self.target = N_FRAMES
        self.warmup_done = False
        self.gt_count = 0
        self.fps_s = 0.0

        self.summary_rows = None
        self.profile_name = "balanced"
        self.profile = METHOD_PROFILES[self.profile_name]

        self.stop_event = threading.Event()
        self.thread = None

    def start(self, profile_name="balanced"):
        if self.thread and self.thread.is_alive():
            return
        if profile_name not in METHOD_PROFILES:
            profile_name = "balanced"

        self.profile_name = profile_name
        self.profile = METHOD_PROFILES[profile_name]

        self.stop_event.clear()
        self.running = True
        self.done = False
        self.error = None
        self.collected = 0
        self.gt_count = 0
        self.fps_s = 0.0
        self.summary_rows = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _set_latest(self, panel_bgr):
        ok, buf = cv2.imencode(".jpg", panel_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return
        with self.lock:
            self.latest_jpeg = buf.tobytes()

    def _run(self):
        picam2 = None
        profile = self.profile
        try:
            picam2 = Picamera2()
            cfg = picam2.create_video_configuration(main={"size": (CAP_W, CAP_H), "format": "RGB888"})
            picam2.configure(cfg)
            picam2.start()

            blob = make_blob_detector(profile)
            bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=24, detectShadows=False)
            warmup_start = time.perf_counter()

            methods = {
                "Контури: Canny": {"tp": 0, "fp": 0, "fn": 0, "ms": []},
                "Контури: Sobel": {"tp": 0, "fp": 0, "fn": 0, "ms": []},
                "Контури: Laplacian": {"tp": 0, "fp": 0, "fn": 0, "ms": []},
                "Кути: FAST": {"tp": 0, "fp": 0, "fn": 0, "ms": []},
                "Кути: Shi-Tomasi": {"tp": 0, "fp": 0, "fn": 0, "ms": []},
                "Кути: Harris": {"tp": 0, "fp": 0, "fn": 0, "ms": []},
                "Плями: SimpleBlob": {"tp": 0, "fp": 0, "fn": 0, "ms": []},
                "Плями: DoG": {"tp": 0, "fp": 0, "fn": 0, "ms": []},
                "Хребти: Blackhat+Otsu": {"tp": 0, "fp": 0, "fn": 0, "ms": []},
                "Хребти: Hessian-like": {"tp": 0, "fp": 0, "fn": 0, "ms": []},
            }

            ridge1_cache = None
            ridge2_cache = None

            prev = time.perf_counter()
            frame_id = 0

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

                gt_mask, gt_pts = motion_gt(bg, frame)
                self.gt_count = len(gt_pts)

                t0 = time.perf_counter()
                e1 = canny_edges(gray, profile)
                t1 = time.perf_counter()
                methods["Контури: Canny"]["ms"].append((t1 - t0) * 1000.0)
                pred_canny = centers_from_mask(e1, top_k=3, min_area=MIN_OBJ_AREA)

                t0 = time.perf_counter()
                p_fast = fast_points(gray, profile)
                t1 = time.perf_counter()
                methods["Кути: FAST"]["ms"].append((t1 - t0) * 1000.0)
                pred_fast = kmeans_centers_3(p_fast)

                t0 = time.perf_counter()
                kps = blob.detect(gray)
                t1 = time.perf_counter()
                methods["Плями: SimpleBlob"]["ms"].append((t1 - t0) * 1000.0)
                if kps:
                    kps_sorted = sorted(kps, key=lambda k: k.size, reverse=True)[:3]
                    pred_blob = [(int(k.pt[0]), int(k.pt[1])) for k in kps_sorted]
                else:
                    pred_blob = []

                if frame_id % HEAVY_PERIOD == 0 or ridge1_cache is None:
                    t0 = time.perf_counter()
                    ridge1_cache = blackhat_otsu_ridges(gray, profile)
                    t1 = time.perf_counter()
                    methods["Хребти: Blackhat+Otsu"]["ms"].append((t1 - t0) * 1000.0)

                pred_r1 = centers_from_mask(ridge1_cache, top_k=3, min_area=MIN_OBJ_AREA) if ridge1_cache is not None else []

                tl = ensure_bgr(e1)

                corners_vis = frame.copy()
                for x, y in p_fast[:MAX_DRAW_PTS]:
                    cv2.circle(corners_vis, (int(x), int(y)), 1, (0, 0, 255), -1)
                tr = corners_vis

                blobs_vis = frame.copy()
                if kps:
                    cv2.drawKeypoints(
                        blobs_vis,
                        kps,
                        blobs_vis,
                        (255, 0, 0),
                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                    )
                bl = blobs_vis

                br = ensure_bgr(ridge1_cache if ridge1_cache is not None else np.zeros_like(gray))

                panel = stack_2x2(tl, tr, bl, br)
                panel = resize_w(panel, PANEL_W)

                gt_small = cv2.cvtColor(
                    cv2.resize(gt_mask, (160, 120), interpolation=cv2.INTER_NEAREST),
                    cv2.COLOR_GRAY2BGR,
                )
                panel[50:170, 8:168] = gt_small

                lines = [
                    (15, 12, f"profile={self.profile_name} | GT по руху | warmup={'OK' if self.warmup_done else '...'} | GT={len(gt_pts)}/3 | FPS~{self.fps_s:.1f}"),
                    (15, 205, f"Авто-збір: кадр {self.collected}/{self.target} | STOP=кнопка"),
                    (15, 180, "GT mask (рух)"),
                    (panel.shape[1]//2 + 15, 12, f"Кути (FAST) | точки={len(p_fast)} | центри={len(pred_fast)}"),
                    (15, panel.shape[0]//2 + 12, f"Плями (SimpleBlob) | blobs={len(kps) if kps else 0} | центри={len(pred_blob)}"),
                    (panel.shape[1]//2 + 15, panel.shape[0]//2 + 12, f"Хребти (Blackhat+Otsu) | центри={len(pred_r1)} | HEAVY={HEAVY_PERIOD}"),
                ]
                panel = draw_ua_text(panel, lines)

                self._set_latest(panel)

                if frame_id % PREVIEW_EVERY_N == 0:
                    try:
                        cv2.imwrite(self.last_panel_path, panel)
                    except Exception:
                        pass

                if self.warmup_done and len(gt_pts) == 3:
                    pred_map = {
                        "Контури: Canny": pred_canny,
                        "Кути: FAST": pred_fast,
                        "Плями: SimpleBlob": pred_blob,
                        "Хребти: Blackhat+Otsu": pred_r1,
                    }
                    for name, pred in pred_map.items():
                        tp, fp, fn = match_points(gt_pts, pred, MATCH_DIST_PX)
                        methods[name]["tp"] += tp
                        methods[name]["fp"] += fp
                        methods[name]["fn"] += fn
                    self.collected += 1

                if self.collected >= self.target:
                    break

                frame_id += 1

                if STREAM_FPS_LIMIT > 0:
                    time.sleep(max(0.0, (1.0 / STREAM_FPS_LIMIT) - 0.001))

            summary_rows = []
            for name in ["Контури: Canny", "Кути: FAST", "Плями: SimpleBlob", "Хребти: Blackhat+Otsu"]:
                d = methods[name]
                tp, fp, fn = d["tp"], d["fp"], d["fn"]
                P, R, F1, ACC = compute_metrics(tp, fp, fn)
                mean_ms = float(np.mean(d["ms"])) if len(d["ms"]) else 0.0
                summary_rows.append({
                    "name": name,
                    "P": P, "R": R, "F1": F1, "ACC": ACC,
                    "mean_ms": mean_ms,
                    "tp": tp, "fp": fp, "fn": fn
                })

            self.summary_rows = summary_rows
            save_plots_and_csv(summary_rows, self.profile_name)

            self.done = (self.collected >= self.target) and not self.stop_event.is_set()
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
  <title>Камера: 3 рухомі об’єкти + методи (2x2)</title>
  <style>
    body{margin:0;background:#0b0d12;color:#fff;font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial}
    .wrap{max-width:1100px;margin:0 auto;padding:32px 18px}
    h1{font-size:42px;margin:0 0 8px;text-align:center}
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
    select{background:#111827;color:#fff;border:1px solid #243047;border-radius:12px;padding:12px 14px;font-size:16px}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Камера: 3 рухомі об’єкти + методи (2x2)</h1>
  <div class="sub">Обери профіль параметрів методів, потім натисни Старт.</div>

  <div class="btns">
    <select id="profile">
      <option value="balanced">balanced</option>
      <option value="strict_clean">strict_clean</option>
      <option value="sensitive_recall">sensitive_recall</option>
      <option value="blob_focus">blob_focus</option>
      <option value="fast_dense">fast_dense</option>
      <option value="blackhat_strong">blackhat_strong</option>
    </select>
    <button onclick="startRun()">Старт</button>
    <button onclick="stopRun()">Зупинити</button>
    <button class="secondary" id="resBtn" onclick="openResults()" style="display:none">Відкрити результати</button>
  </div>

  <div class="status">
    <div class="pill">
      <div id="dot" class="dot wait"></div>
      <div id="st">Очікування запуску…</div>
    </div>
  </div>

  <div class="panelWrap">
    <div class="panel">
      <img id="panel" src="/panel_feed" />
    </div>
  </div>

  <div class="sub" style="margin-top:14px">
    last_panel.png зберігається в папку коду. Результати: <a href="/results">/results</a>
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
      st.textContent = '✅ Завершено — profile=' + j.profile + ' — зібрано ' + j.collected + '/' + j.target + ' кадрів';
      resBtn.style.display='inline-block';
      return;
    }

    dot.className = 'dot wait';
    st.textContent = 'profile=' + j.profile + ' | Збір: ' + j.collected + '/' + j.target + ' | warmup=' + (j.warmup_done?'OK':'...') + ' | GT=' + j.gt_count + '/3 | FPS~' + j.fps.toFixed(1);
    resBtn.style.display='none';
  }catch(e){}
  setTimeout(poll, 600);
}
poll();

async function startRun(){
  const profile = document.getElementById('profile').value;
  await fetch('/api/start', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({profile: profile})
  });
  setTimeout(()=>location.reload(), 200);
}

async function stopRun(){
  await fetch('/api/stop', {method:'POST'});
  setTimeout(()=>location.reload(), 400);
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
    data = request.get_json(silent=True) or {}
    profile = data.get("profile", "balanced")
    runner.start(profile)
    return jsonify({"ok": True, "profile": profile})


@app.route("/api/status")
def api_status():
    return jsonify({
        "running": runner.running,
        "done": runner.done,
        "error": runner.error,
        "collected": runner.collected,
        "target": runner.target,
        "warmup_done": runner.warmup_done,
        "gt_count": runner.gt_count,
        "fps": runner.fps_s,
        "profile": runner.profile_name,
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
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        if runner.done and not runner.running:
            time.sleep(0.3)
        else:
            time.sleep(0.03)


@app.route("/panel_feed")
def panel_feed():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


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
    def fmt(x): return f"{x:.3f}"
    tr = ""
    for r in rows:
        tr += f"<tr><td>{r['name']}</td><td>{fmt(r['P'])}</td><td>{fmt(r['R'])}</td><td>{fmt(r['F1'])}</td><td>{fmt(r['ACC'])}</td><td>{r['mean_ms']:.2f}</td><td>{r['tp']}</td><td>{r['fp']}</td><td>{r['fn']}</td></tr>"
    return f"""
    <html>
    <head><meta charset="utf-8"/>
    <title>Results</title>
    <style>
      body{{font-family:system-ui;background:#0b0d12;color:#fff;padding:24px}}
      table{{border-collapse:collapse;width:100%;max-width:1100px}}
      th,td{{border:1px solid #23314a;padding:10px}}
      th{{background:#111827}}
      a{{color:#9ca3af}}
      .imgs img{{max-width:520px;border-radius:14px;border:1px solid #23314a;margin:10px}}
    </style>
    </head>
    <body>
      <h2>✅ Метрики | profile={runner.profile_name}</h2>
      <p><a href="/">← Назад</a></p>
      <table>
        <tr><th>Method</th><th>P</th><th>R</th><th>F1</th><th>Acc</th><th>Mean ms</th><th>TP</th><th>FP</th><th>FN</th></tr>
        {tr}
      </table>
      <h3 style="margin-top:20px">Графіки</h3>
      <div class="imgs">
        <img src="/file/metrics_prf.png" />
        <img src="/file/metrics_time_ms.png" />
        <img src="/file/metrics_accuracy.png" />
      </div>
      <p>CSV: <a href="/file/metrics_summary.csv">metrics_summary.csv</a></p>
    </body>
    </html>
    """


@app.route("/file/<path:fname>")
def file_get(fname):
    return send_from_directory(SCRIPT_DIR, fname, as_attachment=False)


def main():
    print("Запуск Flask...")
    print("Відкрий у браузері:")
    print("  http://<IP_RPI>:5000")
    print(f"last_panel.png: {os.path.join(SCRIPT_DIR, 'last_panel.png')}")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()