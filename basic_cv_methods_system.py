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

HEAVY_PERIOD = 8
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
# Найкращі параметри з попередніх прогонів
# -------------------------------
CANNY_T1 = 50
CANNY_T2 = 140

FAST_THRESHOLD = 22
FAST_NONMAX = True

BLOB_MIN_AREA = 160
BLOB_MAX_AREA = 30000
BLOB_MIN_THRESHOLD = 50

BH_KERNEL = 15
BH_ITERATIONS = 1


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


def keypoints_to_mask(shape, kps):
    mask = np.zeros(shape, dtype=np.uint8)
    if not kps:
        return mask
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = max(2, int(kp.size / 2))
        cv2.circle(mask, (x, y), r, 255, -1)
    return mask


def canny_method(gray):
    g = cv2.GaussianBlur(gray, (3, 3), 0.8)
    edges = cv2.Canny(g, CANNY_T1, CANNY_T2)
    # трохи потовщуємо, щоб порівняння по масці було адекватніше
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    vis = ensure_bgr(edges)
    return edges, vis, f"Canny | t1={CANNY_T1} t2={CANNY_T2}"


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

    vis = frame_bgr.copy()
    for x, y in pts:
        cv2.circle(vis, (x, y), 1, (0, 0, 255), -1)

    return mask, vis, f"FAST | thr={FAST_THRESHOLD} nonmax={FAST_NONMAX} pts={len(pts)}"


def blob_method(gray, frame_bgr):
    p = cv2.SimpleBlobDetector_Params()
    p.minThreshold = float(BLOB_MIN_THRESHOLD)
    p.maxThreshold = 255
    p.filterByArea = True
    p.minArea = float(BLOB_MIN_AREA)
    p.maxArea = float(BLOB_MAX_AREA)
    p.filterByCircularity = False
    p.filterByConvexity = False
    p.filterByInertia = False
    p.filterByColor = False

    detector = cv2.SimpleBlobDetector_create(p)
    kps = detector.detect(gray)

    mask = keypoints_to_mask(gray.shape, kps)

    vis = frame_bgr.copy()
    if kps:
        cv2.drawKeypoints(
            vis,
            kps,
            vis,
            (255, 0, 0),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

    return mask, vis, f"SimpleBlob | minA={BLOB_MIN_AREA} maxA={BLOB_MAX_AREA} minT={BLOB_MIN_THRESHOLD} blobs={len(kps) if kps else 0}"


def blackhat_method(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    k = BH_KERNEL if BH_KERNEL % 2 == 1 else BH_KERNEL + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
    _, th = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(
        th,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=BH_ITERATIONS,
    )
    vis = ensure_bgr(th)
    return th, vis, f"Blackhat+Otsu | kernel={BH_KERNEL} iter={BH_ITERATIONS}"


METHODS = [
    ("Контури: Canny", canny_method),
    ("Кути: FAST", fast_method),
    ("Плями: SimpleBlob", blob_method),
    ("Хребти: Blackhat+Otsu", blackhat_method),
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

    # 1
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

    # 2
    plt.figure(figsize=(16, 7))
    plt.bar(names, MS)
    plt.xticks(rotation=15, ha="right", fontsize=11)
    plt.yticks(fontsize=11)
    plt.title("Середній час методу (мс)", fontsize=16)
    plt.ylabel("ms", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "metrics_time_ms.png"), dpi=220)
    plt.close()

    # 3
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
        wri.writerow(["Method", "Precision", "Recall", "F1", "Acc", "Mean_ms", "TP", "FP", "FN"])
        for r in summary_rows:
            wri.writerow([
                r["name"], r["P"], r["R"], r["F1"], r["ACC"],
                r["mean_ms"], r["tp"], r["fp"], r["fn"]
            ])


class Runner:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.last_panel_path = os.path.join(SCRIPT_DIR, "last_panel.png")

        self.running = False
        self.done = False
        self.error = None

        self.current_method = "-"
        self.method_index = 0
        self.method_total = len(METHODS)

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
        self.current_method = "-"
        self.method_index = 0
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

            for idx, (method_name, method_func) in enumerate(METHODS, start=1):
                if self.stop_event.is_set():
                    break

                self.current_method = method_name
                self.method_index = idx
                self.collected = 0

                method_stats = {"tp": 0, "fp": 0, "fn": 0, "ms": []}

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

                    # метод
                    t0 = time.perf_counter()
                    if method_name == "Контури: Canny":
                        pred_mask, vis, method_info = method_func(gray)
                    elif method_name == "Кути: FAST":
                        pred_mask, vis, method_info = method_func(gray, frame)
                    elif method_name == "Плями: SimpleBlob":
                        pred_mask, vis, method_info = method_func(gray, frame)
                    else:
                        pred_mask, vis, method_info = method_func(gray)
                    t1 = time.perf_counter()

                    # панель
                    tl = ensure_bgr(gt_mask)
                    tr = ensure_bgr(pred_mask)
                    bl = frame.copy()
                    br = vis.copy()

                    panel = stack_2x2(tl, tr, bl, br)
                    panel = resize_w(panel, PANEL_W)

                    lines = [
                        (15, 12, f"Метод {self.method_index}/{self.method_total}: {method_name}"),
                        (15, 42, f"Кадр {self.collected}/{self.target} | warmup={'OK' if self.warmup_done else '...'} | GT pixels={self.gt_pixels} | FPS~{self.fps_s:.1f}"),
                        (15, 72, method_info),
                        (15, 102, "TL: GT mask | TR: Pred mask | BL: Original | BR: Method view"),
                    ]
                    panel = draw_ua_text(panel, lines)
                    self._set_latest(panel)

                    if frame_id % PREVIEW_EVERY_N == 0:
                        try:
                            cv2.imwrite(self.last_panel_path, panel)
                            cv2.imwrite(os.path.join(self.run_dir, f"preview_{method_name.replace(':', '').replace(' ', '_')}.png"), panel)
                        except Exception:
                            pass

                    # збір тільки якщо warmup вже пройшов і є якийсь рух
                    if self.warmup_done and self.gt_pixels >= GT_MIN_PIXELS:
                        tp, fp, fn = pixel_match(gt_mask, pred_mask)
                        method_stats["tp"] += tp
                        method_stats["fp"] += fp
                        method_stats["fn"] += fn
                        method_stats["ms"].append((t1 - t0) * 1000.0)
                        self.collected += 1

                    if self.collected >= self.target:
                        break

                    frame_id += 1

                    if STREAM_FPS_LIMIT > 0:
                        time.sleep(max(0.0, (1.0 / STREAM_FPS_LIMIT) - 0.001))

                tp, fp, fn = method_stats["tp"], method_stats["fp"], method_stats["fn"]
                P, R, F1, ACC = compute_metrics(tp, fp, fn)
                mean_ms = float(np.mean(method_stats["ms"])) if method_stats["ms"] else 0.0

                summary_rows.append({
                    "name": method_name,
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
  <div class="sub">Один запуск = усі методи по черзі. Для кожного методу збирається 100 валідних кадрів.</div>

  <div class="hint">
    GT тепер рахується по всій картині через маску руху. Логіка "3 об'єкти в кадрі" прибрана.
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
      'Метод ' + j.method_index + '/' + j.method_total +
      ' | ' + j.current_method +
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

    run_dir = runner.run_dir
    run_name = runner.run_name

    def fmt(x):
        return f"{x:.3f}"

    tr = ""
    for r in rows:
        tr += (
            f"<tr>"
            f"<td>{r['name']}</td>"
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
        table{{border-collapse:collapse;width:100%;max-width:980px;font-size:15px}}
        th,td{{border:1px solid #23314a;padding:8px 10px;text-align:center}}
        th{{background:#111827}}
        a{{color:#9ca3af}}
        .imgs{{display:flex;flex-direction:column;gap:18px;max-width:980px}}
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