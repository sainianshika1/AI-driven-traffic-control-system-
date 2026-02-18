import cv2
import time
import numpy as np
import requests
from ultralytics import YOLO
from collections import deque

# =========================
# CONFIGURATION
# =========================
VIDEO_PATH = "heavy_traffic.mp4"

TRAFFIC_MODEL_PATH   = "yolo26n.pt"
AMBULANCE_MODEL_PATH = "best_2.pt"

# Confidence thresholds (raised for optimised performance)
TRAFFIC_CONF   = 0.6   # was 0.5
AMBULANCE_CONF = 0.6   # was 0.4

AZURE_KEY = "7MxOjQLKTJrYIraHwDJzNVkiczHvbuQcJFU5M9woQjF1R4s8eT9rJQQJ99CBACYeBjFowopiAAAgAZMP14uL"
# Traffic light fixed location (your intersection in Delhi — change as needed)
TRAFFIC_LIGHT_LAT = 28.6129   # India Gate area
TRAFFIC_LIGHT_LON = 77.2295

GPS_FILE = "ambulance_location.txt"   # written by gps_server.py

# Fallback location used when GPS file is not available (Connaught Place, Delhi)
FALLBACK_LAT = 28.6315
FALLBACK_LON = 77.2167

def get_ambulance_location():
    """Read latest ambulance GPS coords from file sent by phone.
    Falls back to default location if file not found."""
    try:
        with open(GPS_FILE, "r") as f:
            lat, lon = map(float, f.read().strip().split(","))
            return lat, lon
    except:
        return FALLBACK_LAT, FALLBACK_LON  # fallback so ETA always works
ETA_INTERVAL = 5

NUM_LANES = 2
GREEN_TIME = 40
YELLOW_TIME = 5

EM_CONFIRM_SEC = 3
EM_CLEAR_SEC = 2

FRAME_W, FRAME_H = 1100, 600   # wider canvas for dashboard panels
VIDEO_W, VIDEO_H = 660, 480    # video region
PANEL_W = FRAME_W - VIDEO_W    # right panel width = 440

MAX_VEHICLES_PER_LANE = 15
EXTEND_STEP = 10
MAX_EXTENSION_TOTAL = 120

# =========================
# COLORS  (BGR)
# =========================
BG_DARK      = (15, 15, 25)
PANEL_BG     = (22, 28, 38)
ACCENT_BLUE  = (255, 160, 50)
ACCENT_CYAN  = (220, 210, 60)
GREEN_COL    = (60, 220, 100)
YELLOW_COL   = (30, 210, 240)
RED_COL      = (60, 60, 220)
WHITE        = (240, 240, 245)
GRAY         = (100, 110, 125)
ALERT_RED    = (40, 40, 230)
NEON_GREEN   = (80, 255, 120)
NEON_AMBER   = (0, 180, 255)

# =========================
# LOAD MODELS
# =========================
traffic_model   = YOLO(TRAFFIC_MODEL_PATH,   task="detect")
ambulance_model = YOLO(AMBULANCE_MODEL_PATH, task="detect")
for m in (traffic_model, ambulance_model):
    m.overrides["verbose"] = False

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
EM_CONFIRM_FRAMES = int(EM_CONFIRM_SEC * fps)
EM_CLEAR_FRAMES   = int(EM_CLEAR_SEC  * fps)

# =========================
# LANES  (in video region)
# =========================
lane_width = VIDEO_W // NUM_LANES
lanes = {
    i + 1: (i * lane_width,
            VIDEO_W if i == NUM_LANES - 1 else (i + 1) * lane_width)
    for i in range(NUM_LANES)
}

def get_lane_from_x(cx):
    for lid, (x1, x2) in lanes.items():
        if x1 <= cx < x2:
            return lid
    return None

# =========================
# ETA
# =========================
def get_eta():
    """Calculate ETA from ambulance's live GPS location to the traffic light."""
    amb_lat, amb_lon = get_ambulance_location()
    start = f"{amb_lat},{amb_lon}"
    end   = f"{TRAFFIC_LIGHT_LAT},{TRAFFIC_LIGHT_LON}"

    url = (
        "https://atlas.microsoft.com/route/directions/json"
        f"?api-version=1.0&query={start}:{end}"
        f"&travelMode=car&subscription-key={AZURE_KEY}"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if "routes" not in data:
            print(f"[ETA ERROR] {data}")
            return None

        return data["routes"][0]["summary"]["travelTimeInSeconds"]

    except requests.exceptions.ConnectionError:
        print("[ETA ERROR] No internet connection")
        return None
    except requests.exceptions.Timeout:
        print("[ETA ERROR] Request timed out")
        return None
    except Exception as e:
        print(f"[ETA ERROR] {type(e).__name__}: {e}")
        return None

eta = None
last_eta_time = 0

# =========================
# UI HELPER FUNCTIONS
# =========================

def draw_rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1, alpha=1.0):
    """Draw a filled or outlined rounded rectangle."""
    if thickness == -1:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(overlay, (cx, cy), r, color, -1)
        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        else:
            img[:] = overlay
    else:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(img, (cx, cy), r, color, thickness)

def blend_rect(img, x1, y1, x2, y2, color, alpha=0.55):
    roi = img[y1:y2, x1:x2]
    solid = np.full_like(roi, color)
    cv2.addWeighted(solid, alpha, roi, 1 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi

def draw_text_centered(img, text, cx, cy, font, scale, color, thick=1):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(img, text, (cx - tw//2, cy + th//2), font, scale, color, thick, cv2.LINE_AA)

def draw_glow_circle(img, cx, cy, r, color, alpha=0.4):
    for dr in range(r + 12, r - 1, -2):
        a = alpha * (1 - (dr - r) / 14)
        blend_rect(img, max(0,cx-dr), max(0,cy-dr), min(img.shape[1],cx+dr), min(img.shape[0],cy+dr),
                   color, alpha=max(0, a * 0.15))
    cv2.circle(img, (cx, cy), r, color, -1)

def draw_progress_bar(img, x, y, w, h, value, max_val, bar_color, bg_color=(40,45,55)):
    blend_rect(img, x, y, x+w, y+h, bg_color, alpha=0.85)
    cv2.rectangle(img, (x, y), (x+w, y+h), GRAY, 1)
    fill = int(w * min(value, max_val) / max_val)
    if fill > 0:
        blend_rect(img, x, y, x+fill, y+h, bar_color, alpha=0.95)
        cv2.rectangle(img, (x, y), (x+fill, y+h), bar_color, 1)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN

def build_dashboard(video_frame, state, current_lane, green_remaining, yellow_remaining,
                    emergency_active, emergency_lane, lane_counts, eta, sim_sec, pulse):
    """Compose the full 1100×600 dashboard canvas."""

    # --- Base canvas ---
    canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    canvas[:] = BG_DARK

    # =====================
    # TOP HEADER BAR
    # =====================
    blend_rect(canvas, 0, 0, FRAME_W, 48, PANEL_BG, alpha=0.98)
    cv2.line(canvas, (0, 48), (FRAME_W, 48), ACCENT_BLUE, 1)

    # Logo / Title
    # Header title color follows signal state
    header_col = GREEN_COL if state == "GREEN" else YELLOW_COL if state == "YELLOW" else RED_COL
    cv2.putText(canvas, "SMART TRAFFIC CONTROL SYSTEM", (20, 32),
                FONT, 0.65, header_col, 1, cv2.LINE_AA)

    # Live badge
    lx = FRAME_W - 120
    blend_rect(canvas, lx, 10, lx+95, 38, (30,10,10), alpha=0.9)
    cv2.rectangle(canvas, (lx, 10), (lx+95, 38), ALERT_RED, 1)
    dot_col = ALERT_RED if pulse else (80,30,30)
    cv2.circle(canvas, (lx+16, 24), 5, dot_col, -1)
    cv2.putText(canvas, "LIVE", (lx+28, 30), FONT, 0.55, WHITE, 1, cv2.LINE_AA)

    # Clock — color follows signal state
    t_str = time.strftime("%H:%M:%S")
    cv2.putText(canvas, t_str, (FRAME_W//2 - 40, 32), FONT, 0.6, header_col, 1, cv2.LINE_AA)

    # =====================
    # VIDEO FEED (left)
    # =====================
    vid_x, vid_y = 10, 58
    if video_frame is not None:
        vf = cv2.resize(video_frame, (VIDEO_W, VIDEO_H))
        canvas[vid_y:vid_y+VIDEO_H, vid_x:vid_x+VIDEO_W] = vf
    cv2.rectangle(canvas, (vid_x, vid_y), (vid_x+VIDEO_W, vid_y+VIDEO_H), ACCENT_BLUE, 2)

    # Lane divider overlay
    mid_x = vid_x + VIDEO_W // 2
    cv2.line(canvas, (mid_x, vid_y), (mid_x, vid_y+VIDEO_H), (60,80,100), 1)

    # Lane labels
    for i, (lid, (lx1, lx2)) in enumerate(lanes.items()):
        label_x = vid_x + (lx1 + lx2)//2
        col = NEON_GREEN if (lid == current_lane and state == "GREEN") else \
              NEON_AMBER if (lid == current_lane and state == "YELLOW") else GRAY
        blend_rect(canvas, vid_x+lx1+5, vid_y+VIDEO_H-28, vid_x+lx2-5, vid_y+VIDEO_H-4,
                   (20,25,35), alpha=0.8)
        draw_text_centered(canvas, f"LANE {lid}", label_x, vid_y+VIDEO_H-14,
                           FONT, 0.45, col, 1)

    # Emergency bounding-box flash on video
    if emergency_active and pulse:
        cv2.rectangle(canvas, (vid_x+2, vid_y+2), (vid_x+VIDEO_W-2, vid_y+VIDEO_H-2), ALERT_RED, 3)

    # Sim time label — color follows signal state
    _sc = GREEN_COL if state=="GREEN" else YELLOW_COL if state=="YELLOW" else RED_COL
    cv2.putText(canvas, f"T+{sim_sec:04d}s", (vid_x+5, vid_y+VIDEO_H+18),
                FONT_MONO, 1.0, _sc, 1, cv2.LINE_AA)

    # =====================
    # RIGHT PANEL
    # =====================
    px = VIDEO_W + 20   # panel start x
    py = 58             # panel start y

    # Signal state color (used throughout the panel)
    state_col = GREEN_COL if state=="GREEN" else YELLOW_COL if state=="YELLOW" else RED_COL

    # --- Signal State Card ---
    card_y = py + 5
    blend_rect(canvas, px, card_y, FRAME_W-10, card_y+130, PANEL_BG, alpha=0.95)
    cv2.rectangle(canvas, (px, card_y), (FRAME_W-10, card_y+130), state_col, 2)

    # Traffic light circles
    light_x = px + 35
    light_y_base = card_y + 22
    for i, (lname, lcolor_off, lcolor_on) in enumerate([
        ("RED",    (30,15,15), RED_COL),
        ("YELLOW", (25,25,15), YELLOW_COL),
        ("GREEN",  (15,30,15), GREEN_COL),
    ]):
        active = (lname == state) or (lname == "RED" and state not in ("GREEN","YELLOW"))
        col = lcolor_on if active else lcolor_off
        cy_ = light_y_base + i * 28
        if active and pulse:
            draw_glow_circle(canvas, light_x, cy_, 9, col, alpha=0.5)
        else:
            cv2.circle(canvas, (light_x, cy_), 9, col, -1)
        cv2.circle(canvas, (light_x, cy_), 9, (60,70,80), 1)
    cv2.putText(canvas, state, (light_x+22, card_y+30), FONT, 0.9, state_col, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Active Lane: {current_lane}", (light_x+22, card_y+56),
                FONT, 0.5, state_col, 1, cv2.LINE_AA)

    # Timer bar
    if state == "GREEN":
        rem, mx, bar_c = green_remaining, GREEN_TIME, GREEN_COL
        rem_label = f"{green_remaining}s remaining"
    elif state == "YELLOW":
        rem, mx, bar_c = yellow_remaining, YELLOW_TIME, YELLOW_COL
        rem_label = f"{yellow_remaining}s remaining"
    else:
        rem, mx, bar_c = 0, GREEN_TIME, RED_COL
        rem_label = "Switching..."
    draw_progress_bar(canvas, light_x+22, card_y+68, PANEL_W-60, 14, rem, mx, bar_c)
    cv2.putText(canvas, rem_label, (light_x+22, card_y+102),
                FONT, 0.42, state_col, 1, cv2.LINE_AA)

    # --- Lane Density Cards ---
    card_y2 = card_y + 138
    blend_rect(canvas, px, card_y2, FRAME_W-10, card_y2+110, PANEL_BG, alpha=0.95)
    cv2.rectangle(canvas, (px, card_y2), (FRAME_W-10, card_y2+110), ACCENT_BLUE, 1)
    cv2.putText(canvas, "LANE DENSITY", (px+10, card_y2+18),
                FONT, 0.5, ACCENT_CYAN, 1, cv2.LINE_AA)
    cv2.line(canvas, (px+10, card_y2+24), (FRAME_W-20, card_y2+24), (40,50,65), 1)

    for i, lid in enumerate(lanes):
        cnt = lane_counts.get(lid, 0)
        ly = card_y2 + 38 + i * 34
        is_active = (lid == current_lane)
        label_col = NEON_GREEN if is_active else WHITE
        cv2.putText(canvas, f"Lane {lid}", (px+12, ly+12), FONT, 0.48, label_col, 1, cv2.LINE_AA)
        density_pct = min(cnt / MAX_VEHICLES_PER_LANE, 1.0)
        bar_col = RED_COL if density_pct > 0.7 else YELLOW_COL if density_pct > 0.4 else GREEN_COL
        draw_progress_bar(canvas, px+75, ly, PANEL_W-110, 18, cnt, MAX_VEHICLES_PER_LANE, bar_col)
        cv2.putText(canvas, f"{density_pct:.2f}", (FRAME_W-48, ly+14), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        if is_active:
            cv2.rectangle(canvas, (px+3, ly-4), (FRAME_W-14, ly+22), GREEN_COL, 1)

    # --- Emergency Card ---
    card_y3 = card_y2 + 128
    em_bg = (35,10,10) if emergency_active else PANEL_BG
    blend_rect(canvas, px, card_y3, FRAME_W-10, card_y3+80, em_bg, alpha=0.95)
    border_col = ALERT_RED if emergency_active else ACCENT_BLUE
    cv2.rectangle(canvas, (px, card_y3), (FRAME_W-10, card_y3+80), border_col, 1 if not emergency_active else 2)

    # Siren icon (simple triangle)
    icon_col = ALERT_RED if (emergency_active and pulse) else (70,40,40) if emergency_active else GRAY
    siren_pts = np.array([[px+22, card_y3+14], [px+10, card_y3+38], [px+34, card_y3+38]], np.int32)
    cv2.fillPoly(canvas, [siren_pts], icon_col)
    cv2.putText(canvas, "!", (px+17, card_y3+35), FONT, 0.5, WHITE, 1, cv2.LINE_AA)

    em_title_col = ALERT_RED if emergency_active else GRAY
    em_status = "EMERGENCY DETECTED" if emergency_active else "NO EMERGENCY"
    cv2.putText(canvas, em_status, (px+44, card_y3+20), FONT, 0.52, em_title_col, 1, cv2.LINE_AA)
    if emergency_active and emergency_lane:
        cv2.putText(canvas, f"Vehicle in Lane {emergency_lane} — Preempting Signal",
                    (px+44, card_y3+42), FONT, 0.42, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas,
                "Priority routing active" if emergency_active else "All lanes normal",
                (px+44, card_y3+62), FONT, 0.42, GRAY, 1, cv2.LINE_AA)

    # --- ETA Card ---
    card_y4 = card_y3 + 98
    blend_rect(canvas, px, card_y4, FRAME_W-10, card_y4+70, PANEL_BG, alpha=0.95)
    cv2.rectangle(canvas, (px, card_y4), (FRAME_W-10, card_y4+70), ACCENT_BLUE, 1)
    import os
    eta_source = "LIVE GPS" if os.path.exists(GPS_FILE) else "DEFAULT LOCATION"
    cv2.putText(canvas, f"AMBULANCE ETA  [{eta_source}]", (px+10, card_y4+18), FONT, 0.42, ACCENT_CYAN, 1, cv2.LINE_AA)
    cv2.line(canvas, (px+10, card_y4+24), (FRAME_W-20, card_y4+24), (40,50,65), 1)

    if eta:
        eta_min = eta // 60
        eta_sec = eta % 60
        eta_str  = f"{eta_min}m {eta_sec:02d}s"
        eta_col  = ALERT_RED if eta < 120 else NEON_AMBER if eta < 300 else NEON_GREEN
        cv2.putText(canvas, eta_str, (px+12, card_y4+56),
                    FONT, 1.0, eta_col, 2, cv2.LINE_AA)
        status_txt = "⚠ FAST ARRIVAL" if eta < 120 else "ON ROUTE"
        cv2.putText(canvas, status_txt, (px+130, card_y4+56), FONT, 0.42, eta_col, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "Fetching...", (px+12, card_y4+50), FONT, 0.6, GRAY, 1, cv2.LINE_AA)

    # --- Bottom status bar ---
    cv2.line(canvas, (0, FRAME_H-26), (FRAME_W, FRAME_H-26), (40,50,65), 1)
    blend_rect(canvas, 0, FRAME_H-26, FRAME_W, FRAME_H, PANEL_BG, alpha=0.9)
    cv2.putText(canvas, f"System Runtime: {sim_sec}s   |   Press Q to quit   |   Lanes: {NUM_LANES}   |   FPS: {fps:.0f}",
                (15, FRAME_H-8), FONT, 0.38, _sc, 1, cv2.LINE_AA)

    return canvas


# =========================
# STORAGE
# =========================
density_history = {lid: deque(maxlen=5) for lid in lanes}

emergency_active = False
emergency_lane   = None
em_detect_count  = 0
em_clear_count   = 0
pending_emergency_lane = None
preempting = False
em_extension_start_time = None
em_extension_used = 0

state = "GREEN"
green_remaining  = GREEN_TIME
yellow_remaining = 0

real_start    = time.monotonic()
last_print_sec = -1
lane_counts   = {lid: 0 for lid in lanes}

# Warmup
INITIAL_WARMUP_FRAMES = int(2 * fps)
warmup_counts = {lid: 0 for lid in lanes}
for _ in range(INITIAL_WARMUP_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
    results = traffic_model(frame, conf=TRAFFIC_CONF)[0]
    if results.boxes is not None:
        for box in results.boxes:
            x1, _, x2, _ = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            lane = get_lane_from_x(cx)
            if lane:
                warmup_counts[lane] += 1

current_lane = max(warmup_counts, key=warmup_counts.get)
print(f"🚦 INITIAL GREEN SELECTED → Lane {current_lane}")

# =========================
# MAIN LOOP
# =========================
pulse = True
pulse_counter = 0

while True:
    now = time.monotonic()
    sim_sec = int(now - real_start)

    # Pulse every ~15 frames for blinking effects
    pulse_counter += 1
    if pulse_counter % 15 == 0:
        pulse = not pulse

    ret, frame = cap.read()
    if not ret:
        frame = None

    # ETA update
    if time.time() - last_eta_time > ETA_INTERVAL:
        eta = get_eta()
        last_eta_time = time.time()

    # ================= VIDEO PROCESSING =================
    if frame is not None:
        frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
        results = traffic_model(frame, conf=TRAFFIC_CONF)[0]
        lane_counts = {lid: 0 for lid in lanes}

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                lane = get_lane_from_x(cx)
                if lane:
                    lane_counts[lane] += 1
                # Draw bounding boxes on video
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 180, 60), 1)

        # Emergency detection — ambulance only
        detected = False
        detected_lane = None

        res = ambulance_model(frame, conf=AMBULANCE_CONF)
        if res and res[0].boxes:
            for box in res[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                lane = get_lane_from_x(cx)
                if lane:
                    detected = True
                    detected_lane = lane
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 230), 3)
                    cv2.putText(frame, "EMERGENCY", (x1, y1-8),
                                FONT, 0.55, (40,40,230), 2, cv2.LINE_AA)
                    break

        if detected:
            em_detect_count += 1
            em_clear_count = 0
            if em_detect_count >= EM_CONFIRM_FRAMES and not emergency_active:
                emergency_active = True
                emergency_lane = detected_lane
                pending_emergency_lane = emergency_lane
                em_extension_start_time = sim_sec
                em_extension_used = 0
        else:
            em_clear_count += 1
            em_detect_count = 0
            if em_clear_count >= EM_CLEAR_FRAMES:
                emergency_active = False
                emergency_lane = None
                pending_emergency_lane = None
                em_extension_start_time = None
                em_extension_used = 0
                ETA_THRESHOLD = 120
                if (not emergency_active) and (eta is not None) and (eta < ETA_THRESHOLD):
                    emergency_active = True
                    emergency_lane = current_lane
                    pending_emergency_lane = emergency_lane
                    em_extension_start_time = sim_sec
                    em_extension_used = 0

    # ================= TRAFFIC LOGIC =================
    if sim_sec != last_print_sec:
        last_print_sec = sim_sec

        if state == "GREEN":
            green_remaining -= 1
            if emergency_active and current_lane != emergency_lane and not preempting:
                if green_remaining > 10:
                    green_remaining = 10
                    preempting = True
            if (emergency_active and current_lane == emergency_lane and
                    green_remaining <= 10 and em_extension_start_time is not None and
                    em_extension_used < MAX_EXTENSION_TOTAL):
                green_remaining += EXTEND_STEP
                em_extension_used += EXTEND_STEP
            if green_remaining <= 0:
                state = "YELLOW"
                yellow_remaining = YELLOW_TIME

        elif state == "YELLOW":
            yellow_remaining -= 1
            if yellow_remaining <= 0:
                state = "RED"

        elif state == "RED":
            if emergency_active and pending_emergency_lane:
                current_lane = pending_emergency_lane
                pending_emergency_lane = None
            else:
                current_lane = (current_lane % NUM_LANES) + 1
            state = "GREEN"
            green_remaining = GREEN_TIME
            preempting = False

        print(f"[{sim_sec:03d}s] {state} | Lane={current_lane} | "
              f"Rem={'GREEN='+str(green_remaining) if state=='GREEN' else 'YELLOW='+str(yellow_remaining) if state=='YELLOW' else 'RED'}s | "
              f"Emergency={emergency_active} | ETA={eta}")

    # ================= RENDER DASHBOARD =================
    dashboard = build_dashboard(
        frame, state, current_lane, green_remaining, yellow_remaining,
        emergency_active, emergency_lane, lane_counts, eta, sim_sec, pulse
    )

    cv2.imshow("Smart Emergency Traffic Management", dashboard)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
