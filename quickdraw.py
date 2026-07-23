import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import random
from game_logic import PlayerState, resolve_round

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0,5), (5, 6), (6,7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]

# All colors are BGR (OpenCV order, not RGB!)
PINK = {
    "hot": (180, 105, 255),
    "deep": (147, 20, 255),
    "magenta": (255, 0, 255),
    "orchid": (214, 112, 218),
    "light": (193, 182, 255),
    "blush": (220, 209, 255),
    "muted": (90, 70, 110),
    "bg_dark": (35, 10, 30),
}

FINGER_COLORS = {
    "thumb": PINK["deep"],
    "index": PINK["hot"],
    "middle": PINK["orchid"],
    "ring": PINK["magenta"],
    "pinky": PINK["light"],
}

LANDMARK_GROUP = (
    ["wrist"] +
    ["thumb"] * 4 +
    ["index"] * 4 +
    ["middle"] * 4 +
    ["ring"] * 4 +
    ["pinky"] * 4
)

VALID_MOVES = ["reload", "shoot", "sheath", "slash", "shield", "deflect"]

GUN_ICON = [
    "..............................",
    "........#.................#...",
    "...#######################....",
    "...#######################....",
    "...#######################....",
    "...#######################....",
    "...#################..........",
    "...#############..............",
    "...#####....####..............",
    "...#####...#...#..............",
    "...#####...#...#..............",
    "...#####....###...............",
    "....#####.....................",
    "....#####.....................",
    ".....#####....................",
    ".....#####....................",
    "......#####...................",
    "......#####...................",
]

SWORD_ICON = [
    "........#........",
    ".......###.......",
    ".......###.......",
    "......#####......",
    "......#####......",
    "......#####......",
    "......#####......",
    "......#####......",
    "......#####......",
    "......#####......",
    "......#####......",
    "......#####......",
    "......#####......",
    "......#####......",
    "......#####......",
    "......#####......",
    ".....#######.....",
    "....#########....",
    "#################",
    ".###############.",
    ".......###.......",
    ".......###.......",
    ".......###.......",
    ".......###.......",
    "......#####......",
    "......#####......",
    ".......###.......",
]

def draw_round_banner(frame, text, color):
    h, w = frame.shape[:2]
    band_y1, band_y2 = h // 2 - 60, h // 2 + 40

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, band_y1), (w, band_y2), PINK["bg_dark"], -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4)
    tx = (w - tw) // 2
    ty = band_y1 + (band_y2 - band_y1) // 2 + th // 2
    cv2.putText(frame, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 4)

def draw_match_over_banner(frame, winner_text, p1, p2):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), PINK["bg_dark"], -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    (tw, th), _ = cv2.getTextSize(winner_text, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 6)
    tx = (w - tw) // 2
    ty = h // 2 - 20
    cv2.putText(frame, winner_text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, PINK["hot"], 6)

    score_text = f"score - you: {p1.round_wins} ai: {p2.round_wins}"
    (sw, sh), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    sx = (w - sw) // 2
    cv2.putText(frame, score_text, (sx, ty + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, PINK["blush"], 2)

def draw_pixel_icon(frame, pattern, x, y, color, block=4):
    for row_idx, row in enumerate(pattern):
        for col_idx, cell in enumerate(row):
            if cell == '#':
                px = x + col_idx * block
                py = y + row_idx * block
                cv2.rectangle(frame, (px, py), (px + block, py + block), color, -1)


def draw_hud_panel(frame, x, y, w, h):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), PINK["bg_dark"], -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), PINK["hot"], 1)


def draw_weapon_status(frame, player, label, x, y):
    panel_w, panel_h = 170, 110
    draw_hud_panel(frame, x, y, panel_w, panel_h)

    cv2.putText(frame, label, (x + 10, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, PINK["blush"], 2)

    gun_color = PINK["hot"] if player.gun_loaded else PINK["muted"]
    sword_color = PINK["hot"] if player.sword_sheathed else PINK["muted"]

    draw_pixel_icon(frame, GUN_ICON, x + 15, y + 35, gun_color, block=3)
    draw_pixel_icon(frame, SWORD_ICON, x + 110, y + 25, sword_color, block=3)


def random_ai_move():
    return random.choice(VALID_MOVES)


def fingers_up_pattern(lm):
    return [lm[t].y < lm[p].y for t, p in zip(FINGER_TIPS, FINGER_PIPS)]


def is_thumb_up(lm):
    index, middle, ring, pinky = fingers_up_pattern(lm)
    if index or middle or ring or pinky:
        return False
    return lm[4].y < lm[3].y < lm[2].y and lm[4].y < lm[0].y


def is_thumb_down(lm):
    index, middle, ring, pinky = fingers_up_pattern(lm)
    if index or middle or ring or pinky:
        return False
    return lm[4].y > lm[3].y > lm[2].y and lm[4].y > lm[0].y


def is_L_frame(lm):
    index, middle, ring, pinky = fingers_up_pattern(lm)
    if middle or ring or pinky or not index:
        return False
    pointing_up = lm[8].y < lm[6].y
    thumb_sideways = abs(lm[4].x - lm[2].x) > 0.05
    return pointing_up and thumb_sideways


def landmark_dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def is_finger_extended(lm, tip_idx, pip_idx, mcp_idx):
    return landmark_dist(lm[tip_idx], lm[mcp_idx]) > landmark_dist(lm[pip_idx], lm[mcp_idx]) * 1.3


def is_hammer_cocked(lm):
    return lm[4].y < lm[3].y and lm[4].y < lm[2].y


def is_gun_shape(lm):
    index_ext = is_finger_extended(lm, 8, 6, 5)
    middle_ext = is_finger_extended(lm, 12, 10, 9)
    ring_ext = is_finger_extended(lm, 16, 14, 13)
    pinky_ext = is_finger_extended(lm, 20, 18, 17)
    return index_ext and middle_ext and not ring_ext and not pinky_ext


def is_hammer_dropped(lm):
    return lm[4].y > lm[3].y


def is_reload(lm):
    return is_gun_shape(lm) and is_hammer_cocked(lm)


def is_shoot(lm):
    return is_gun_shape(lm) and is_hammer_dropped(lm)


def is_deflect(hands):
    if len(hands) < 2:
        return False
    a, b = hands[0], hands[1]
    if not (is_L_frame(a) and is_L_frame(b)):
        return False
    thumbs_close = abs(a[4].x - b[4].x) < 0.18 and abs(a[4].y - b[4].y) < 0.18
    return thumbs_close


def is_shield(hands):
    if len(hands) < 2:
        return False
    wrist_a, wrist_b = hands[0][0], hands[1][0]
    close_together = abs(wrist_a.x - wrist_b.x) < 0.22
    chest_height = 0.25 < wrist_a.y < 0.8 and 0.25 < wrist_b.y < 0.8
    return close_together and chest_height


def classify_action(hands, handedness_labels):
    if len(hands) < 2:
        return None

    a, b = hands[0], hands[1]

    if is_reload(a) and is_reload(b):
        return "reload"
    if is_shoot(a) and is_shoot(b):
        return "shoot"
    if is_thumb_down(a) and is_thumb_down(b):
        return "sheath"
    if is_thumb_up(a) and is_thumb_up(b):
        return "slash"
    if is_deflect(hands):
        return "deflect"
    if is_shield(hands):
        return "shield"

    return None


def draw_hand(frame, hand_landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start, end in HAND_CONNECTIONS:
        group = LANDMARK_GROUP[end]
        color = FINGER_COLORS.get(group, PINK["blush"])
        cv2.line(frame, points[start], points[end], color, 2)
    for i, (x, y) in enumerate(points):
        group = LANDMARK_GROUP[i]
        color = FINGER_COLORS.get(group, PINK["blush"])
        cv2.circle(frame, (x, y), 5, color, -1)


cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0

current_action = None
round_start_time = time.time()
ROUND_LENGTH = 3.0

p1 = PlayerState("you")
p2 = PlayerState("ai")
last_result_text = ""
match_over = False

round_banner_text = ""
round_banner_color = PINK["blush"]
round_banner_until = 0
BANNER_DURATION = 1.4

while True:
    success, frame = cap.read()
    if not success:
        print("failed to read from webcam")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    frame_timestamp_ms += 33
    result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            draw_hand(frame, hand_landmarks)

        labels = [h[0].category_name for h in result.handedness]
        detected = classify_action(result.hand_landmarks, labels)

        if detected:
            current_action = detected

    elapsed = time.time() - round_start_time
    time_left = max(0, ROUND_LENGTH - elapsed)

    if elapsed >= ROUND_LENGTH and not match_over:
        p1_move = current_action or "reload"
        p2_move = random_ai_move()
        outcome = resolve_round(p1, p1_move, p2, p2_move)

        last_result_text = f"you: {p1_move} | ai: {p2_move} -> {outcome}"
        print(last_result_text)

        current_action = None
        round_start_time = time.time()

        if outcome == "round_veto":
            last_result_text += " [ROUND VETO] "
            round_banner_text, round_banner_color = "CLASH - VETO", PINK["light"]
        elif outcome == p1.name:
            round_banner_text, round_banner_color = "ROUND WON", PINK["hot"]
        elif outcome == p2.name:
            round_banner_text, round_banner_color = "ROUND LOST", PINK["muted"]
        else:
            round_banner_text, round_banner_color = "", PINK["blush"]

        round_banner_until = time.time() + BANNER_DURATION

        if p1.round_wins >= 3:
            match_over = True
            last_result_text = "[WIN]"
        elif p2.round_wins >= 3:
            match_over = True
            last_result_text = "[LOSS]"

    cv2.putText(frame, f"move: {current_action or '...'}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, PINK["hot"], 2)
    cv2.putText(frame, f"time left: {time_left:.1f}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, PINK["light"], 2)
    cv2.putText(frame, f"score: [you] {p1.round_wins} [ai] {p2.round_wins}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, PINK["blush"], 2)
    cv2.putText(frame, last_result_text, (30, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, PINK["magenta"], 2)

    draw_weapon_status(frame, p1, "you", 30, 170)
    h, w = frame.shape[:2]
    draw_weapon_status(frame, p2, "ai", w - 200, 170)

    if round_banner_text and time.time() < round_banner_until:
        draw_round_banner(frame, round_banner_text, round_banner_color)

    if match_over:
        winner_text = "VICTORY" if p1.round_wins > p2.round_wins else "DEFEAT"
        draw_match_over_banner(frame, winner_text, p1, p2)

    cv2.imshow("quickdraw", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()