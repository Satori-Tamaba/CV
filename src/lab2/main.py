"""Трекинг плоского объекта: SIFT + гомография + оптический поток (LK).
1-й кадр видео - эталон. pip install opencv-python numpy"""
import cv2
import numpy as np

# ---------- настройки ----------
VIDEO = "vid/mona-lisa.avi"
OUT = "out/result.avi"
LABEL = "object"
BOX = None
SHOW = True           # окно просмотра (Esc - выход)
# --------------------------------
MIN_INL, MIN_TRACK = 10, 8
RATIO, RANSAC = 0.8, 4.0
SCALES = (1.0, 0.5, 0.25)    # масштабы эталона (объект далеко -> мелкий)
REDETECT = 15                # раз в N кадров поправлять дрейф детектором
SMOOTH = 0.7
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2)
clahe = cv2.createCLAHE(2.0, (8, 8))


def prep(img):
    return clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def homography(src, dst, need):
    if len(src) < need:
        return None
    H, m = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC)
    if H is None or m.sum() < need:
        return None
    m = m.ravel() > 0
    return H, src[m], dst[m]


def detect(ref_pts, ref_des, gray):
    kp, des = sift.detectAndCompute(gray, None)
    if des is None or len(kp) < 2:
        return None
    good = [p[0] for p in bf.knnMatch(ref_des, des, k=2)
            if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    src = ref_pts[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    return homography(src, dst, MIN_INL)


def flow(prev, gray, ref_pts, pts):
    nxt, s1, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, **LK)
    back, s2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, nxt, None, **LK)
    ok = (s1.ravel() == 1) & (s2.ravel() == 1) & (np.linalg.norm(pts - back, axis=2).ravel() < 1.5)
    return homography(ref_pts[ok], nxt[ok], MIN_TRACK)


def valid(q, prev, shape):
    p = q.reshape(4, 2)
    if not cv2.isContourConvex(p):
        return False
    a = cv2.contourArea(p)
    if not 0.001 * shape[0] * shape[1] < a < 4 * shape[0] * shape[1]:
        return False
    if prev is None:
        return True
    b = cv2.contourArea(prev.reshape(4, 2))
    return 0.5 < a / max(b, 1e-6) < 2 and np.linalg.norm(p - prev.reshape(4, 2), axis=1).mean() < 0.6 * b ** 0.5


def main():
    cap = cv2.VideoCapture(VIDEO)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"не удалось открыть {VIDEO}")
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = BOX or (0, 0, w, h)
    ref = frame[y0:y1, x0:x1]
    corners = np.float32([[0, 0], [x1 - x0, 0], [x1 - x0, y1 - y0], [0, y1 - y0]]).reshape(-1, 1, 2)

    # эталон: точки на нескольких масштабах (координаты приведены к исходному размеру)
    pts, des = [], []
    for s in SCALES:
        img = cv2.resize(ref, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        kp, d = sift.detectAndCompute(prep(img), None)
        if d is not None:
            pts += [(k.pt[0] / s, k.pt[1] / s) for k in kp]
            des.append(d)
    ref_pts, ref_des = np.float32(pts), np.vstack(des)

    out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 30, (w, h))
    prev_gray = cur = rp = prev_q = shown = None
    n = found = 0
    while ok:
        n += 1
        gray = prep(frame)
        q = None
        if cur is not None:                              # 1) оптический поток
            r = flow(prev_gray, gray, rp, cur)
            if r:
                qq = cv2.perspectiveTransform(corners, r[0])
                if valid(qq, prev_q, frame.shape):
                    q, rp, cur = qq, r[1], r[2]
        if q is None or n % REDETECT == 0:               # 2) SIFT: поиск / коррекция дрейфа
            r = detect(ref_pts, ref_des, gray)
            if r:
                qq = cv2.perspectiveTransform(corners, r[0])
                if valid(qq, prev_q, frame.shape) or q is None and valid(qq, None, frame.shape):
                    q, rp, cur = qq, r[1], r[2]
        prev_gray = gray
        if q is None:
            cur = prev_q = shown = None
            cv2.putText(frame, "not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            found += 1
            prev_q = q
            shown = q if shown is None else SMOOTH * shown + (1 - SMOOTH) * q
            p = shown.reshape(4, 2)
            cv2.polylines(frame, [p.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
            tx, ty = p.min(axis=0).astype(int)
            cv2.putText(frame, LABEL, (max(tx, 5), max(ty - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        out.write(frame)
        if SHOW:
            cv2.imshow("tracking", frame)
            if cv2.waitKey(1) == 27:
                break
        ok, frame = cap.read()
    out.release()
    cv2.destroyAllWindows()
    print(f"кадров: {n}, объект найден: {found} ({100 * found / n:.1f}%) -> {OUT}")


if __name__ == "__main__":
    main()