import cv2
import numpy as np

def detect_card_edges_from_bytes(image_bytes, inset_pct=0.04,
                                 close_kernel=(15,15), open_kernel=(25,25)):
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    orig = cv2.imdecode(data, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    # 1. Clean mask
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, bw = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel)
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kc)
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, open_kernel)
    clean = cv2.morphologyEx(closed, cv2.MORPH_OPEN, ko)

    # 2. Outer edge via largest contour
    cnts, _ = cv2.findContours(clean,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    outer = {"left": x, "right": x+w-1, "top": y, "bottom": y+h-1}

    # 3. Inner border by inset + re-contour
    dx, dy = int(inset_pct*w), int(inset_pct*h)
    crop = clean[y+dy:y+h-dy, x+dx:x+w-dx]
    cnts, _ = cv2.findContours(crop,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    ax, ay, aw, ah = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    inner = {
        "left":   x+dx+ax,
        "right":  x+dx+ax+aw-1,
        "top":    y+dy+ay,
        "bottom": y+dy+ay+ah-1
    }

    return {"outer_edge": outer, "inner_border": inner}
