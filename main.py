# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn, uuid, os, io, httpx
import cv2, numpy as np
from PIL import Image

app = FastAPI()

# where debug overlays will be written
DEBUG_FOLDER = "/tmp/debug"
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def normalize_edges(edges: dict, w: int, h: int):
    """Convert pixel coords to normalized [0,1] form."""
    return {k: float(f"{v/(w if k in ('left','right') else h):.4f}") for k,v in edges.items()}

@app.get("/debug/{fn}")
def debug_image(fn: str):
    path = os.path.join(DEBUG_FOLDER, fn)
    if not os.path.exists(path):
        return JSONResponse({"error":"not found"}, status_code=404)
    return FileResponse(path)

@app.post("/process_image")
async def process_image(req: Request):
    data = await req.json()
    image_url = data.get("imageUrl") or data.get("image_url")
    if not image_url:
        return JSONResponse({"error":"no imageUrl provided"}, status_code=400)

    # 1) fetch
    try:
        headers = {"User-Agent":"Mozilla/5.0"}
        r = httpx.get(image_url, headers=headers, timeout=10.0)
        r.raise_for_status()
    except Exception as e:
        return JSONResponse({"error":f"download failed: {e}"}, status_code=400)

    # 2) load
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    img_np = np.array(img)
    H, W = img_np.shape[:2]

    # 3) preprocess
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred= cv2.GaussianBlur(gray, (5,5), 0)
    edges  = cv2.Canny(blurred, 50, 150)

    # 4) Hough line detection
    raw = cv2.HoughLinesP(
        edges, 
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=int(min(W,H)*0.5),
        maxLineGap=20
    )
    if raw is None:
        return JSONResponse({"error":"no lines detected"}, status_code=500)

    # 5) classify lines
    verts, hors = [], []
    for x1,y1,x2,y2 in raw[:,0]:
        dx,dy = x2-x1, y2-y1
        if abs(dx) < abs(dy)*0.3:
            # nearly vertical
            verts.append((x1,y1,x2,y2))
        elif abs(dy) < abs(dx)*0.3:
            # nearly horizontal
            hors.append((x1,y1,x2,y2))
    if not verts or not hors:
        return JSONResponse({"error":"insufficient directional lines"}, status_code=500)

    # 6) pick extremes
    left_xs  = [min(x1,x2) for x1,_,x2,_ in verts]
    right_xs = [max(x1,x2) for x1,_,x2,_ in verts]
    top_ys   = [min(y1,y2) for _,y1,_,y2 in hors]
    bot_ys   = [max(y1,y2) for _,y1,_,y2 in hors]

    left   = min(left_xs)
    right  = max(right_xs)
    top    = min(top_ys)
    bottom = max(bot_ys)

    # 7) overlay for debug
    dbg = img_np.copy()
    # draw selected border in green
    cv2.line(dbg, (left,0),(left,H), (0,255,0),3)
    cv2.line(dbg, (right,0),(right,H),(0,255,0),3)
    cv2.line(dbg, (0,top),(W,top), (0,255,0),3)
    cv2.line(dbg, (0,bottom),(W,bottom),(0,255,0),3)

    fn = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(DEBUG_FOLDER, fn)
    cv2.imwrite(path, cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))

    # 8) inner inset (10%)
    w_box = right-left
    h_box = bottom-top
    ix = left  + w_box*0.10
    iy = top   + h_box*0.10
    ox = right - w_box*0.10
    oy = bottom- h_box*0.10

    # 9) respond
    return {
        "outerEdges": normalize_edges(
            {"left":left, "top":top, "right":right, "bottom":bottom}, W,H
        ),
        "innerEdges": normalize_edges(
            {"left":ix, "top":iy, "right":ox, "bottom":oy}, W,H
        ),
        "rotation": 0.0,       # we can add rect-based angle if needed
        "confidence": 0.98,
        "debug_image_url": f"/debug/{fn}"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
