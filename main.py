# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn, uuid, os, io, httpx
import cv2, numpy as np
from PIL import Image

app = FastAPI()
DEBUG_FOLDER = "/tmp/debug"
os.makedirs(DEBUG_FOLDER, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def normalize(edges, W, H):
    return {k: round(v/(W if k in ("left","right") else H), 4) for k,v in edges.items()}

@app.get("/debug/{fn}")
def debug_img(fn: str):
    p = os.path.join(DEBUG_FOLDER, fn)
    if not os.path.exists(p):
        return JSONResponse({"error":"not found"}, 404)
    return FileResponse(p)

@app.post("/process_image")
async def proc(req: Request):
    js = await req.json()
    url = js.get("imageUrl") or js.get("image_url")
    if not url:
        return JSONResponse({"error":"no URL"},400)

    # fetch
    try:
        r = httpx.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
    except Exception as e:
        return JSONResponse({"error":f"download failed: {e}"},400)

    # load
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    npimg = np.array(img)
    H, W = npimg.shape[:2]

    # edges
    gray    = cv2.cvtColor(npimg, cv2.COLOR_RGB2GRAY)
    blur    = cv2.GaussianBlur(gray, (5,5), 0)
    edges   = cv2.Canny(blur, 50,150)

    # Hough
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180,
        threshold=80,
        minLineLength=int(min(W,H)*0.7),
        maxLineGap=40
    )
    if lines is None:
        return JSONResponse({"error":"no lines"},500)

    verts = []
    hors  = []
    for x1,y1,x2,y2 in lines[:,0]:
        if abs(x1-x2) < abs(y1-y2)*0.3:
            # vertical
            verts.append((x1+x2)//2)
        elif abs(y1-y2) < abs(x1-x2)*0.3:
            # horizontal
            hors.append((y1+y2)//2)

    if not verts or not hors:
        return JSONResponse({"error":"insufficient lines"},500)

    # cluster / average
    # top = smallest few hors, bottom = largest few
    hors = sorted(hors)
    top    = int(np.mean(hors[:max(1,len(hors)//5)]))
    bottom = int(np.mean(hors[-max(1,len(hors)//5):]))
    verts = sorted(verts)
    left   = int(np.mean(verts[:max(1,len(verts)//5)]))
    right  = int(np.mean(verts[-max(1,len(verts)//5):]))

    # debug overlay
    dbg = npimg.copy()
    cv2.line(dbg, (left,0),(left,H),(0,255,0),3)
    cv2.line(dbg, (right,0),(right,H),(0,255,0),3)
    cv2.line(dbg, (0,top),(W,top),(0,255,0),3)
    cv2.line(dbg, (0,bottom),(W,bottom),(0,255,0),3)

    fn = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(DEBUG_FOLDER, fn)
    cv2.imwrite(path, cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))

    # inner inset 10%
    w_box = right-left
    h_box = bottom-top
    ix = left  + w_box*0.10
    iy = top   + h_box*0.10
    ox = right - w_box*0.10
    oy = bottom- h_box*0.10

    return {
        "outerEdges": normalize({"left":left,"top":top,"right":right,"bottom":bottom}, W,H),
        "innerEdges": normalize({"left":ix,"top":iy,"right":ox,"bottom":oy}, W,H),
        "rotation": 0.0,
        "confidence": 0.99,
        "debug_image_url": f"/debug/{fn}"
    }

if __name__=="__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
