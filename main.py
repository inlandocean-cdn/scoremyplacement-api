{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import os\
import subprocess\
import tempfile\
import base64\
import httpx\
from fastapi import FastAPI, HTTPException\
from pydantic import BaseModel\
\
app = FastAPI()\
\
VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY")\
\
class AnalyzeRequest(BaseModel):\
    video_url: str\
    scan_id: str\
\
@app.get("/health")\
def health():\
    return \{"status": "ok"\}\
\
@app.post("/analyze")\
async def analyze_video(request: AnalyzeRequest):\
    with tempfile.TemporaryDirectory() as tmpdir:\
        video_path = f"\{tmpdir\}/video.mp4"\
        frames_dir = f"\{tmpdir\}/frames"\
        os.makedirs(frames_dir, exist_ok=True)\
\
        # 1. Download video\
        async with httpx.AsyncClient() as client:\
            response = await client.get(request.video_url, timeout=60)\
            if response.status_code != 200:\
                raise HTTPException(status_code=400, detail="Failed to download video")\
            with open(video_path, "wb") as f:\
                f.write(response.content)\
\
        # 2. Extract frames with FFmpeg (1 frame every 5 seconds, max 10 frames)\
        subprocess.run([\
            "ffmpeg", "-i", video_path,\
            "-vf", "fps=1/5",\
            "-frames:v", "10",\
            "-q:v", "2",\
            f"\{frames_dir\}/frame_%03d.jpg"\
        ], capture_output=True, timeout=60)\
\
        # 3. Read extracted frames\
        frame_files = sorted([\
            f for f in os.listdir(frames_dir) \
            if f.endswith(".jpg")\
        ])\
\
        if not frame_files:\
            raise HTTPException(status_code=400, detail="No frames extracted")\
\
        # 4. Convert frames to base64\
        frames_base64 = []\
        for frame_file in frame_files[:10]:\
            with open(f"\{frames_dir\}/\{frame_file\}", "rb") as f:\
                frames_base64.append(base64.b64encode(f.read()).decode())\
\
        # 5. Call Google Vision API\
        requests_payload = [\
            \{\
                "image": \{"content": frame\},\
                "features": [\
                    \{"type": "LOGO_DETECTION", "maxResults": 10\},\
                    \{"type": "OBJECT_LOCALIZATION", "maxResults": 10\},\
                    \{"type": "LABEL_DETECTION", "maxResults": 10\}\
                ]\
            \}\
            for frame in frames_base64\
        ]\
\
        async with httpx.AsyncClient() as client:\
            vision_response = await client.post(\
                f"https://vision.googleapis.com/v1/images:annotate?key=\{VISION_API_KEY\}",\
                json=\{"requests": requests_payload\},\
                timeout=30\
            )\
            vision_data = vision_response.json()\
\
        # 6. Process detections\
        brand_map = \{\}\
\
        for frame_result in vision_data.get("responses", []):\
            if frame_result.get("error"):\
                continue\
\
            # Logos\
            for logo in frame_result.get("logoAnnotations", []):\
                name = logo["description"]\
                if name not in brand_map:\
                    brand_map[name] = \{\
                        "name": name,\
                        "frame_count": 0,\
                        "total_confidence": 0,\
                        "prominence": "Low"\
                    \}\
                brand_map[name]["frame_count"] += 1\
                brand_map[name]["total_confidence"] += logo.get("score", 0.8) * 100\
                brand_map[name]["prominence"] = "High"\
\
            # Objects\
            for obj in frame_result.get("localizedObjectAnnotations", []):\
                if obj.get("score", 0) < 0.6:\
                    continue\
                name = obj["name"]\
                if name not in brand_map:\
                    brand_map[name] = \{\
                        "name": name,\
                        "frame_count": 0,\
                        "total_confidence": 0,\
                        "prominence": "Low"\
                    \}\
                brand_map[name]["frame_count"] += 1\
                brand_map[name]["total_confidence"] += obj.get("score", 0.6) * 100\
\
                verts = obj.get("boundingPoly", \{\}).get("normalizedVertices", [])\
                if len(verts) >= 3:\
                    width = abs(verts[1].get("x", 0) - verts[0].get("x", 0))\
                    height = abs(verts[2].get("y", 0) - verts[0].get("y", 0))\
                    area = width * height\
                    if area > 0.25 and brand_map[name]["prominence"] != "High":\
                        brand_map[name]["prominence"] = "High"\
                    elif area > 0.08 and brand_map[name]["prominence"] == "Low":\
                        brand_map[name]["prominence"] = "Medium"\
\
        # 7. Build brands array\
        brands = [\
            \{\
                "name": b["name"],\
                "screen_time": b["frame_count"] * 5,\
                "prominence": b["prominence"],\
                "context": (\
                    "Actively used" if b["prominence"] == "High"\
                    else "Background placement" if b["prominence"] == "Medium"\
                    else "Brief appearance"\
                ),\
                "score": min(round(b["total_confidence"] / b["frame_count"]), 100)\
            \}\
            for b in brand_map.values()\
        ]\
\
        # 8. Calculate overall score\
        if brands:\
            overall_score = min(round(\
                sum(b["score"] for b in brands) / len(brands) +\
                (10 if any(b["prominence"] == "High" for b in brands) else 0)\
            ), 100)\
        else:\
            overall_score = 45\
\
        # 9. Recommendations\
        recommendations = []\
        if not any(b["prominence"] == "High" for b in brands):\
            recommendations.append(\
                "Request center-frame placement for higher prominence scores"\
            )\
        if any(b["context"] == "Brief appearance" for b in brands):\
            recommendations.append(\
                "Increase screen time \'97 brief appearances score significantly lower"\
            )\
        if not brands:\
            recommendations.append(\
                "No brands detected \'97 ensure product/logo is clearly visible on camera"\
            )\
        recommendations.append(\
            "Aim for placement in first 5 minutes when viewer retention is highest"\
        )\
\
        return \{\
            "success": True,\
            "score": overall_score,\
            "brands": brands,\
            "recommendations": recommendations,\
            "frames_analyzed": len(frame_files)\
        \}\
```\
\
---\
\
## File 2: `requirements.txt`\
```\
fastapi\
uvicorn\
httpx\
python-multipart}