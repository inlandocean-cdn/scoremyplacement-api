import os
import subprocess
import tempfile
import base64
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY")

# Generic labels to ignore — not real brands
GENERIC_LABELS = {
    "bottled and jarred packaged goods", "food", "drink", "beverage",
    "product", "bottle", "can", "package", "container", "box",
    "label", "plastic", "glass", "liquid", "snack", "ingredient",
    "cuisine", "dish", "meal", "fruit", "vegetable", "tableware",
    "drinkware", "soft drink", "alcoholic beverage", "dairy product",
    "fast food", "junk food", "convenience food", "processed food",
    "tin", "jar", "bag", "wrapper", "carton", "tube"
}

# Known brand keywords to look for in text detection
BRAND_KEYWORDS = [
    "nike", "adidas", "apple", "google", "amazon", "coca-cola", "pepsi",
    "jamieson", "trader joe", "tic tac", "nestle", "kraft", "heinz",
    "samsung", "sony", "microsoft", "starbucks", "mcdonalds", "subway",
    "lululemon", "under armour", "puma", "reebok", "new balance",
    "loreal", "maybelline", "revlon", "dove", "tide", "bounty",
    "colgate", "listerine", "advil", "tylenol", "celsius", "red bull",
    "monster", "gatorade", "vitamin water", "fiji", "evian", "dasani",
]

class AnalyzeRequest(BaseModel):
    video_url: str
    scan_id: str

@app.get("/health")
def health():
    return {"status": "ok"}

def is_likely_brand(text: str) -> bool:
    """Check if text is likely a brand name vs generic label."""
    lower = text.lower().strip()
    if lower in GENERIC_LABELS:
        return False
    # Filter out very short or very long strings
    if len(lower) < 2 or len(lower) > 40:
        return False
    # Filter out strings that are just numbers
    if lower.replace(" ", "").isdigit():
        return False
    # Filter out common non-brand words
    non_brands = {
        "the", "and", "for", "with", "from", "this", "that", "are",
        "was", "has", "have", "been", "will", "can", "may", "new",
        "old", "big", "small", "best", "good", "great", "natural",
        "organic", "original", "classic", "premium", "quality", "fresh"
    }
    if lower in non_brands:
        return False
    return True

def extract_brands_from_text(text_annotations: list) -> list:
    """Extract likely brand names from text detection results."""
    if not text_annotations:
        return []
    
    # First annotation is the full text block
    full_text = text_annotations[0].get("description", "").lower() if text_annotations else ""
    
    found_brands = []
    for keyword in BRAND_KEYWORDS:
        if keyword in full_text:
            # Find the properly cased version
            for word in text_annotations[1:]:
                desc = word.get("description", "")
                if desc.lower() == keyword or keyword in desc.lower():
                    if desc not in found_brands and len(desc) > 2:
                        found_brands.append(desc)
                        break
            else:
                # Use the keyword itself capitalized
                found_brands.append(keyword.title())
    
    return found_brands

@app.post("/analyze")
async def analyze_video(request: AnalyzeRequest):
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = f"{tmpdir}/video.mp4"
        frames_dir = f"{tmpdir}/frames"
        os.makedirs(frames_dir, exist_ok=True)

        # 1. Download video
        async with httpx.AsyncClient() as client:
            response = await client.get(request.video_url, timeout=60)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download video")
            with open(video_path, "wb") as f:
                f.write(response.content)

        # 2. Extract frames with FFmpeg
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-vf", "fps=1/5",
            "-frames:v", "10",
            "-q:v", "2",
            f"{frames_dir}/frame_%03d.jpg"
        ], capture_output=True, timeout=60)

        # 3. Read extracted frames
        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.endswith(".jpg")
        ])

        if not frame_files:
            raise HTTPException(status_code=400, detail="No frames extracted")

        # 4. Convert frames to base64
        frames_base64 = []
        for frame_file in frame_files[:10]:
            with open(f"{frames_dir}/{frame_file}", "rb") as f:
                frames_base64.append(base64.b64encode(f.read()).decode())

        # 5. Call Google Vision API with TEXT_DETECTION added
        requests_payload = [
            {
                "image": {"content": frame},
                "features": [
                    {"type": "LOGO_DETECTION", "maxResults": 10},
                    {"type": "OBJECT_LOCALIZATION", "maxResults": 10},
                    {"type": "LABEL_DETECTION", "maxResults": 10},
                    {"type": "TEXT_DETECTION", "maxResults": 10}
                ]
            }
            for frame in frames_base64
        ]

        async with httpx.AsyncClient() as client:
            vision_response = await client.post(
                f"https://vision.googleapis.com/v1/images:annotate?key={VISION_API_KEY}",
                json={"requests": requests_payload},
                timeout=30
            )
            vision_data = vision_response.json()

        # 6. Process detections
        brand_map = {}

        for frame_result in vision_data.get("responses", []):
            if frame_result.get("error"):
                continue

            # Logos (highest confidence — always include)
            for logo in frame_result.get("logoAnnotations", []):
                name = logo["description"]
                if name not in brand_map:
                    brand_map[name] = {
                        "name": name,
                        "frame_count": 0,
                        "total_confidence": 0,
                        "prominence": "Low",
                        "source": "logo"
                    }
                brand_map[name]["frame_count"] += 1
                brand_map[name]["total_confidence"] += logo.get("score", 0.8) * 100
                brand_map[name]["prominence"] = "High"

            # Text detection — extract brand names from OCR
            text_annotations = frame_result.get("textAnnotations", [])
            text_brands = extract_brands_from_text(text_annotations)
            for brand_name in text_brands:
                if brand_name not in brand_map:
                    brand_map[brand_name] = {
                        "name": brand_name,
                        "frame_count": 0,
                        "total_confidence": 0,
                        "prominence": "Medium",
                        "source": "text"
                    }
                brand_map[brand_name]["frame_count"] += 1
                brand_map[brand_name]["total_confidence"] += 75
                if brand_map[brand_name]["prominence"] == "Low":
                    brand_map[brand_name]["prominence"] = "Medium"

            # Objects — only include if they pass brand filter
            for obj in frame_result.get("localizedObjectAnnotations", []):
                if obj.get("score", 0) < 0.6:
                    continue
                name = obj["name"]
                if not is_likely_brand(name):
                    continue
                if name not in brand_map:
                    brand_map[name] = {
                        "name": name,
                        "frame_count": 0,
                        "total_confidence": 0,
                        "prominence": "Low",
                        "source": "object"
                    }
                brand_map[name]["frame_count"] += 1
                brand_map[name]["total_confidence"] += obj.get("score", 0.6) * 100

                verts = obj.get("boundingPoly", {}).get("normalizedVertices", [])
                if len(verts) >= 3:
                    width = abs(verts[1].get("x", 0) - verts[0].get("x", 0))
                    height = abs(verts[2].get("y", 0) - verts[0].get("y", 0))
                    area = width * height
                    if area > 0.25 and brand_map[name]["prominence"] != "High":
                        brand_map[name]["prominence"] = "High"
                    elif area > 0.08 and brand_map[name]["prominence"] == "Low":
                        brand_map[name]["prominence"] = "Medium"

        # 7. Build brands array — filter out generics
        brands = [
            {
                "name": b["name"],
                "screen_time": b["frame_count"] * 5,
                "prominence": b["prominence"],
                "context": (
                    "Actively used" if b["prominence"] == "High"
                    else "Background placement" if b["prominence"] == "Medium"
                    else "Brief appearance"
                ),
                "score": min(round(b["total_confidence"] / b["frame_count"]), 100)
            }
            for b in brand_map.values()
            if is_likely_brand(b["name"])
        ]

        # 8. Calculate overall score
        if brands:
            overall_score = min(round(
                sum(b["score"] for b in brands) / len(brands) +
                (10 if any(b["prominence"] == "High" for b in brands) else 0)
            ), 100)
        else:
            # No real brands detected — low score
            overall_score = 30

        # 9. Recommendations
        recommendations = []
        if not brands:
            recommendations.append(
                "No brands detected — ensure product logo is clearly visible on camera"
            )
        if brands and not any(b["prominence"] == "High" for b in brands):
            recommendations.append(
                "Request center-frame placement for higher prominence scores"
            )
        if brands and any(b["context"] == "Brief appearance" for b in brands):
            recommendations.append(
                "Increase screen time — brief appearances score significantly lower"
            )
        recommendations.append(
            "Aim for placement in first 5 minutes when viewer retention is highest"
        )

        return {
            "success": True,
            "score": overall_score,
            "brands": brands,
            "recommendations": recommendations,
            "frames_analyzed": len(frame_files)
        }
