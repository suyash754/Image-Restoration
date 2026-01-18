from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import subprocess

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict/")
async def predict_api(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    model_path: str = Form(default=r"C:\Users\suyas\OneDrive\Desktop\partialconv-master\partialconv-master\pretrained_pconv.pth")
):
    try:
        os.makedirs("input_images", exist_ok=True)
        img_path = f"input_images/{image.filename}"
        mask_path = f"input_images/{mask.filename}"

        with open(img_path, "wb") as f1, open(mask_path, "wb") as f2:
            shutil.copyfileobj(image.file, f1)
            shutil.copyfileobj(mask.file, f2)

        print("🚀 Running model...")
        cmd = [
            "python", "predict.py",
            "--img", img_path,
            "--mask", mask_path,
            "--model", model_path,
            "--resize", "True"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        print("✅ STDOUT:\n", result.stdout)
        print("❌ STDERR:\n", result.stderr)

        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        output_path = f"output/restored_{image.filename}"
        if not os.path.exists(output_path):
            raise RuntimeError("Output file not found!")

        return FileResponse(output_path, media_type="image/png")

    except Exception as e:
        print("❗ BACKEND ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
