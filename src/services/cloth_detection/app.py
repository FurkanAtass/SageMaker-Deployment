import asyncio
import io
import json
import os
from contextlib import asynccontextmanager

import fastapi
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from PIL import Image

from src.models.cloth_detection.model import ClothDetectionYOLO

load_dotenv()
env = os.getenv("ENV").lower()

UPDATE_INTERVAL_SECONDS = int(os.getenv("MODEL_UPDATE_INTERVAL", "300"))

model = ClothDetectionYOLO(env=env)

async def _update_loop():
    while True:
        try:
            await asyncio.to_thread(model.update_model_version)
        except Exception as e:
            print(f"Model update check failed: {e}")
        await asyncio.sleep(UPDATE_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_update_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)

@app.get("/ping")
async def ping():
    health = model is not None
    status = 200 if health else 404
    content = json.dumps({"status": "healthy"}) if health else json.dumps({})
    return fastapi.Response(content=content, status_code=status, media_type="application/json")


@app.post("/invocations")
async def cloth_detection(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    results = model.predict([image])
    return {"results": [json.loads(r.to_json()) for r in results]}
