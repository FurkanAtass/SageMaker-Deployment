import asyncio
import io
import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, status, Response
from PIL import Image

from src.models.cloth_detection.model import ClothDetectionYOLO

load_dotenv()
env = os.getenv("ENV").lower()

UPDATE_INTERVAL_SECONDS = int(os.getenv("MODEL_UPDATE_INTERVAL", "300"))

model = ClothDetectionYOLO(env=env)

#model.update_model_version()

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
    return Response(
        content=json.dumps({"status": "healthy"}), 
        status_code=status.HTTP_200_OK, 
        media_type="application/json"
    )


@app.post("/invocations")
async def cloth_detection(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    model_resp = model.predict([image])
    results = {"results": [json.loads(r.to_json()) for r in model_resp]}
    
    response = Response(
        content = json.dumps(results),
        status_code=status.HTTP_200_OK,
        media_type="application/json"
    )
    return response
