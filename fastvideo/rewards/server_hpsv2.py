import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import base64
import io
import uvicorn
import open_clip
import glob

# Configuration - set via environment variables
MODEL_PATH = os.environ.get("HPSV2_PATH", "")  # Path to HPSv2 checkpoint directory
PORT = 8163

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
preprocess = None
tokenizer = None


class ImageRequest(BaseModel):
    image_base64: str
    prompt: str


def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


@app.on_event("startup")
def load_model():
    global model, preprocess, tokenizer
    print(f"[HPSv2] Loading from {MODEL_PATH}")
    
    pt_files = glob.glob(os.path.join(MODEL_PATH, "*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in {MODEL_PATH}")
    checkpoint_path = pt_files[0]
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14',
        pretrained=None,
        precision='amp',
        device=device,
        force_quick_gelu=False,
        pretrained_image=False
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    print("[HPSv2] Loaded successfully.")


@app.post("/score")
async def score(req: ImageRequest):
    try:
        image = decode_image(req.image_base64)
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        text_tensor = tokenizer([req.prompt]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            hps_score = (image_features @ text_features.T).item()

        return {"score": hps_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
