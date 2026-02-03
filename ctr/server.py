import io
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from predict import CTRPredictor
import uvicorn
import os

app = FastAPI(title="Food CTR Prediction API")

# 全局加载模型（只加载一次）
predictor = None

@app.on_event("startup")
async def load_model():
    global predictor
    predictor = CTRPredictor(
        checkpoint_path='./output/seed_42_split_80/checkpoint_epoch_2.pt',
        bert_model_name="models--hfl--chinese-bert-wwm-ext",
        image_model_name="models--facebook--convnextv2-huge-22k-512"
    )
    print("Model loaded successfully!")


@app.post("/predict")
async def predict_ctr(
    food_name: str = Form(...),
    food_type: str = Form(...),
    city_name: str = Form(...),
    shop_name: str = Form(...),
    image:  UploadFile = File(...)
):
    """
    预测餐品的 CTR
    
    - food_name: 菜品名称
    - food_type:  菜品类型
    - city_name: 城市名称
    - shop_name: 店铺名称
    - image: 餐品图片文件
    """
    temp_path = None  # Initialize temp_path before try block
    try:
        # 读取上传的图片
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 保存临时文件（或直接修改 predict 方法支持 PIL Image）
        temp_path = f"/tmp/{image.filename}"
        pil_image.save(temp_path)
        
        # 预测
        ctr = predictor.predict(
            food_name=food_name,
            food_type=food_type,
            city_name=city_name,
            shop_name=shop_name,
            image_path=temp_path
        )
        
        return JSONResponse({
            "success": True,
            "data": {
                "ctr":  round(ctr, 6),
                "ctr_percent": f"{ctr * 100:.2f}%"
            }
        })
        
    except Exception as e: 
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

    finally:
        # 删除临时文件
        if temp_path is not None and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Failed to remove temp file {temp_path}: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": predictor is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8199)