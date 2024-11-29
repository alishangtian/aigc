from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Request
from app.models.qwen2vl7b import generate_output
import json

app = FastAPI()

# 从配置文件中加载 API Key
def load_api_keys():
    try:
        with open("config.json", "r") as config_file:
            config = json.load(config_file)
            return set(config.get("allowed_api_keys", []))
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuration file not found"
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid configuration file format"
        )

allowed_api_keys = load_api_keys()
print("Allowed API Keys:", allowed_api_keys)

# 依赖函数来检查API Key
async def verify_api_key(request: Request):
    x_apikey = request.headers.get("x-apikey")
    if x_apikey not in allowed_api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Request"
        )
    return x_apikey

@app.post("/recognize")
async def recognize_image(
    image: UploadFile = File(...),
    prompt: str = "",
    api_key: str = Depends(verify_api_key)
):
    try:
        # 读取图片内容
        image_content = await image.read()
        # 使用模型生成输出
        output = generate_output(image_content, prompt)
        return {"output": output}
    except Exception as e:
        return {"error": str(e)}