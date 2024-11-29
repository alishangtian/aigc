import os
import logging
import uuid
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Request
from app.models.qwen2vl7b import generate_output
import yaml

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# 从配置文件中加载配置
def load_config():
    try:
        with open("config.yaml", "r") as config_file:
            config = yaml.safe_load(config_file)
            logger.info("Configuration loaded successfully.")
            return config
    except FileNotFoundError:
        logger.error("Configuration file not found.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuration file not found"
        )
    except yaml.YAMLError:
        logger.error("Invalid configuration file format.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid configuration file format"
        )

config = load_config()
allowed_api_keys = set(config.get("allowed_api_keys", []))
temp_save_dir = config.get("temp_save_dir", "/data2/maoxiaobing/.tmp")

logger.info("Allowed API Keys: %s", allowed_api_keys)
logger.info("Temporary save directory: %s", temp_save_dir)

# 依赖函数来检查API Key
async def verify_api_key(request: Request):
    x_apikey = request.headers.get("x-apikey")
    if x_apikey not in allowed_api_keys:
        logger.warning(f"Unauthorized access attempt with API key: {x_apikey}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Request"
        )
    logger.info(f"API key verified: {x_apikey}")
    return x_apikey

@app.post("/recognize")
async def recognize_image(
    image: UploadFile = File(...),
    prompt: str = "",
    api_key: str = Depends(verify_api_key)
):
    try:
        logger.info("Processing image recognition request.")
        os.makedirs(temp_save_dir, exist_ok=True) 
        
        # 使用 uuid 生成唯一的文件名
        file_name = f"{uuid.uuid4()}.{image.filename.split('.')[-1]}"
        file_path = os.path.join(temp_save_dir, file_name)
        
        with open(file_path, "wb") as buffer:
            buffer.write(await image.read())
        logger.info(f"Image saved to: {file_path}")
        file_url = f"file://{file_path}"
        # 使用模型生成输出
        output = generate_output(file_url, prompt)
        logger.info("Output generated successfully.")
        return {"output": output}
    except Exception as e:
        logger.error(f"Error processing image recognition request: {str(e)}")
        return {"error": str(e)}