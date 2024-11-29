import logging
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Request
from app.models.qwen2vl7b import generate_output
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# 从配置文件中加载 API Key
def load_api_keys():
    try:
        with open("config.json", "r") as config_file:
            config = json.load(config_file)
            api_keys = config.get("allowed_api_keys", [])
            logger.info(f"Loaded {len(api_keys)} API keys from config file.")
            return set(api_keys)
    except FileNotFoundError:
        logger.error("Configuration file not found.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuration file not found"
        )
    except json.JSONDecodeError:
        logger.error("Invalid configuration file format.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid configuration file format"
        )

allowed_api_keys = load_api_keys()
logger.info("Allowed API Keys: %s", allowed_api_keys)

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
        # 读取图片内容
        image_content = await image.read()
        logger.info("Image content read successfully.")
        # 使用模型生成输出
        output = generate_output(image_content, prompt)
        logger.info("Output generated successfully.")
        return {"output": output}
    except Exception as e:
        logger.error(f"Error processing image recognition request: {str(e)}")
        return {"error": str(e)}