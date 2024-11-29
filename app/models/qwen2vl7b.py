import logging
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# 设置日志级别为 DEBUG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cache_dir = "./model_dir"
model_name = "Qwen/Qwen2-VL-7B-Instruct"

# Load the model and processor
logging.info("Loading model and processor...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
min_pixels = 256*28*28
max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels,cache_dir=cache_dir)

def generate_output(image_url, prompt):
    logging.info("Generating output...")
    
    # Open the image
    logging.info(f"image_url: {image_url}")
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    logging.info(f"Prepared messages: {messages}")
    
    # Prepare for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    logging.info(f"Generated text: {text}")
    
    image_inputs, video_inputs = process_vision_info(messages)
    logging.info(f"Processed image inputs: {image_inputs}")
    logging.info(f"Processed video inputs: {video_inputs}")
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    logging.info(f"Processed inputs: {inputs}")
    
    inputs = inputs.to("cuda")
    logging.info("Moved inputs to CUDA")
    
    # Inference
    logging.info("Starting inference...")
    # Use mixed precision for inference
    with torch.cuda.amp.autocast():
        # Inference
        logging.info("Starting inference...")
        generated_ids = model.generate(**inputs, max_new_tokens=64)  # Reduce max_new_tokens
        logging.info(f"Generated IDs: {generated_ids}")
    # generated_ids = model.generate(**inputs, max_new_tokens=128)
    logging.info(f"Generated IDs: {generated_ids}")
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    logging.info(f"Trimmed generated IDs: {generated_ids_trimmed}")
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    logging.info(f"Decoded output text: {output_text}")
    
    return output_text[0]