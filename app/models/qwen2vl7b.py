import logging
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

# 设置日志级别为 DEBUG
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cache_dir = "./model_dir"
model_name = "Qwen/Qwen2-VL-7B-Instruct"

# Load the model and processor
logging.info("Loading model and processor...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto", cache_dir=cache_dir
)
processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)

def generate_output(image_content, prompt):
    logging.info("Generating output...")
    
    # Open the image
    # logging.debug(f"Opening image: {image_content}")
    image = Image.open(image_content)
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    logging.debug(f"Prepared messages: {messages}")
    
    # Prepare for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    logging.debug(f"Generated text: {text}")
    
    image_inputs, video_inputs = process_vision_info(messages)
    logging.debug(f"Processed image inputs: {image_inputs}")
    logging.debug(f"Processed video inputs: {video_inputs}")
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    logging.debug(f"Processed inputs: {inputs}")
    
    inputs = inputs.to("cuda")
    logging.debug("Moved inputs to CUDA")
    
    # Inference
    logging.info("Starting inference...")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    logging.debug(f"Generated IDs: {generated_ids}")
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    logging.debug(f"Trimmed generated IDs: {generated_ids_trimmed}")
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    logging.debug(f"Decoded output text: {output_text}")
    
    return output_text[0]