import json
import os

CONFIG_DIR = "config"
MODELS_FILE = CONFIG_DIR+"/models.json"
PROMPT_FILE = CONFIG_DIR+"/prompts.json"

def load_or_create_models_config():
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)

    """Loads model configurations from JSON, or creates a default file."""
    if os.path.exists(MODELS_FILE):
        try:
            with open(MODELS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass # Fallback to creating the default file

    # Default models if the file doesn't exist or is invalid
    default_models = {
        "Qwen2.5-VL-7B (Relaxed Captioner)": {
            "model": "Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed",
            "type": "qwen2.5",
            "default": True
        },
        "Qwen2.5-VL-32B-Instruct": {
            "model": "unsloth/Qwen2.5-VL-32B-Instruct",
            "type": "qwen2.5",
            "default": False
        },
        "Qwen2.5-VL-7B (NSFW Caption V4)": {
            "model": "thesby/Qwen2.5-VL-7B-NSFW-Caption-V4",
            "type": "qwen2.5",
            "default": False
        },
        "JoyCaption Beta One": {
            "model": "fancyfeast/llama-joycaption-beta-one-hf-llava",
            "type": "llava",
            "default": False
        },
    }
    with open(MODELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(default_models, f, indent=4, ensure_ascii=False)
    return default_models

def load_prompts():
    """Loads prompts from the JSON file, or creates a default file."""
    if os.path.exists(PROMPT_FILE):
        try:
            with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    default_prompts = {
        "Simple Description" : {
            "prompt": "Describe this image.",
            "type": "qwen2.5",
            "default": False
        },
        "Detailed Description": {
            "prompt": "Describe this image, with every single detail.",
            "type": "qwen2.5",
            "default": True
        },
        "llama - Descriptive": {
            "prompt": "Write a long descriptive caption for this image in a formal tone.",
            "type": "llava",
            "default": True
        }
    }
    save_prompts(default_prompts)
    return default_prompts

def save_prompts(prompts_dict):
    """Saves the prompts dictionary to the JSON file."""
    with open(PROMPT_FILE, 'w', encoding='utf-8') as f:
        json.dump(prompts_dict, f, indent=4, ensure_ascii=False)