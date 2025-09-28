import cv2
import gradio as gr
import gc
import json
import os
import shutil
import torch
from functools import partial
from PIL import Image
from src.zip import create_zip_archive
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# --- 1. Configuration ---

INPUT_DIR = "input"
CONFIG_DIR = "config"
if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)
if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)

PROMPT_FILE = CONFIG_DIR+"/prompts.json"
MODELS_FILE = CONFIG_DIR+"/models.json"


RESIZE_IMAGE_SIZE = 1024
FRAMES_TO_EXTRACT_PER_VIDEO = 1 # basically caption the first frame.

# Global variables for the model and processor
model = None
processor = None

# --- Gallery Configuration ---
ITEMS_PER_PAGE = 40  # Number of images to display per page. Adjust as needed. WARNING : due to bad algo, must be divisible by ITEMS_PER_ROW
ITEMS_PER_ROW = 4    # Number of items per row in the grid.

# --- CSS for styling ---
CSS_PATH = "./src/css/style.css"



# --- 2. Core Functions ---

def load_prompts():
    """Loads prompts from the JSON file, or creates a default file."""
    if os.path.exists(PROMPT_FILE):
        try:
            with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    default_prompts = {
        "Simple Description" : "Describe this image.",
        "Detailed Description": "Describe this image, with every single detail.",
    }
    save_prompts(default_prompts)
    return default_prompts

def load_or_create_models_config():
    """Loads model configurations from JSON, or creates a default file."""
    if os.path.exists(MODELS_FILE):
        try:
            with open(MODELS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass # Fallback to creating the default file

    # Default models if the file doesn't exist or is invalid
    default_models = {
        "Qwen2.5-VL-7B (NSFW Caption V3)": "thesby/Qwen2.5-VL-7B-NSFW-Caption-V3",
        "Qwen2.5-VL-7B (Relaxed Captioner)": "Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed"
    }
    with open(MODELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(default_models, f, indent=4, ensure_ascii=False)
    return default_models


def save_prompts(prompts_dict):
    """Saves the prompts dictionary to the JSON file."""
    with open(PROMPT_FILE, 'w', encoding='utf-8') as f:
        json.dump(prompts_dict, f, indent=4, ensure_ascii=False)

def load_model(model_id):
    """Loads the selected model and processor into VRAM."""
    global model, processor
    if model is None:
        status = f"⏳ Loading model '{model_id}'... please wait.\n" # <-- MODIFICATION : Affiche le nom du modèle
        yield status, False
        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map="auto"
            )
            status += "✅ Model loaded successfully.\n"
            yield status, True
        except Exception as e:
            status += f"❌ ERROR while loading: {e}\n"
            yield status, False
    else:
        yield "✅ Model already loaded.", True

def unload_model():
    """Unloads the model and processor from VRAM."""
    global model, processor

    if model is None:
        return "⚠️ No model to unload.", False

    status = "♻️ Unloading model...\n"
    yield status, True
    try:
        if model is not None:
            del model
            model = None
        if processor is not None:
            del processor
            processor = None

        # clean VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # clean RAM
        gc.collect()

        status += "✅ Model unloaded successfully.\n"
        yield status, False
    except Exception as e:
        status += f"❌ ERROR while unloading: {e}\n"
        yield status, True

def on_model_change(is_loaded):
    """Called when the user selects a different model. Forces unload."""
    if is_loaded:
        unload_gen = unload_model()
        status, loaded_state = "", False
        for status, loaded_state in unload_gen:
            pass
        status += "\nModel selection changed. Please click 'Load Model'."
        return status, loaded_state
    return "Model selection changed. Ready to load.", False

def is_model_loaded():
    return model is not None


def get_media_files():
    """Returns a sorted list of media file paths in the 'input' directory."""
    supported_image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
    supported_video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    supported_extensions = supported_image_extensions + supported_video_extensions

    files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)
             if os.path.splitext(f)[1].lower() in supported_extensions]
    return sorted(files)

def is_video_file(filename):
    """Checks if a file is a video based on its extension."""
    supported_video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    return os.path.splitext(filename)[1].lower() in supported_video_extensions

def read_caption_file(image_path):
    """Reads the caption from the corresponding .txt file, if it exists."""
    txt_path = os.path.splitext(image_path)[0] + '.txt'
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""


def generate_captions_for_video(video_path, prompt, temperature, seed):
    """Extracts frames from a video, generates a caption for each, and combines them."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: Could not open video file."

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames < FRAMES_TO_EXTRACT_PER_VIDEO:
            frame_indices = [int(i) for i in range(total_frames)]
        else:
            frame_indices = [int(i) for i in torch.linspace(0, total_frames - 1, FRAMES_TO_EXTRACT_PER_VIDEO)]

        combined_caption = ""

        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                # Convertir l'image de OpenCV (BGR) en PIL (RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Générer la légende pour cette image
                frame_caption = generate_caption(pil_image, prompt, temperature,
                                                 seed)  # On passe l'objet image directement

                # Ajouter un horodatage
                timestamp = frame_index / fps
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)

                combined_caption += f"[{minutes:02d}:{seconds:02d}] {frame_caption}\n"

        cap.release()
        return combined_caption.strip()

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return f"Video Processing Error: {e}"


def generate_caption(image_source, prompt, temperature, seed):  # 'image_path' devient 'image_source'
    """Generate a caption for a given image (path or PIL object)."""
    try:
        if isinstance(image_source, str):
            image = Image.open(image_source).convert("RGB")
        else:
            image = image_source

        max_size = RESIZE_IMAGE_SIZE
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size))

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt", truncation=True).to(model.device, dtype=torch.bfloat16)
        generation_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": False,
        }

        if temperature > 0.0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature

        if seed != -1 and generation_kwargs["do_sample"]:
            torch.manual_seed(int(seed))

        generated_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids = [out[len(ins):] for ins, out in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return response

    except Exception as e:
        print(f"Error generating caption: {e}")
        return f"Processing Error: {e}"
    finally:
        import gc
        torch.cuda.empty_cache()
        gc.collect()

def save_caption(image_path, caption):
    """Saves the caption to a .txt file."""
    txt_path = os.path.splitext(image_path)[0] + '.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(caption)
    print(f"Caption saved for {os.path.basename(image_path)}")

# --- 3. Gradio Interface Functions ---

def initialize_app_state():
    """Generator to manage the UI state on load. This function now ONLY handles model loading."""
    yield {
        status_output: gr.Textbox(value="Loading model into VRAM... Please wait.", interactive=False),
        start_button: gr.Button(interactive=False)
    }
    yield {
        status_output: gr.Textbox(value="Model ready. You can now generate captions.", interactive=False),
        start_button: gr.Button(interactive=True)
    }

def process_all_images(prompt, lock_states, temperature, seed, progress=gr.Progress(track_tqdm=True)):
    """Processes all media in the directory, skipping locked ones."""
    media_files = get_media_files()
    if not media_files:
        return "No media found in the 'input' folder.", lock_states

    unlocked_media = [mf for mf in media_files if not lock_states.get(os.path.basename(mf), False)]

    if not unlocked_media:
        return f"Process finished. All {len(media_files)} media are locked.", lock_states

    print(f"Starting captioning for {len(unlocked_media)} unlocked media files with prompt: '{prompt}'")

    for media_path in progress.tqdm(unlocked_media, desc="Generating captions..."):
        if is_video_file(media_path):
            caption = generate_captions_for_video(media_path, prompt, temperature, seed)
        else:
            caption = generate_caption(media_path, prompt, temperature, seed)
        save_caption(media_path, caption)

    return f"Process finished! {len(unlocked_media)} new captions generated.", lock_states

def handle_upload(files, lock_states):
    for file_obj in files:
        shutil.copy(file_obj.name, INPUT_DIR)

    image_files = get_media_files()
    for img_path in image_files:
        basename = os.path.basename(img_path)
        if basename not in lock_states:
            lock_states[basename] = False

    return lock_states, 1 # Return to page 1 after upload

def toggle_lock(image_basename, current_states):
    """Toggles the lock state of an image."""
    current_states[image_basename] = not current_states.get(image_basename, False)
    new_lock_status = current_states[image_basename]

    new_button_text = "🔒" if new_lock_status else "🔓"
    textbox_interactive = not new_lock_status
    column_classes = ["image-card"]
    if new_lock_status:
        column_classes.append("locked-item")

    return current_states, gr.Button(value=new_button_text), gr.Textbox(interactive=textbox_interactive), gr.Column(elem_classes=column_classes)

def delete_file_pair(image_basename):
    """Deletes an image and its associated .txt file."""
    print(f"Attempting to delete {image_basename}...")
    base_path = os.path.splitext(os.path.join(INPUT_DIR, image_basename))[0]
    img_path = None
    txt_path = base_path + ".txt"

    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
        if os.path.exists(base_path + ext):
            img_path = base_path + ext
            break

    if img_path and os.path.exists(img_path):
        os.remove(img_path)
        print(f"Image deleted: {img_path}")
    if os.path.exists(txt_path):
        os.remove(txt_path)
        print(f"Caption deleted: {txt_path}")

def delete_all_images(lock_states):
    """Deletes all images and their captions that are not locked."""
    images_to_delete = [
        os.path.basename(img) for img in get_media_files()
        if not lock_states.get(os.path.basename(img), False)
    ]
    
    if not images_to_delete:
        return "No unlocked images to delete.", gr.Group(visible=False)

    print(f"Deleting {len(images_to_delete)} unlocked images.")
    for basename in images_to_delete:
        delete_file_pair(basename)

    return f"{len(images_to_delete)} unlocked images have been deleted.", gr.Group(visible=False)

def process_single_image(image_basename, prompt, temperature, seed):
    """Generates a caption for a single image and updates its text field."""
    print(f"Generating single caption for {image_basename}...")
    media_path = os.path.join(INPUT_DIR, image_basename)
    if is_video_file(media_path):
        # Si c'est une vidéo, on appelle la fonction de traitement vidéo
        caption = generate_captions_for_video(media_path, prompt, temperature, seed)
    else:
        # Sinon, c'est une image, on utilise la fonction originale
        caption = generate_caption(media_path, prompt, temperature, seed)

    # La sauvegarde et le retour du résultat ne changent pas
    save_caption(media_path, caption)
    return caption

def save_caption_from_ui(image_basename, new_caption):
    if image_basename:
        media_path = os.path.join(INPUT_DIR, image_basename)
        save_caption(media_path, new_caption)
        print(f"Caption for {image_basename} updated via UI edit.")

def refresh_gallery_ui(lock_states, page_num=1):
    """Refreshes the gallery UI for a given page."""
    media_files = get_media_files()  # Renommé
    total_media = len(media_files)
    total_pages = (total_media + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE if total_media > 0 else 1
    page_num = max(1, min(page_num, total_pages))

    start_index = (page_num - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    page_media_files = media_files[start_index:end_index]

    for media_path in media_files:
        basename = os.path.basename(media_path)
        if basename not in lock_states:
            lock_states[basename] = False

    ui_updates = []
    for i in range(ITEMS_PER_PAGE):
        if i < len(page_media_files):
            media_path = page_media_files[i]
            basename = os.path.basename(media_path)
            caption = read_caption_file(media_path)
            is_locked = lock_states.get(basename, False)
            is_vid = is_video_file(media_path)

            column_classes = ["image-card"]
            if is_locked:
                column_classes.append("locked-item")

            # Ici, on choisit d'afficher l'image OU la vidéo
            ui_updates.extend([
                gr.Column(visible=True, elem_classes=column_classes),
                gr.Image(value=media_path if not is_vid else None, visible=not is_vid),
                gr.Video(value=media_path if is_vid else None, visible=is_vid),
                gr.Textbox(value=caption, label=basename, interactive=not is_locked, visible=True),
                gr.Button(value="🔒" if is_locked else "🔓", visible=True),
                gr.Button(visible=True),
                gr.Button(visible=True),
                gr.Textbox(value=basename, visible=False)
            ])
        else:
            # S'assurer de cacher les 8 composants
            ui_updates.extend([gr.Column(visible=False)] + [gr.update(visible=False)] * 7)

    page_info = f"Page {page_num} / {total_pages} ({total_media} media)"
    prev_button_interactive = page_num > 1
    next_button_interactive = page_num < total_pages

    return [lock_states, page_num, page_info, gr.Button(interactive=prev_button_interactive),
            gr.Button(interactive=next_button_interactive)] + ui_updates

def load_and_refresh_prompts():
    """Loads prompts from file and updates the dropdown menu."""
    prompts = load_prompts()
    default_prompt_title = list(prompts.keys())[0] if prompts else ""
    return prompts, gr.Dropdown(choices=list(prompts.keys()), value=default_prompt_title), prompts.get(default_prompt_title, "")

def update_prompt_from_dropdown(selected_title, prompts_dict):
    """Updates the prompt text field from the dropdown selection."""
    return prompts_dict.get(selected_title, "")

def add_new_prompt(title, content, prompts_dict, prompts=None):
    """Adds a new prompt, saves it, and updates the UI."""
    if not title or not content:
        current_selection = list(prompts_dict.keys())[0] if prompts else ""
        return prompts_dict, gr.Dropdown(choices=list(prompts.keys()), value=current_selection)

    prompts_dict[title] = content
    save_prompts(prompts_dict)

    return prompts_dict, gr.Dropdown(choices=list(prompts_dict.keys()), value=title)

def load_model_wrapper(selected_name, models_dict):
    model_id = models_dict.get(selected_name)
    if not model_id:
        yield "Erreur : Nom du modèle non trouvé dans la configuration.", False
        return
    yield from load_model(model_id)

# --- 4. Create the Gradio Interface ---

with gr.Blocks(theme=gr.themes.Soft(), title="Caption Forge", css_paths=CSS_PATH) as app:
    gr.Markdown("# Caption Forge")
    gr.Markdown("Drop your images, adjust the prompt, lock captions you want to keep, then start the process.")
    gr.Markdown("images and captions are stored in ./input folder.")

    is_model_loaded_state = gr.State(is_model_loaded())
    models_state = gr.State({})

    lock_states = gr.State({})
    prompts_state = gr.State({})
    current_page = gr.State(1)

    with gr.Row():
        with gr.Column(scale=3):
            file_uploader = gr.File(label="Drop images or videos here", file_count="multiple", file_types=["image", "video"])
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(label="Select Model", interactive=True)
            load_model_button = gr.Button("Load Model", visible=not is_model_loaded())
            unload_model_button = gr.Button("Unload Model", visible=is_model_loaded())
            status_output = gr.Textbox(label="Status", interactive=False, value="Initializing...", lines=3)
            download_zip_button = gr.Button("Download all as .ZIP")
            download_output = gr.File(label="Download Link", visible=False)


    gr.Markdown("### Prompt Configuration")
    with gr.Row(equal_height=True):
        prompt_selector = gr.Dropdown(label="Choose a prompt")
        new_prompt_title = gr.Textbox(label="save the current prompt as", placeholder="Enter a new title...")
        save_prompt_button = gr.Button("Save Prompt")

    prompt_input = gr.Textbox(label="Prompt", lines=3, interactive=True)

    with gr.Row():
        temp_slider = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.05, label="Temperature",
                                info="Higher values increase creativity/randomness. 0.0 is deterministic.")
        seed_input = gr.Number(value=-1, label="Seed", precision=0,
                               info="Set a specific seed for reproducible results. -1 means random.")
    
    start_button = gr.Button("Generate captions for unlocked images", variant="primary", interactive=False)

    gr.Markdown("---")
    gr.Markdown("## Image Gallery & Captions")

    with gr.Group(visible=False) as confirmation_group:
        gr.Markdown("### Are you sure you want to delete all unlocked images? This action is irreversible.")
        with gr.Row():
            cancel_delete_button = gr.Button("Cancel")
            confirm_delete_button = gr.Button("Yes, delete all", variant="stop")

    with gr.Row():
        prev_button = gr.Button("⬅️ Previous", interactive=False)
        page_indicator = gr.Textbox("Page 1 / 1", interactive=False, text_align="center", show_label=False)
        next_button = gr.Button("Next ➡️", interactive=False)
        delete_all_button = gr.Button("🗑️ Delete all", variant="stop")

    image_components = []
    with gr.Blocks():
        for i in range(0, ITEMS_PER_PAGE, ITEMS_PER_ROW):
            with gr.Row():
                for j in range(ITEMS_PER_ROW):
                    with gr.Column(visible=False, scale=1, elem_classes=["image-card"]) as col:
                        with gr.Group(elem_classes=["image-container"]):
                            img = gr.Image(show_label=False, type="filepath", show_download_button=False, visible=False)
                            vid = gr.Video(show_label=False, show_download_button=False, visible=False)
                        caption_text = gr.Textbox(show_label=True, interactive=True, lines=4)
                        with gr.Row(elem_classes=["action-buttons-row"]):
                            lock_button = gr.Button("🔓", elem_classes="action-buttons")
                            # save_button = gr.Button("💾", elem_classes="action-buttons")
                            delete_button = gr.Button("🗑️", elem_classes="action-buttons")
                            single_caption_button = gr.Button("🤖", elem_classes="action-buttons")
                        hidden_filename = gr.Textbox(visible=False)
                        image_components.append({
                            "col": col,
                            "img": img,
                            "vid": vid,
                            "caption": caption_text,
                            "lock": lock_button,
                            "delete": delete_button,
                            "single_caption": single_caption_button,
                            "hidden_filename": hidden_filename
                        })

    # --- 5. Connect Events ---
    flat_ui_outputs = []
    for comp in image_components:
        flat_ui_outputs.extend([comp["col"], comp["img"], comp["vid"], comp["caption"], comp["lock"], comp["delete"], comp["single_caption"], comp["hidden_filename"]])

    app.load(
        fn=load_and_refresh_prompts,
        outputs=[prompts_state, prompt_selector, prompt_input]
    ).then(
        fn=refresh_gallery_ui,
        inputs=[lock_states, current_page],
        outputs=[lock_states, current_page, page_indicator, prev_button, next_button] + flat_ui_outputs
    ).then(
    fn=load_or_create_models_config,
    outputs=[models_state]
    ).then(
    fn=lambda models_dict: gr.Dropdown(choices=list(models_dict.keys()), value=list(models_dict.keys())[0]),
    inputs=[models_state],
    outputs=[model_selector]
)

    app.load(
        fn=initialize_app_state,
        outputs=[status_output, start_button]
    )

    file_uploader.upload(
        fn=handle_upload, inputs=[file_uploader, lock_states], outputs=[lock_states, current_page]
    ).then(
        fn=refresh_gallery_ui, inputs=[lock_states, current_page],
        outputs=[lock_states, current_page, page_indicator, prev_button, next_button] + flat_ui_outputs
    )

    # --- MODIFICATION: Ajout de temp_slider et seed_input aux entrées
    start_button.click(
        fn=process_all_images, inputs=[prompt_input, lock_states, temp_slider, seed_input], outputs=[status_output, lock_states]
    ).then(
        fn=refresh_gallery_ui, inputs=[lock_states, current_page],
        outputs=[lock_states, current_page, page_indicator, prev_button, next_button] + flat_ui_outputs
    )

    download_zip_button.click(
        fn=partial(create_zip_archive, INPUT_DIR),
        outputs=download_output
    ).then(
        lambda: gr.File(visible=True),
        outputs=download_output
    )

    load_model_button.click(
        fn=load_model_wrapper,
        inputs=[model_selector, models_state],
        outputs=[status_output, is_model_loaded_state]
    ).then(
        lambda loaded: gr.update(visible=not loaded),
        inputs=[is_model_loaded_state],
        outputs=load_model_button
    ).then(
        lambda loaded: gr.update(visible=loaded),
        inputs=[is_model_loaded_state],
        outputs=unload_model_button
    )

    unload_model_button.click(
        fn=unload_model,
        outputs=[status_output, is_model_loaded_state]
    ).then(
        lambda loaded: gr.update(
            visible=not loaded,
        ),
        inputs=[is_model_loaded_state],
        outputs=load_model_button
    ).then(
        lambda loaded: gr.update(
            visible=loaded,
        ),
        inputs=[is_model_loaded_state],
        outputs=unload_model_button
    )

    model_selector.change(
        fn=on_model_change,
        inputs=[is_model_loaded_state],
        outputs=[status_output, is_model_loaded_state]
    ).then(
        lambda loaded: gr.update(visible=not loaded),
        inputs=[is_model_loaded_state],
        outputs=load_model_button
    ).then(
        lambda loaded: gr.update(visible=loaded),
        inputs=[is_model_loaded_state],
        outputs=unload_model_button
    )

    prompt_selector.change(fn=update_prompt_from_dropdown, inputs=[prompt_selector, prompts_state], outputs=[prompt_input])
    save_prompt_button.click(
        fn=add_new_prompt, inputs=[new_prompt_title, prompt_input, prompts_state], outputs=[prompts_state, prompt_selector]
    ).then(lambda: gr.Textbox(value=""), outputs=new_prompt_title)

    prev_button.click(
        fn=lambda p: p - 1, inputs=[current_page], outputs=[current_page]
    ).then(
        fn=refresh_gallery_ui, inputs=[lock_states, current_page],
        outputs=[lock_states, current_page, page_indicator, prev_button, next_button] + flat_ui_outputs
    )

    next_button.click(
        fn=lambda p: p + 1, inputs=[current_page], outputs=[current_page]
    ).then(
        fn=refresh_gallery_ui, inputs=[lock_states, current_page],
        outputs=[lock_states, current_page, page_indicator, prev_button, next_button] + flat_ui_outputs
    )
    
    delete_all_button.click(fn=lambda: gr.Group(visible=True), outputs=[confirmation_group])
    cancel_delete_button.click(fn=lambda: gr.Group(visible=False), outputs=[confirmation_group])
    confirm_delete_button.click(
        fn=delete_all_images, inputs=[lock_states], outputs=[status_output, confirmation_group]
    ).then(
        fn=refresh_gallery_ui, inputs=[lock_states, current_page],
        outputs=[lock_states, current_page, page_indicator, prev_button, next_button] + flat_ui_outputs
    )

    for i in range(ITEMS_PER_PAGE):
        comp = image_components[i]
        comp["lock"].click(
            fn=toggle_lock, inputs=[comp["hidden_filename"], lock_states],
            outputs=[lock_states, comp["lock"], comp["caption"], comp["col"]]
        )
        comp["delete"].click(
            fn=delete_file_pair, inputs=[comp["hidden_filename"]]
        ).then(
            fn=refresh_gallery_ui, inputs=[lock_states, current_page],
            outputs=[lock_states, current_page, page_indicator, prev_button, next_button] + flat_ui_outputs
        )

        comp["single_caption"].click(
            fn=process_single_image, inputs=[comp["hidden_filename"], prompt_input, temp_slider, seed_input], outputs=[comp["caption"]]
        )

        comp["caption"].blur(
            fn=save_caption_from_ui,
            inputs=[comp["hidden_filename"], comp["caption"]],
            outputs=None
        )

# --- 6. Launch the App ---

if __name__ == "__main__":
    app.queue().launch()
