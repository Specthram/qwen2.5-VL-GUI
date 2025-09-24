import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import gradio as gr
import os
import shutil
import zipfile
from functools import partial
import json

# --- 1. Configuration ---

INPUT_DIR = "input"
if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)

PROMPT_FILE = "prompts.json"
MODEL_ID = "Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed"

# Global variables for the model and processor
model = None
processor = None

# --- Gallery Configuration ---
ITEMS_PER_PAGE = 40  # Number of images to display per page. Adjust as needed.
ITEMS_PER_ROW = 5    # Number of items per row in the grid.

# --- CSS for styling ---
CSS = """
/* Grid Item Styling for consistent height */
.image-card {
    display: flex !important;
    flex-direction: column !important;
    height: 100%; /* Make card fill the column height */
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    overflow: hidden;
}
.image-card .image-container {
    height: 250px; /* Fixed height for image container */
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #f5f5f5;
}
.image-card .image-container img {
    width: 100%;
    height: 100%;
    object-fit: cover; /* Ensures image covers the area without distortion */
}
.image-card textarea {
    flex-grow: 1; /* Allows the textbox to fill available space */
    resize: none !important; /* Prevent manual resizing */
}
.image-card .action-buttons-row {
    flex-shrink: 0; /* Prevent button row from shrinking */
}

/* Locked Item Styling */
.locked-item { background-color: #f0f0f0; }
.locked-item .image-container img { filter: grayscale(80%) opacity(0.7); }
.locked-item textarea { background-color: #e9e9e9 !important; }

/* Action Button Sizing */
.action-buttons { min-width: 40px !important; max-width: 40px !important; }
"""



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
        "Simple Description" : "in few words, describe this scene content.",
        "Detailed Description": "Describe this scene in detail.",
        "custom_1": """You are a specialized scene analysis system. All output must be strictly factual, objective, and devoid of any personal opinions, judgments, or biases.
Your task is to generate concise and factually descriptive captions for each image provided.
Captions must be precise, comprehensive, and meticulously aligned with the visual content depicted in the image and any given tags.
Caption Style: Generate concise captions that are no more than 50 words.
Focus on combining multiple descriptors into small phrases.
Follow this structure: "A <subject> doing <action>, they are wearing <clothes>. The background is <background description>. <Additional camera, lighting, or style information>.
If tags are present, consolidate tags into descriptive phrases where possible, such as "frilled black dress" instead of "dress, frilled dress, black dress"."""
    }
    save_prompts(default_prompts)
    return default_prompts

def save_prompts(prompts_dict):
    """Saves the prompts dictionary to the JSON file."""
    with open(PROMPT_FILE, 'w', encoding='utf-8') as f:
        json.dump(prompts_dict, f, indent=4, ensure_ascii=False)

def load_model():
    """Loads the model and processor into VRAM."""
    global model, processor
    if model is None:
        print("Starting model load... This may take several minutes.")
        try:
            processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID, dtype=torch.bfloat16, device_map="auto"
            )
            print("Model loaded successfully onto the GPU.")
        except Exception as e:
            print(f"CRITICAL ERROR while loading model: {e}")
            raise

def get_image_files():
    """Returns a sorted list of image file paths in the 'input' directory."""
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
    files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)
             if os.path.splitext(f)[1].lower() in supported_extensions]
    return sorted(files)

def read_caption_file(image_path):
    """Reads the caption from the corresponding .txt file, if it exists."""
    txt_path = os.path.splitext(image_path)[0] + '.txt'
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""

def generate_caption(image_path, prompt, temperature, seed):
    """Generates a caption for a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
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
        print(f"Error generating caption for {image_path}: {e}")
        return f"Processing Error: {e}"

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
    load_model()
    yield {
        status_output: gr.Textbox(value="Model ready. You can now generate captions.", interactive=False),
        start_button: gr.Button(interactive=True)
    }

def process_all_images(prompt, lock_states, temperature, seed, progress=gr.Progress(track_tqdm=True)):
    """Processes all images in the directory, skipping locked ones."""
    images = get_image_files()
    if not images:
        return "No images found in the 'input' folder.", lock_states

    unlocked_images = [img for img in images if not lock_states.get(os.path.basename(img), False)]

    if not unlocked_images:
        return f"Process finished. All {len(images)} images are locked.", lock_states

    print(f"Starting captioning for {len(unlocked_images)} unlocked images with prompt: '{prompt}'")

    for image_path in progress.tqdm(unlocked_images, desc="Generating captions..."):
        caption = generate_caption(image_path, prompt, temperature, seed)
        save_caption(image_path, caption)

    return f"Process finished! {len(unlocked_images)} new captions generated.", lock_states

def handle_upload(files, lock_states):
    for file_obj in files:
        shutil.copy(file_obj.name, INPUT_DIR)

    image_files = get_image_files()
    for img_path in image_files:
        basename = os.path.basename(img_path)
        if basename not in lock_states:
            lock_states[basename] = False

    return lock_states, 1 # Return to page 1 after upload

def create_zip_archive():
    """Creates a zip archive of the input directory."""
    zip_path = "input_archive.zip"
    print("Creating ZIP archive...")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(INPUT_DIR):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)
    print(f"Archive created: {zip_path}")
    return zip_path

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

def delete_all_unlocked_images(lock_states):
    """Deletes all images and their captions that are not locked."""
    images_to_delete = [
        os.path.basename(img) for img in get_image_files()
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
    image_path = os.path.join(INPUT_DIR, image_basename)
    caption = generate_caption(image_path, prompt, temperature, seed)
    save_caption(image_path, caption)
    return caption

def refresh_gallery_ui(lock_states, page_num=1):
    """Refreshes the gallery UI for a given page."""
    image_files = get_image_files()
    total_images = len(image_files)
    total_pages = (total_images + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE if total_images > 0 else 1
    page_num = max(1, min(page_num, total_pages))

    start_index = (page_num - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    page_image_files = image_files[start_index:end_index]

    for img_path in image_files:
        basename = os.path.basename(img_path)
        if basename not in lock_states:
            lock_states[basename] = False

    ui_updates = []
    for i in range(ITEMS_PER_PAGE):
        if i < len(page_image_files):
            img_path = page_image_files[i]
            basename = os.path.basename(img_path)
            caption = read_caption_file(img_path)
            is_locked = lock_states.get(basename, False)

            column_classes = ["image-card"]
            if is_locked:
                column_classes.append("locked-item")

            ui_updates.extend([
                gr.Column(visible=True, elem_classes=column_classes),
                gr.Image(value=img_path, visible=True),
                gr.Textbox(value=caption, label=basename, interactive=not is_locked, visible=True),
                gr.Button(value="🔒" if is_locked else "🔓", visible=True),
                gr.Button(visible=True),
                gr.Button(visible=True),
                gr.Textbox(value=basename, visible=False)
            ])
        else:
            ui_updates.extend([gr.Column(visible=False)] + [gr.update(visible=False)] * 6)

    page_info = f"Page {page_num} / {total_pages} ({total_images} images)"
    prev_button_interactive = page_num > 1
    next_button_interactive = page_num < total_pages

    return [lock_states, page_num, page_info, gr.Button(interactive=prev_button_interactive), gr.Button(interactive=next_button_interactive)] + ui_updates

def load_and_refresh_prompts():
    """Loads prompts from file and updates the dropdown menu."""
    prompts = load_prompts()
    default_prompt_title = list(prompts.keys())[0] if prompts else ""
    return prompts, gr.Dropdown(choices=list(prompts.keys()), value=default_prompt_title), prompts.get(default_prompt_title, "")

def update_prompt_from_dropdown(selected_title, prompts_dict):
    """Updates the prompt text field from the dropdown selection."""
    return prompts_dict.get(selected_title, "")

def add_new_prompt(title, content, prompts_dict):
    """Adds a new prompt, saves it, and updates the UI."""
    if not title or not content:
        current_selection = list(prompts_dict.keys())[0] if prompts else ""
        return prompts_dict, gr.Dropdown(choices=list(prompts.keys()), value=current_selection)

    prompts_dict[title] = content
    save_prompts(prompts_dict)

    return prompts_dict, gr.Dropdown(choices=list(prompts_dict.keys()), value=title)

# --- 4. Create the Gradio Interface ---

with gr.Blocks(theme=gr.themes.Soft(), title="Image Captioning with Qwen-VL", css=CSS) as app:
    gr.Markdown("# Image Captioning Tool with Qwen2.5-VL")
    gr.Markdown("Drop your images, adjust the prompt, lock captions you want to keep, then start the process.")
    gr.Markdown("images and captions are stored in ./input folder.")

    lock_states = gr.State({})
    prompts_state = gr.State({})
    current_page = gr.State(1)

    with gr.Row():
        with gr.Column(scale=3):
            file_uploader = gr.File(label="Drop images here", file_count="multiple", file_types=["image"])
        with gr.Column(scale=1):
             download_zip_button = gr.Button("Download all as .ZIP")
             status_output = gr.Textbox(label="Status", interactive=False, value="Initializing...", lines=3)
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
                            img = gr.Image(show_label=False, type="filepath", show_download_button=False)
                        caption_text = gr.Textbox(show_label=True, interactive=True, lines=4)
                        with gr.Row(elem_classes=["action-buttons-row"]):
                            lock_button = gr.Button("🔓", elem_classes="action-buttons")
                            delete_button = gr.Button("🗑️", elem_classes="action-buttons")
                            single_caption_button = gr.Button("🤖", elem_classes="action-buttons")
                        hidden_filename = gr.Textbox(visible=False)
                        image_components.append({
                            "col": col, "img": img, "caption": caption_text,
                            "lock": lock_button, "delete": delete_button,
                            "single_caption": single_caption_button,
                            "hidden_filename": hidden_filename
                        })

    # --- 5. Connect Events ---
    flat_ui_outputs = []
    for comp in image_components:
        flat_ui_outputs.extend([comp["col"], comp["img"], comp["caption"], comp["lock"], comp["delete"], comp["single_caption"], comp["hidden_filename"]])

    app.load(
        fn=load_and_refresh_prompts,
        outputs=[prompts_state, prompt_selector, prompt_input]
    ).then(
        fn=refresh_gallery_ui,
        inputs=[lock_states, current_page],
        outputs=[lock_states, current_page, page_indicator, prev_button, next_button] + flat_ui_outputs
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

    download_zip_button.click(fn=create_zip_archive, outputs=download_output).then(lambda: gr.File(visible=True), outputs=download_output)
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
        fn=delete_all_unlocked_images, inputs=[lock_states], outputs=[status_output, confirmation_group]
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
        # --- MODIFICATION: Ajout de temp_slider et seed_input aux entrées
        comp["single_caption"].click(
            fn=process_single_image, inputs=[comp["hidden_filename"], prompt_input, temp_slider, seed_input], outputs=[comp["caption"]]
        )

# --- 6. Launch the App ---

if __name__ == "__main__":
    app.queue().launch()