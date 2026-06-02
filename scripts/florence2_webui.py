import os
import gradio as gr
from PIL import Image
from modules import scripts
from florence_engine import FlorenceEngine

# List of models supported by default
model_list = [
    'microsoft/Florence-2-base',
    'microsoft/Florence-2-base-ft',
    'microsoft/Florence-2-large',
    'microsoft/Florence-2-large-ft',
    'MiaoshouAI/Florence-2-large-PromptGen-v2.0',
    'MiaoshouAI/Florence-2-base-PromptGen-v1.5'
]

engine = FlorenceEngine()

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as florence_tab:
        gr.Markdown("# Florence-2 Vision-Language Companion")
        gr.Markdown("An integrated suite for descriptive captioning, object detection, phrase grounding, segmentations, OCR, and smart inpaint mask generation.")
        
        with gr.Row():
            # Global Loader Settings Panel (Left column)
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 🛠️ Model Loader Settings")
                model_sel = gr.Dropdown(choices=model_list, value='microsoft/Florence-2-large-ft', label="Model Variant")
                precision_sel = gr.Dropdown(choices=['fp16', 'bf16', 'fp32'], value='fp16', label="Precision")
                device_sel = gr.Dropdown(choices=['auto', 'cuda', 'cpu'], value='auto', label="Execution Device")
                
                with gr.Row():
                    btn_load = gr.Button("🔌 Load Model", variant="primary")
                    btn_unload = gr.Button("⚠️ Unload Model", variant="stop")
                    
                status_text = gr.Markdown("**Status:** Model not loaded")
                
            # Main Processing Panel (Right column)
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Single Image"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                input_img = gr.Image(type="pil", label="Input Image")
                                task_sel = gr.Dropdown(
                                    choices=[
                                        "Caption",
                                        "Detailed Caption",
                                        "More Detailed Caption",
                                        "Object Detection",
                                        "Dense Region Caption",
                                        "Region Proposal",
                                        "Caption to Phrase Grounding",
                                        "Referring Expression Segmentation",
                                        "OCR",
                                        "OCR with Region"
                                    ],
                                    value="Detailed Caption",
                                    label="Task Type"
                                )
                                prompt_text = gr.Textbox(
                                    label="Phrase Prompt (for Phrase Grounding / Segmentation)",
                                    placeholder="e.g. 'the blue sky' or 'dog'",
                                    visible=True
                                )
                                btn_run = gr.Button("🚀 Run Vision Task", variant="primary")
                                
                            with gr.Column(scale=1):
                                output_text = gr.Textbox(label="Result Description", interactive=False, show_copy_button=True)
                                visualized_img = gr.Image(type="pil", label="Visualized Output")
                                output_mask = gr.Image(type="pil", label="Generated Inpaint Mask")
                                
                                with gr.Row():
                                    btn_send_txt = gr.Button("📋 Send Prompt to txt2img", variant="secondary")
                                    btn_send_img = gr.Button("📋 Send Prompt to img2img", variant="secondary")
                                    
                    with gr.Tab("Batch Dataset Processing"):
                        gr.Markdown("### 📂 Bulk Image Dataset Preprocessing (LoRA Training Preparation)")
                        gr.Markdown("Generate high-quality matching descriptive captions/tags next to all images in a target directory.")
                        
                        batch_dir = gr.Textbox(label="Dataset Directory Path", placeholder="e.g. C:\\dataset\\my_lora_images")
                        
                        with gr.Row():
                            batch_task = gr.Dropdown(
                                choices=["Caption", "Detailed Caption", "More Detailed Caption"],
                                value="Detailed Caption",
                                label="Caption Task"
                            )
                            write_mode = gr.Checkbox(label="Overwrite Existing TXT Files", value=True)
                            
                        with gr.Row():
                            prefix_text = gr.Textbox(label="Prepend Tag/Trigger Word", placeholder="e.g. '1girl, masterpiece, '")
                            suffix_text = gr.Textbox(label="Append Tag/Trigger Word", placeholder="e.g. ', detailed background'")
                            
                        btn_batch_run = gr.Button("⚡ Start Batch Captioning", variant="primary")
                        batch_status = gr.HTML(value="Status: Idle", label="Batch Progress Log")

        # Bind loader functions
        def load_fn(name, prec, dev):
            try:
                res = engine.load_model(name, prec, dev)
                return f"**Status:** {res}"
            except Exception as e:
                return f"**Status Error:** {str(e)}"
                
        def unload_fn():
            res = engine.unload_model()
            return f"**Status:** {res}"

        btn_load.click(load_fn, inputs=[model_sel, precision_sel, device_sel], outputs=[status_text])
        btn_unload.click(unload_fn, outputs=[status_text])

        # Bind run task function
        def run_fn(img, task, p_text):
            if img is None:
                return "Error: No image uploaded.", None, None
            try:
                if engine.model is None:
                    engine.load_model(model_sel.value, precision_sel.value, device_sel.value)
                    
                text_res, vis_img, mask_img = engine.run_task(img, task, p_text)
                return text_res, vis_img, mask_img
            except Exception as e:
                return f"Execution Error: {str(e)}", None, None

        btn_run.click(run_fn, inputs=[input_img, task_sel, prompt_text], outputs=[output_text, visualized_img, output_mask])

        # Bind javascript functions to send text
        btn_send_txt.click(
            fn=None,
            inputs=[output_text],
            outputs=[],
            _js="(val) => { if (typeof send_florence_prompt === 'function') { send_florence_prompt(val, 'txt2img'); } else { alert('Helper JS not loaded yet'); } }"
        )
        
        btn_send_img.click(
            fn=None,
            inputs=[output_text],
            outputs=[],
            _js="(val) => { if (typeof send_florence_prompt === 'function') { send_florence_prompt(val, 'img2img'); } else { alert('Helper JS not loaded yet'); } }"
        )

        # Batch runner
        def batch_run_fn(directory, task, overwrite, prefix, suffix, progress=gr.Progress()):
            if not directory or not os.path.exists(directory):
                return "Status: Error - Invalid or non-existent folder directory."
                
            valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
            img_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(valid_exts)]
            
            if not img_paths:
                return "Status: Error - No images found in dataset directory."

            total = len(img_paths)
            print(f"[Florence-2] Starting batch processing for {total} images...")
            
            if engine.model is None:
                engine.load_model(model_sel.value, precision_sel.value, device_sel.value)

            processed = 0
            for idx, img_path in enumerate(img_paths):
                progress(idx / total, desc=f"Captioning {os.path.basename(img_path)}")
                
                try:
                    txt_path = os.path.splitext(img_path)[0] + ".txt"
                    if not overwrite and os.path.exists(txt_path):
                        processed += 1
                        continue
                        
                    img = Image.open(img_path)
                    text_res, _, _ = engine.run_task(img, task)
                    
                    final_caption = text_res
                    if prefix:
                        final_caption = prefix + final_caption
                    if suffix:
                        final_caption = final_caption + suffix
                        
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(final_caption)
                        
                    processed += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")

            return f"Status: Successful! Processed {processed}/{total} images in dataset directory."

        btn_batch_run.click(
            batch_run_fn, 
            inputs=[batch_dir, batch_task, write_mode, prefix_text, suffix_text], 
            outputs=[batch_status]
        )

        return [(florence_tab, "Florence-2", "florence2_tab")]

scripts.script_callbacks.on_ui_tabs(on_ui_tabs)
