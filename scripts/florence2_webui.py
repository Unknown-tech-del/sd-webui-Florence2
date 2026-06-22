import os
import gradio as gr
from PIL import Image
from modules import scripts
from florence_engine import FlorenceEngine

# List of models supported by default
florence_model_list = [
    'microsoft/Florence-2-base',
    'microsoft/Florence-2-base-ft',
    'microsoft/Florence-2-large',
    'microsoft/Florence-2-large-ft',
    'HuggingFaceM4/Florence-2-DocVQA',
    'MiaoshouAI/Florence-2-base-PromptGen-v1.5',
    'MiaoshouAI/Florence-2-large-PromptGen-v1.5',
    'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
    'MiaoshouAI/Florence-2-large-PromptGen-v2.0',
    'thwri/CogFlorence-2.2-Large',
    'gokaygokay/Florence-2-SD3-Captioner',
    'gokaygokay/Florence-2-Flux-Large',
    'NikshepShetty/Florence-2-pixelpros'
]

qwen_model_list = [
    'Qwen/Qwen3-VL-2B-Instruct',
    'Qwen/Qwen3-VL-2B-Thinking',
    'Qwen/Qwen2.5-VL-3B-Instruct',
    'Qwen/Qwen2.5-VL-7B-Instruct',
    'Qwen/Qwen2-VL-2B-Instruct',
    'Qwen/Qwen2-VL-7B-Instruct'
]

engine = FlorenceEngine()

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as vision_tab:
        gr.Markdown("# Multimodal Vision Companion")
        gr.Markdown("Explore visual intelligence locally. Switch between Microsoft's structured **Florence-2** model and Alibaba's conversational **Qwen-VL** model.")
        
        with gr.Tabs():
            # ==================== FLORENCE-2 SUB-TAB ====================
            with gr.Tab("Florence-2"):
                with gr.Row():
                    # Global Loader Settings Panel (Left column)
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("### 🛠️ Model Loader Settings")
                        model_sel = gr.Dropdown(choices=florence_model_list, value='microsoft/Florence-2-large-ft', label="Model Variant")
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

                # Bind loader functions for Florence-2
                def load_fn_florence(name, prec, dev):
                    try:
                        res = engine.load_model(name, prec, dev)
                        return f"**Status:** {res}"
                    except Exception as e:
                        return f"**Status Error:** {str(e)}"
                        
                def unload_fn_florence():
                    res = engine.unload_model()
                    return f"**Status:** {res}"

                btn_load.click(load_fn_florence, inputs=[model_sel, precision_sel, device_sel], outputs=[status_text])
                btn_unload.click(unload_fn_florence, outputs=[status_text])

                # Bind run task function for Florence-2
                def run_fn_florence(img, task, p_text):
                    if img is None:
                        return "Error: No image uploaded.", None, None
                    try:
                        if engine.model is None or engine.loaded_model_name != model_sel.value:
                            engine.load_model(model_sel.value, precision_sel.value, device_sel.value)
                            
                        text_res, vis_img, mask_img = engine.run_task(img, task, p_text)
                        return text_res, vis_img, mask_img
                    except Exception as e:
                        return f"Execution Error: {str(e)}", None, None

                btn_run.click(run_fn_florence, inputs=[input_img, task_sel, prompt_text], outputs=[output_text, visualized_img, output_mask])

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

                # Batch runner for Florence-2
                def batch_run_fn_florence(directory, task, overwrite, prefix, suffix, progress=gr.Progress()):
                    if not directory or not os.path.exists(directory):
                        return "Status: Error - Invalid or non-existent folder directory."
                        
                    valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
                    img_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(valid_exts)]
                    
                    if not img_paths:
                        return "Status: Error - No images found in dataset directory."

                    total = len(img_paths)
                    print(f"[Florence-2] Starting batch processing for {total} images...")
                    
                    if engine.model is None or engine.loaded_model_name != model_sel.value:
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
                    batch_run_fn_florence, 
                    inputs=[batch_dir, batch_task, write_mode, prefix_text, suffix_text], 
                    outputs=[batch_status]
                )

            # ==================== QWEN-VL SUB-TAB ====================
            with gr.Tab("Qwen-VL"):
                with gr.Row():
                    # Global Loader Settings Panel (Left column)
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("### 🛠️ Model Loader Settings")
                        q_model_sel = gr.Dropdown(choices=qwen_model_list, value='Qwen/Qwen3-VL-2B-Instruct', label="Model Variant")
                        q_precision_sel = gr.Dropdown(choices=['fp16', 'bf16', 'fp32'], value='bf16', label="Precision")
                        q_device_sel = gr.Dropdown(choices=['auto', 'cuda', 'cpu'], value='auto', label="Execution Device")
                        
                        gr.Markdown("### ⚙️ Generation Parameters")
                        q_max_tokens = gr.Slider(minimum=1, maximum=4096, value=1024, step=1, label="Max New Tokens")
                        q_temp = gr.Slider(minimum=0.0, maximum=1.5, value=0.7, step=0.05, label="Temperature")
                        q_top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top P")
                        q_do_sample = gr.Checkbox(label="Enable Sampling", value=False)
                        
                        with gr.Row():
                            q_btn_load = gr.Button("🔌 Load Model", variant="primary")
                            q_btn_unload = gr.Button("⚠️ Unload Model", variant="stop")
                            
                        q_status_text = gr.Markdown("**Status:** Model not loaded")
                        
                    # Main Processing Panel (Right column)
                    with gr.Column(scale=3):
                        with gr.Tabs():
                            with gr.Tab("Single Image"):
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        q_input_img = gr.Image(type="pil", label="Input Image")
                                        q_task_sel = gr.Dropdown(
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
                                                "OCR with Region",
                                                "Custom Prompt"
                                            ],
                                            value="Detailed Caption",
                                            label="Task Preset / Custom Mode"
                                        )
                                        q_prompt_text = gr.Textbox(
                                            label="Prompt / Phrase Override",
                                            placeholder="Specify phrase for grounding/segmentation OR write your custom question...",
                                            visible=True
                                        )
                                        q_btn_run = gr.Button("🚀 Run Vision Task", variant="primary")
                                        
                                    with gr.Column(scale=1):
                                        q_output_text = gr.Textbox(label="Result Description", interactive=False, show_copy_button=True)
                                        q_visualized_img = gr.Image(type="pil", label="Visualized Output")
                                        q_output_mask = gr.Image(type="pil", label="Generated Inpaint Mask")
                                        
                                        with gr.Row():
                                            q_btn_send_txt = gr.Button("📋 Send Prompt to txt2img", variant="secondary")
                                            q_btn_send_img = gr.Button("📋 Send Prompt to img2img", variant="secondary")
                                            
                            with gr.Tab("Batch Dataset Processing"):
                                gr.Markdown("### 📂 Bulk Image Dataset Preprocessing (LoRA Training Preparation)")
                                gr.Markdown("Generate high-quality matching descriptive captions/tags next to all images in a target directory.")
                                
                                q_batch_dir = gr.Textbox(label="Dataset Directory Path", placeholder="e.g. C:\\dataset\\my_lora_images")
                                
                                with gr.Row():
                                    q_batch_task = gr.Dropdown(
                                        choices=["Caption", "Detailed Caption", "More Detailed Caption", "Custom Prompt"],
                                        value="Detailed Caption",
                                        label="Caption Task Preset"
                                    )
                                    q_write_mode = gr.Checkbox(label="Overwrite Existing TXT Files", value=True)
                                    
                                with gr.Row():
                                    q_batch_prompt = gr.Textbox(label="Batch Custom Prompt Override (Used only when Custom Prompt task selected)", placeholder="e.g. Describe the main subject of this image.")
                                    
                                with gr.Row():
                                    q_prefix_text = gr.Textbox(label="Prepend Tag/Trigger Word", placeholder="e.g. '1girl, masterpiece, '")
                                    q_suffix_text = gr.Textbox(label="Append Tag/Trigger Word", placeholder="e.g. ', detailed background'")
                                    
                                q_btn_batch_run = gr.Button("⚡ Start Batch Captioning", variant="primary")
                                q_batch_status = gr.HTML(value="Status: Idle", label="Batch Progress Log")

                # Bind loader functions for Qwen-VL
                def load_fn_qwen(name, prec, dev):
                    try:
                        res = engine.load_model(name, prec, dev)
                        return f"**Status:** {res}"
                    except Exception as e:
                        return f"**Status Error:** {str(e)}"
                        
                def unload_fn_qwen():
                    res = engine.unload_model()
                    return f"**Status:** {res}"

                q_btn_load.click(load_fn_qwen, inputs=[q_model_sel, q_precision_sel, q_device_sel], outputs=[q_status_text])
                q_btn_unload.click(unload_fn_qwen, outputs=[q_status_text])

                # Bind run task function for Qwen-VL
                def run_fn_qwen(img, task, p_text, max_tok, temp, top_p_val, sample):
                    if img is None:
                        return "Error: No image uploaded.", None, None
                    try:
                        if engine.model is None or engine.loaded_model_name != q_model_sel.value:
                            engine.load_model(q_model_sel.value, q_precision_sel.value, q_device_sel.value)
                            
                        # If custom prompt mode, use it as the task
                        p_input = p_text
                        if task == "Custom Prompt":
                            actual_task = p_text
                        else:
                            actual_task = task
                            
                        text_res, vis_img, mask_img = engine.run_task(
                            img, actual_task, p_input,
                            max_tokens=int(max_tok),
                            temperature=float(temp),
                            top_p=float(top_p_val),
                            do_sample=sample
                        )
                        return text_res, vis_img, mask_img
                    except Exception as e:
                        return f"Execution Error: {str(e)}", None, None

                q_btn_run.click(
                    run_fn_qwen, 
                    inputs=[q_input_img, q_task_sel, q_prompt_text, q_max_tokens, q_temp, q_top_p, q_do_sample], 
                    outputs=[q_output_text, q_visualized_img, q_output_mask]
                )

                # Bind javascript functions to send text
                q_btn_send_txt.click(
                    fn=None,
                    inputs=[q_output_text],
                    outputs=[],
                    _js="(val) => { if (typeof send_florence_prompt === 'function') { send_florence_prompt(val, 'txt2img'); } else { alert('Helper JS not loaded yet'); } }"
                )
                
                q_btn_send_img.click(
                    fn=None,
                    inputs=[q_output_text],
                    outputs=[],
                    _js="(val) => { if (typeof send_florence_prompt === 'function') { send_florence_prompt(val, 'img2img'); } else { alert('Helper JS not loaded yet'); } }"
                )

                # Batch runner for Qwen-VL
                def batch_run_fn_qwen(directory, task, custom_p, overwrite, prefix, suffix, max_tok, temp, top_p_val, sample, progress=gr.Progress()):
                    if not directory or not os.path.exists(directory):
                        return "Status: Error - Invalid or non-existent folder directory."
                        
                    valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
                    img_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(valid_exts)]
                    
                    if not img_paths:
                        return "Status: Error - No images found in dataset directory."

                    total = len(img_paths)
                    print(f"[Qwen-VL] Starting batch processing for {total} images...")
                    
                    if engine.model is None or engine.loaded_model_name != q_model_sel.value:
                        engine.load_model(q_model_sel.value, q_precision_sel.value, q_device_sel.value)

                    processed = 0
                    for idx, img_path in enumerate(img_paths):
                        progress(idx / total, desc=f"Captioning {os.path.basename(img_path)}")
                        
                        try:
                            txt_path = os.path.splitext(img_path)[0] + ".txt"
                            if not overwrite and os.path.exists(txt_path):
                                processed += 1
                                continue
                                
                            img = Image.open(img_path)
                            
                            p_input = custom_p
                            if task == "Custom Prompt":
                                actual_task = custom_p
                            else:
                                actual_task = task

                            text_res, _, _ = engine.run_task(
                                img, actual_task, p_input,
                                max_tokens=int(max_tok),
                                temperature=float(temp),
                                top_p=float(top_p_val),
                                do_sample=sample
                            )
                            
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

                q_btn_batch_run.click(
                    batch_run_fn_qwen, 
                    inputs=[q_batch_dir, q_batch_task, q_batch_prompt, q_write_mode, q_prefix_text, q_suffix_text, q_max_tokens, q_temp, q_top_p, q_do_sample], 
                    outputs=[q_batch_status]
                )

        return [(vision_tab, "Vision", "vision_tab")]

scripts.script_callbacks.on_ui_tabs(on_ui_tabs)
