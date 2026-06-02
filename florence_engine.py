import os
import gc
import torch
from PIL import Image, ImageDraw
from huggingface_hub import snapshot_download

# Inject mock implementations into transformers.utils to prevent import errors with older transformers versions
try:
    import transformers.utils
    for name in ["is_flash_attn_2_available", "is_flash_attn_available", "is_flash_attn_greater_or_equal_2_10", "is_flash_attn_greater_or_equal_2"]:
        setattr(transformers.utils, name, lambda: False)
except Exception:
    pass

try:
    import transformers.utils.import_utils
    for name in ["is_flash_attn_2_available", "is_flash_attn_available", "is_flash_attn_greater_or_equal_2_10", "is_flash_attn_greater_or_equal_2"]:
        setattr(transformers.utils.import_utils, name, lambda: False)
except Exception:
    pass

# Patch PretrainedConfig to provide a default forced_bos_token_id class attribute
try:
    import transformers.configuration_utils
    transformers.configuration_utils.PretrainedConfig.forced_bos_token_id = None
except Exception:
    pass

# Patch PreTrainedTokenizerBase to bridge transformers v4/v5 renaming and guarantee additional_special_tokens is always a list of strings
try:
    import transformers.tokenization_utils_base
    
    def _patched_additional_special_tokens(self):
        try:
            val = getattr(self, "extra_special_tokens", None)
            if val is None:
                val = getattr(self, "_additional_special_tokens", [])
        except Exception:
            val = []
        if val is None:
            val = []
        if isinstance(val, dict):
            return list(val.keys())
        elif isinstance(val, list) or isinstance(val, tuple):
            return [str(item) for item in val]
        else:
            return list(val)
            
    transformers.tokenization_utils_base.PreTrainedTokenizerBase.additional_special_tokens = property(_patched_additional_special_tokens)
    
    if not hasattr(transformers.tokenization_utils_base.PreTrainedTokenizerBase, "additional_special_tokens_ids"):
        transformers.tokenization_utils_base.PreTrainedTokenizerBase.additional_special_tokens_ids = property(
            lambda self: getattr(self, "extra_special_tokens_ids", [])
        )
except Exception:
    pass


# Patch PreTrainedModel and torch.nn.Module to include standard v5 attributes for v4 backward compatibility
try:
    import torch.nn
    import transformers.modeling_utils
    
    transformers.modeling_utils.PreTrainedModel._supports_sdpa = True
    transformers.modeling_utils.PreTrainedModel._supports_flash_attn_2 = False
    
    torch.nn.Module._supports_sdpa = True
    torch.nn.Module._supports_flash_attn_2 = False
    
    # Intercept __getattr__ at PreTrainedModel level to bypass PyTorch's C++ nn.Module.__getattr__ lookup limitations
    _original_PreTrainedModel_getattr = getattr(transformers.modeling_utils.PreTrainedModel, "__getattr__", None)
    def _patched_PreTrainedModel_getattr(self, name):
        if name == "_supports_sdpa":
            return True
        if name == "_supports_flash_attn_2":
            return False
        if _original_PreTrainedModel_getattr is not None:
            try:
                return _original_PreTrainedModel_getattr(self, name)
            except AttributeError:
                pass
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    transformers.modeling_utils.PreTrainedModel.__getattr__ = _patched_PreTrainedModel_getattr
except Exception:
    pass







# Windows Compatibility Monkeypatches for Unix-only signal.SIGALRM issues in transformers
import signal
import sys

# Patch signal if SIGALRM doesn't exist (Windows platform)
if not hasattr(signal, "SIGALRM"):
    signal.SIGALRM = 14
    _original_signal = signal.signal
    def _patched_signal(sig, handler):
        try:
            return _original_signal(sig, handler)
        except ValueError:
            # Silently ignore invalid signal registration on Windows
            return None
    signal.signal = _patched_signal

# Patch transformers dynamic module resolver to bypass user authorization prompts and check_imports errors
try:
    import transformers.dynamic_module_utils
    transformers.dynamic_module_utils.resolve_trust_remote_code = lambda *args, **kwargs: True
    
    _original_check_imports = transformers.dynamic_module_utils.check_imports
    def _patched_check_imports(resolved_module_file):
        try:
            return _original_check_imports(resolved_module_file)
        except ImportError as e:
            if 'flash_attn' in str(e):
                print("[Florence-2 Patch] Ignoring flash_attn import check.")
                return []
            raise e
    transformers.dynamic_module_utils.check_imports = _patched_check_imports

    # Intercept get_class_from_dynamic_module to dynamically patch Florence-2 classes loaded from remote code
    _original_get_class = transformers.dynamic_module_utils.get_class_from_dynamic_module
    def _patched_get_class(*args, **kwargs):
        cls = _original_get_class(*args, **kwargs)
        if cls is not None:
            name = getattr(cls, "__name__", None)
            if name == "Florence2ForConditionalGeneration":
                print(f"[Florence-2 Patch] Intercepted Florence2ForConditionalGeneration! Resolving module classes...")
                import sys
                try:
                    module = sys.modules[cls.__module__]
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type):
                            attr_class_name = getattr(attr, "__name__", None)
                            if attr_class_name in ["Florence2ForConditionalGeneration", "Florence2LanguageForConditionalGeneration", "Florence2LanguageModel"]:
                                print(f"[Florence-2 Patch] Patching dynamic module class: {attr_class_name}")
                                
                                # 1. Dynamically inject GenerationMixin into base classes to support .generate() under transformers 4.50+/5.x
                                try:
                                    import transformers.generation
                                    gen_mixin = transformers.generation.GenerationMixin
                                    if gen_mixin not in attr.__bases__:
                                        attr.__bases__ = attr.__bases__ + (gen_mixin,)
                                        print(f"[Florence-2 Patch] Dynamically added GenerationMixin to {attr_class_name} base classes.")
                                except Exception as e:
                                    print(f"[Florence-2 Patch] Failed to add GenerationMixin to {attr_class_name}: {str(e)}")
                                
                                # 2. Patch prepare_inputs_for_generation to prevent past_key_values shape AttributeError crashes
                                _original_prepare = getattr(attr, "prepare_inputs_for_generation", None)
                                if _original_prepare is not None:
                                    def _patched_prepare(self, input_ids, past_key_values=None, **pk_kwargs):
                                        if past_key_values is not None:
                                            try:
                                                if len(past_key_values) > 0 and (past_key_values[0] is None or past_key_values[0][0] is None):
                                                    past_key_values = None
                                            except Exception:
                                                past_key_values = None
                                        return _original_prepare(self, input_ids, past_key_values=past_key_values, **pk_kwargs)
                                    attr.prepare_inputs_for_generation = _patched_prepare
                                    print(f"[Florence-2 Patch] Dynamically patched prepare_inputs_for_generation for {attr_class_name}.")
                except Exception as e:
                    print(f"[Florence-2 Patch] Module introspection failed: {str(e)}")
        return cls
    transformers.dynamic_module_utils.get_class_from_dynamic_module = _patched_get_class
except Exception:
    pass




# Patch json.JSONEncoder to prevent serialization errors for custom configuration objects
import json
_original_default = json.JSONEncoder.default
def _patched_default(self, o):
    try:
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if hasattr(o, "__dict__"):
            return {k: v for k, v in o.__dict__.items() if not k.startswith('_')}
    except Exception:
        pass
    try:
        return _original_default(self, o)
    except TypeError:
        return str(o)
json.JSONEncoder.default = _patched_default

from transformers import AutoProcessor, AutoModelForCausalLM



# Setup base models directory under SD WebUI models folder
try:
    from modules import paths
    models_dir = os.path.join(paths.models_path, "Florence2")
except ImportError:
    # Fallback for standalone/testing environments
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Florence2")

class FlorenceEngine:
    def __init__(self):
        self.model = None
        self.processor = None
        self.loaded_model_name = None
        self.loaded_precision = None
        self.loaded_device = None

    def get_models_dir(self):
        os.makedirs(models_dir, exist_ok=True)
        return models_dir

    def load_model(self, model_name, precision="fp16", device="auto"):
        # Select device
        if device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target_device = device

        # Check precision
        dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }.get(precision, torch.float16)

        # Basename of the model
        model_folder_name = model_name.split("/")[-1]
        model_local_path = os.path.join(self.get_models_dir(), model_folder_name)

        # Download if not present locally
        if not os.path.exists(model_local_path) or not os.listdir(model_local_path):
            print(f"[Florence-2] Downloading model {model_name} to local path {model_local_path}...")
            snapshot_download(
                repo_id=model_name,
                local_dir=model_local_path,
                local_dir_use_symlinks=False
            )

        # Unload existing model if any is loaded
        if self.loaded_model_name != model_name or self.loaded_precision != precision or self.loaded_device != target_device:
            self.unload_model()

            print(f"[Florence-2] Loading model {model_name} on {target_device} in {precision} precision...")
            self.processor = AutoProcessor.from_pretrained(model_local_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_local_path,
                torch_dtype=dtype,
                trust_remote_code=True
            ).to(target_device)
            
            # Unconditionally patch the model classes in memory directly on the loaded objects
            # This completely bypasses any import caching, compilation caching, or re-loads
            try:
                for obj in [self.model, getattr(self.model, "language_model", None)]:
                    if obj is not None:
                        cls = obj.__class__
                        name = getattr(cls, "__name__", None)
                        print(f"[Florence-2 Patch] Active patching on loaded class: {name}")
                        
                        # 1. Inject GenerationMixin
                        import transformers.generation
                        gen_mixin = transformers.generation.GenerationMixin
                        if gen_mixin not in cls.__bases__:
                            cls.__bases__ = cls.__bases__ + (gen_mixin,)
                            print(f"[Florence-2 Patch] Dynamically added GenerationMixin to {name}.")
                            
                        # 2. Patch prepare_inputs_for_generation
                        _original_prepare = getattr(cls, "prepare_inputs_for_generation", None)
                        if _original_prepare is not None:
                            if not getattr(_original_prepare, "__is_patched__", False):
                                def _patched_prepare(self_obj, input_ids, past_key_values=None, **pk_kwargs):
                                    if past_key_values is not None:
                                        try:
                                            if len(past_key_values) > 0 and (past_key_values[0] is None or past_key_values[0][0] is None):
                                                past_key_values = None
                                        except Exception:
                                            past_key_values = None
                                    return _original_prepare(self_obj, input_ids, past_key_values=past_key_values, **pk_kwargs)
                                _patched_prepare.__is_patched__ = True
                                cls.prepare_inputs_for_generation = _patched_prepare
                                print(f"[Florence-2 Patch] Dynamically patched prepare_inputs_for_generation for {name}.")
                                
                        # 3. Ensure generation_config is populated (prevents NoneType _from_model_config AttributeError under transformers 4.50+/5.x)
                        if getattr(obj, "generation_config", None) is None:
                            try:
                                from transformers import GenerationConfig
                                obj.generation_config = GenerationConfig.from_model_config(obj.config)
                                print(f"[Florence-2 Patch] Dynamically initialized generation_config for {name}.")
                            except Exception as e:
                                print(f"[Florence-2 Patch] Failed to initialize generation_config for {name}: {str(e)}")
            except Exception as e:
                print(f"[Florence-2 Patch] Memory object patching failed: {str(e)}")


            self.model.eval()


            self.loaded_model_name = model_name
            self.loaded_precision = precision
            self.loaded_device = target_device
            print("[Florence-2] Model loaded successfully.")
        else:
            print("[Florence-2] Model already loaded with the same settings.")

        return f"Loaded: {self.loaded_model_name} ({self.loaded_precision}) on {self.loaded_device}"

    def unload_model(self):
        if self.model is not None:
            print("[Florence-2] Unloading model from memory...")
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        self.loaded_model_name = None
        self.loaded_precision = None
        self.loaded_device = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "Model unloaded."

    def run_task(self, image, task, prompt_text=""):
        if self.model is None or self.processor is None:
            raise RuntimeError("Florence-2 model is not loaded. Please load a model first.")

        # Ensure image is in RGB
        image_rgb = image.convert("RGB")
        w, h = image_rgb.size

        # Define Florence-2 prompt mapping
        task_prompts = {
            "Caption": "<CAPTION>",
            "Detailed Caption": "<DETAILED_CAPTION>",
            "More Detailed Caption": "<MORE_DETAILED_CAPTION>",
            "Object Detection": "<OD>",
            "Dense Region Caption": "<DENSE_REGION_CAPTION>",
            "Region Proposal": "<REGION_PROPOSAL>",
            "Caption to Phrase Grounding": f"<CAPTION_TO_PHRASE_GROUNDING> {prompt_text}",
            "Referring Expression Segmentation": f"<REFERRING_EXPRESSION_SEGMENTATION> {prompt_text}",
            "OCR": "<OCR>",
            "OCR with Region": "<OCR_WITH_REGION>"
        }

        task_prompt = task_prompts.get(task, "<CAPTION>")

        print(f"[Florence-2] Running task '{task}' with prompt: '{task_prompt}'")

        # Prepare inputs
        inputs = self.processor(text=task_prompt, images=image_rgb, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        pixel_values = inputs["pixel_values"].to(self.model.device, dtype=self.model.dtype)

        # Run inference with use_cache=False to completely bypass KV caching AttributeError compatibility issues on Windows
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
                use_cache=False
            )


        # Decode output
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Post process to parse answer into structural structures
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(w, h)
        )

        print(f"[Florence-2] Raw parsed answer: {parsed_answer}")

        # Extract parsed elements for visualization and mask generation
        bboxes = []
        labels = []
        polygons = []
        polygons_labels = []

        for k, v in parsed_answer.items():
            if isinstance(v, dict):
                if 'bboxes' in v:
                    bboxes.extend(v['bboxes'])
                    labels.extend(v.get('labels', [''] * len(v['bboxes'])))
                if 'polygons' in v:
                    polygons.extend(v['polygons'])
                    polygons_labels.extend(v.get('labels', [''] * len(v['polygons'])))

        # 1. Generate text result description
        text_results = []
        for k, v in parsed_answer.items():
            if isinstance(v, str):
                text_results.append(v)
            elif isinstance(v, dict):
                if 'caption' in v:
                    text_results.append(v['caption'])
                elif 'ocr_text' in v:
                    text_results.append(v['ocr_text'])
                elif 'labels' in v and not 'bboxes' in v and not 'polygons' in v:
                    text_results.append(", ".join(v['labels']))
                else:
                    lines = []
                    if 'bboxes' in v:
                        lbls = v.get('labels', [''] * len(v['bboxes']))
                        for b, l in zip(v['bboxes'], lbls):
                            lines.append(f"{l}: {b}")
                    if 'polygons' in v:
                        lbls = v.get('labels', [''] * len(v['polygons']))
                        for p, l in zip(v['polygons'], lbls):
                            lines.append(f"{l}: {p}")
                    if lines:
                        text_results.append("\n".join(lines))
                    else:
                        text_results.append(str(v))
            else:
                text_results.append(str(v))

        text_result = "\n\n".join(text_results)

        # 2. Visualization Drawing
        visualized_image = image_rgb.copy()
        
        # Draw bounding boxes
        if bboxes:
            draw = ImageDraw.Draw(visualized_image)
            for bbox, label in zip(bboxes, labels):
                xmin, ymin, xmax, ymax = bbox
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
                if label:
                    draw.text((xmin + 4, ymin + 4), label, fill="red")

        # Draw polygons
        if polygons:
            for poly, label in zip(polygons, polygons_labels):
                if len(poly) > 0 and isinstance(poly[0], list):
                    sub_polys = poly
                else:
                    sub_polys = [poly]
                for sub_poly in sub_polys:
                    if len(sub_poly) >= 4:
                        points = [(sub_poly[i], sub_poly[i+1]) for i in range(0, len(sub_poly), 2)]
                        overlay = Image.new("RGBA", visualized_image.size, (0, 0, 0, 0))
                        overlay_draw = ImageDraw.Draw(overlay)
                        overlay_draw.polygon(points, fill=(255, 0, 0, 80), outline=(255, 0, 0, 255))
                        visualized_image = Image.alpha_composite(visualized_image.convert("RGBA"), overlay)
                        visualized_image = visualized_image.convert("RGB")

        # 3. Generate Black-and-White Inpaint Mask
        mask_image = None
        if bboxes or polygons:
            mask_image = Image.new("L", image_rgb.size, 0)
            mask_draw = ImageDraw.Draw(mask_image)
            
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = bbox
                mask_draw.rectangle([xmin, ymin, xmax, ymax], fill=255)
                
            for poly in polygons:
                if len(poly) > 0 and isinstance(poly[0], list):
                    sub_polys = poly
                else:
                    sub_polys = [poly]
                for sub_poly in sub_polys:
                    if len(sub_poly) >= 4:
                        points = [(sub_poly[i], sub_poly[i+1]) for i in range(0, len(sub_poly), 2)]
                        mask_draw.polygon(points, fill=255)

        return text_result, visualized_image, mask_image
