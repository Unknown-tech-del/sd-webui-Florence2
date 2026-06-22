import launch

# Ensure transformers is version 4.57.0 or higher for Qwen3-VL support
upgrade_transformers = False
try:
    import importlib.metadata
    from packaging.version import parse as parse_version
    current_version = importlib.metadata.version("transformers")
    if parse_version(current_version) < parse_version("4.57.0"):
        upgrade_transformers = True
except Exception:
    try:
        import transformers
        v_parts = [int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit()]
        if len(v_parts) >= 2 and (v_parts[0] < 4 or (v_parts[0] == 4 and v_parts[1] < 57)):
            upgrade_transformers = True
    except Exception:
        upgrade_transformers = True

if upgrade_transformers:
    launch.run_pip("install transformers>=4.57.0", "upgrading transformers to support Qwen3-VL")

if not launch.is_installed("timm"):
    launch.run_pip("install timm", "requirements for Florence-2 (timm)")

if not launch.is_installed("einops"):
    launch.run_pip("install einops", "requirements for Florence-2 (einops)")

if not launch.is_installed("accelerate"):
    launch.run_pip("install accelerate", "requirements for Florence-2 (accelerate)")

if not launch.is_installed("qwen_vl_utils"):
    launch.run_pip("install qwen-vl-utils", "requirements for Qwen-VL (qwen-vl-utils)")

