import launch

if not launch.is_installed("transformers"):
    launch.run_pip("install transformers>=4.41.0", "requirements for Florence-2 (transformers)")

if not launch.is_installed("timm"):
    launch.run_pip("install timm", "requirements for Florence-2 (timm)")

if not launch.is_installed("einops"):
    launch.run_pip("install einops", "requirements for Florence-2 (einops)")

if not launch.is_installed("accelerate"):
    launch.run_pip("install accelerate", "requirements for Florence-2 (accelerate)")
