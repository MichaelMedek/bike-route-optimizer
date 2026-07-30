"""Bike-route-optimizer package: DACH flat-preferring, surface-aware bike+rail routing."""

import os

# Disable hf-xet BEFORE huggingface_hub is imported by any submodule (the var is read at import
# time). Xet fetches a per-file xet-read-token; on Streamlit Cloud's anonymous, low-rate-limit
# requests that 429s mid-snapshot. The standard HTTP/CDN path has no such token, so it's robust.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
