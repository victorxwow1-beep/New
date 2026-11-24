# Zenvyra Studio

Independent FastAPI + OpenCV + FFmpeg video clipper. See original ZENVYRA-STUDIO-HOSTING.zip for full source archive, tests and documentation.

## Start locally
Install Python 3.13 and FFmpeg/FFprobe. Run `pip install -r requirements.txt`, then `python -m uvicorn app:app --host 127.0.0.1 --port 8000`. Open http://127.0.0.1:8000.

## Free hosting
Build using Dockerfile; set ZENVYRA_HOSTED=1, ZENVYRA_USERNAME=studio, ZENVYRA_PASSWORD to a random password of at least 16 characters in the hosting dashboard, and ZENVYRA_MAX_UPLOAD_BYTES=500000000. Do not commit secrets. Free-hosted storage is ephemeral, processing capacity limited, and 500 GB uploads are not supported. The application is private single-owner, not a multi-user service. YouTube links are references; supply authorized original media. Highlight detection uses measured motion, audio and scene changes, not a verified semantic multimodal model.
