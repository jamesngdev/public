#!/usr/bin/env bash
set -euo pipefail

# cd to /tmp
cd /tmp

# 1. Install PyTorch (CUDA 12.8) + OmniVoice + API deps
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128
pip install omnivoice fastapi uvicorn httpx soundfile pydantic

# 2. Install cloudflared
curl -L --output cloudflared.deb \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb

# 3. Download main.py
curl -L -o main.py \
    https://raw.githubusercontent.com/jamesngdev/public/refs/heads/main/ominivoice-install/main.py

# 4. Start a Cloudflare quick tunnel in the background, pointed at port 8000
#    Logs go to cloudflared.log so you can grep the trycloudflare.com URL.
nohup cloudflared tunnel --url http://localhost:8000 --no-autoupdate \
    > cloudflared.log 2>&1 &
CF_PID=$!
echo "cloudflared started (pid=$CF_PID)"

# Wait for the public URL to appear, then export it as BASE_URL for main.py
echo "Waiting for tunnel URL..."
for i in {1..30}; do
    TUNNEL_URL=$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' cloudflared.log | head -n1 || true)
    if [[ -n "${TUNNEL_URL:-}" ]]; then
        break
    fi
    sleep 1
done

if [[ -n "${TUNNEL_URL:-}" ]]; then
    echo "Tunnel ready: $TUNNEL_URL"
    export BASE_URL="$TUNNEL_URL"
else
    echo "Warning: could not detect tunnel URL, falling back to default BASE_URL"
fi

# 5. Run main.py
python main.py