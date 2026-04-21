"""
OmniVoice TTS HTTP API.

POST /generate           -> submit a job, returns {job_id, status}
GET  /jobs/{job_id}      -> check status, returns {status, url?, error?}
GET  /files/{job_id}.wav -> download the generated audio (served statically)
"""

import asyncio
import hashlib
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
import soundfile as sf
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl, Field

from omnivoice import OmniVoice

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CACHE_DIR = Path(os.getenv("REF_CACHE_DIR", "cache/ref_audio"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
BASE_URL = os.getenv("BASE_URL", "https://tts.trid.id.vn").rstrip("/")
DEVICE = os.getenv("OMNIVOICE_DEVICE", "cuda:0")  # or "mps", "cpu"
PORT = int(os.getenv("PORT", "8000"))

# Cloudflare Tunnel
#   CF_TOKEN         -> run a named tunnel configured in Zero Trust dashboard
#   CF_TUNNEL_URL    -> public hostname for the named tunnel (used as BASE_URL)
#   (neither set)    -> run a quick tunnel, auto-detect *.trycloudflare.com URL
CF_TOKEN = os.getenv("CF_TOKEN")
CF_TUNNEL_URL = os.getenv("CF_TUNNEL_URL", "").rstrip("/") or None

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mutable runtime state — updated once cloudflared reports its public URL.
runtime = {"base_url": BASE_URL, "tunnel_proc": None}

# --------------------------------------------------------------------------- #
# Model (loaded once at startup)
# --------------------------------------------------------------------------- #
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=DEVICE,
    dtype=torch.float16,
)

# Serialize GPU access so overlapping requests don't OOM or interleave.
gpu_lock = asyncio.Lock()

# In-memory job store. Swap for Redis / DB if you need persistence.
jobs: dict[str, dict] = {}

# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class GenerateRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    ref_audio_url: HttpUrl = Field(..., description="URL to the reference audio (wav/mp3/...)")
    ref_text: Optional[str] = Field(
        None,
        description="Transcript of the reference audio. If omitted, Whisper ASR will transcribe it.",
    )
    speed: float = Field(1.0, gt=0.1, le=3.0, description="Speech speed, 1.0 = normal")


class JobResponse(BaseModel):
    job_id: str
    status: str  # queued | processing | completed | failed
    url: Optional[str] = None
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def download_ref_audio(url: str) -> Path:
    """Download the reference audio, caching by a hash of the URL."""
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    ext = Path(urlparse(url).path).suffix or ".wav"
    cache_path = CACHE_DIR / f"{url_hash}{ext}"

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)

    return cache_path


def _run_model(text: str, ref_audio: Path, ref_text: Optional[str], speed: float) -> "list":
    """Blocking model call. Runs under gpu_lock via asyncio.to_thread."""
    kwargs = {"text": text, "ref_audio": str(ref_audio), "speed": speed}
    if ref_text:
        kwargs["ref_text"] = ref_text
    return model.generate(**kwargs)


async def process_job(job_id: str, req: GenerateRequest) -> None:
    try:
        jobs[job_id]["status"] = "processing"

        ref_path = await download_ref_audio(str(req.ref_audio_url))

        async with gpu_lock:
            audio = await asyncio.to_thread(
                _run_model, req.text, ref_path, req.ref_text, req.speed
            )

        output_path = OUTPUT_DIR / f"{job_id}.wav"
        sf.write(str(output_path), audio[0], 24000)

        jobs[job_id].update(
            status="completed",
            url=f"{runtime['base_url']}/files/{job_id}.wav",
        )
    except Exception as e:
        jobs[job_id].update(status="failed", error=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="OmniVoice TTS API")

# Completed files are served statically at /files/<job_id>.wav
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")


@app.post("/generate", response_model=JobResponse)
async def generate(req: GenerateRequest, background_tasks: BackgroundTasks) -> JobResponse:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "url": None, "error": None}
    background_tasks.add_task(process_job, job_id, req)
    return JobResponse(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(
        job_id=job_id,
        status=job["status"],
        url=job.get("url"),
        error=job.get("error"),
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "device": DEVICE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)