import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from contextlib import asynccontextmanager
import asyncio
from datetime import datetime
import subprocess
from pathlib import Path

import pandas as pd
import pytz
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from config import UPLOAD_DIR
from predict import predict
from evaluate import evaluate_result_df


STD_DIR = Path(__file__).resolve().parent.parent
PUBLIC_DIR = STD_DIR / "public"
APP_ROOT_PATH = os.getenv("APP_ROOT_PATH", "").rstrip("/")
timezone = pytz.timezone("Asia/Seoul")

RMSE_THRESHOLD = 450.0
RETRAIN_SCRIPT = Path(__file__).resolve().parent.parent / "retrain.py"

router = APIRouter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    for d in (PUBLIC_DIR, UPLOAD_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    lifespan=lifespan,
    root_path=APP_ROOT_PATH,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    ico = PUBLIC_DIR / "favicon.ico"
    if ico.exists():
        return FileResponse(str(ico), media_type="image/x-icon")
    png = PUBLIC_DIR / "favicon.png"
    if png.exists():
        return FileResponse(str(png), media_type="image/png")
    return Response(status_code=204)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    return response


async def _read_csv_async(file_path: Path) -> pd.DataFrame:
    def _read():
        return pd.read_csv(file_path)
    return await asyncio.to_thread(_read)


def _run_retrain() -> dict:
    result = subprocess.run(
        ["python", str(RETRAIN_SCRIPT)],
        capture_output=True,
        text=True,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.get("/health")
def health():
    return {"status": "ok", "root_path": APP_ROOT_PATH or "/"}


@app.get("/")
def root():
    index_html = PUBLIC_DIR / "index.html"
    if not index_html.exists():
        return {"message": "public/index.html not found. Place frontend files under /public."}

    html = index_html.read_text(encoding="utf-8", errors="ignore")
    rp = APP_ROOT_PATH or "/"

    if "<base" not in html.lower():
        html = html.replace("<head>", f'<head><base href="{rp if rp.endswith("/") else rp + "/"}">', 1)

    return HTMLResponse(content=html)


@router.post("/upload")
async def post_data_set(file: UploadFile = File(...)):
    try:
        current_time = datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
        new_filename = f"{current_time}_{file.filename}"
        file_location = Path(UPLOAD_DIR) / new_filename

        contents = await file.read()
        await asyncio.to_thread(file_location.write_bytes, contents)

        dataset = await _read_csv_async(file_location)

        building_info_df = pd.read_csv(Path(__file__).resolve().parent.parent / "dataset" / "building_info.csv")

        pred_result = await asyncio.to_thread(
            predict,
            dataset,
            building_info_df,
        )

        eval_result = await asyncio.to_thread(
            evaluate_result_df,
            pred_result["result_df"],
            "actual",
            "predicted",
            RMSE_THRESHOLD,
        )

        retrain_result = None
        if eval_result["retrain_required"]:
            retrain_result = await asyncio.to_thread(_run_retrain)

        return {
            "saved_filename": new_filename,
            "rmse": eval_result["rmse"],
            "mae": eval_result["mae"],
            "mape": eval_result["mape"],
            "rmse_threshold": eval_result["rmse_threshold"],
            "retrain_required": eval_result["retrain_required"],
            "message": eval_result["message"],
            "num_prediction_rows": len(pred_result["result_df"]),
            "preview": pred_result["result_df"].head(20).to_dict(orient="records"),
            "retrain_result": retrain_result,
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router)