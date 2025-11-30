from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sqlite3, os, uuid, sys
from datetime import datetime
from pathlib import Path

# Ensure the current directory is in Python path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pdf_service import extract_pdf_with_ocr
from clustering_service import process_clustering_job
from pca_service import process_pca_job
from plotting_service import process_plotting_job, TERNARY_MAP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "uploads.db")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI(title="ArQave")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=os.path.join(BASE_DIR, "assets")), name="assets")
app.mount("/pages-css", StaticFiles(directory=os.path.join(BASE_DIR, "pages-css")), name="pages-css")
app.mount("/pages-js", StaticFiles(directory=os.path.join(BASE_DIR, "pages-js")), name="pages-js")
app.mount("/ternary", StaticFiles(directory=os.path.join(BASE_DIR, "ternary")), name="ternary")


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

init_db()


@app.get("/")
def root():
    return RedirectResponse(url="/index.html")


@app.get("/health")
def health():
    try:
        conn = db()
        conn.execute("SELECT 1")
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def update_status(job_id: str, status: str, excel_path: str | None = None, error: str | None = None, details: str | None = None):
    conn = db()
    c = conn.cursor()
    c.execute("UPDATE uploads SET status=?, excel_path=?, error_msg=?, extraction_details=? WHERE id=?",
              (status, excel_path, error, details, job_id))
    conn.commit()
    conn.close()


def worker(job_id: str, pdf_path: str):
    try:
        job_dir = Path(OUTPUTS_DIR) / job_id
        result = extract_pdf_with_ocr(
            job_dir=job_dir,
            pdf_path=Path(pdf_path),
            mode="excel",
            pages="all",
        )
        # Store extraction details as JSON
        import json
        details = json.dumps({
            "tables_extracted": result.get("tables", 0),
            "tables_dropped": result.get("tables_dropped", 0),
            "drop_reasons": result.get("drop_reasons", {})
        })
        update_status(job_id, "verified", excel_path=result["result_path"], details=details)
    except Exception as e:
        update_status(job_id, "rejected", error=str(e))


@app.post("/api/pdf/upload")
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        return JSONResponse({"error": "Only PDF allowed"}, status_code=400)
    job_id = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOADS_DIR, f"{job_id}_original.pdf")
    with open(pdf_path, "wb") as f:
        f.write(await file.read())
    conn = db()
    c = conn.cursor()
    c.execute("INSERT INTO uploads(id, filename, pdf_path, status, created_at) VALUES (?, ?, ?, ?, ?)",
              (job_id, file.filename, pdf_path, "pending", datetime.now().isoformat()))
    conn.commit()
    conn.close()
    if background_tasks:
        background_tasks.add_task(worker, job_id, pdf_path)
    return {"job_id": job_id, "status": "pending", "message": "Processing started"}


@app.get("/api/pdf/jobs/{job_id}/status")
async def status(job_id: str):
    conn = db()
    c = conn.cursor()
    c.execute("SELECT * FROM uploads WHERE id = ?", (job_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    
    # Parse extraction details if available
    import json
    details = None
    if row["extraction_details"]:
        try:
            details = json.loads(row["extraction_details"])
        except:
            pass
    
    return {
        "job_id": job_id, 
        "status": row["status"], 
        "filename": row["filename"], 
        "error": row["error_msg"],
        "extraction_details": details
    }


@app.get("/api/pdf/jobs/{job_id}/download")
async def download(job_id: str):
    conn = db()
    c = conn.cursor()
    c.execute("SELECT * FROM uploads WHERE id = ?", (job_id,))
    row = c.fetchone()
    conn.close()
    if not row or row["status"] != "verified":
        return JSONResponse({"error": "File not ready"}, status_code=400)
    xlsx_path = row["excel_path"]
    if not xlsx_path or not os.path.exists(xlsx_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(xlsx_path, filename=f"{job_id}.xlsx")


@app.get("/api/pdf/uploads")
async def recent():
    conn = db()
    c = conn.cursor()
    c.execute("SELECT * FROM uploads ORDER BY created_at DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    return {"uploads": [dict(r) for r in rows]}


# Clustering endpoints
@app.post("/api/clustering/run")
async def clustering_run(
    file: UploadFile = File(...),
    n_clusters: int = 3,
    algo: str = "KMeans"
):
    """Perform clustering on uploaded CSV/Excel file with algorithm selection."""
    allowed_extensions = {".csv", ".xlsx", ".xls"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return JSONResponse(
            {"error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"},
            status_code=400
        )
    
    if not (2 <= n_clusters <= 10):
        return JSONResponse({"error": "n_clusters must be between 2 and 10"}, status_code=400)
    
    if algo not in ["KMeans", "Agglomerative"]:
        return JSONResponse({"error": "algo must be 'KMeans' or 'Agglomerative'"}, status_code=400)
    
    try:
        job_id = str(uuid.uuid4())
        job_dir = Path(OUTPUTS_DIR) / "clustering" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        input_path = job_dir / f"input{file_ext}"
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        output_path = job_dir / "clustered_output.csv"
        metadata = process_clustering_job(input_path, output_path, n_clusters, algo)
        
        return {
            "job_id": job_id,
            "download": f"/api/clustering/download/{job_id}",
            "download_plot": f"/api/clustering/download/{job_id}/plot",
            "plot_base64": metadata["plot_base64"],
            "metadata": {
                "n_clusters": metadata["n_clusters"],
                "n_samples": metadata["n_samples"],
                "algorithm": metadata["algorithm"],
                "columns_used": metadata["columns_used"]
            }
        }
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Processing failed: {str(e)}"}, status_code=500)


@app.get("/api/clustering/download/{job_id}")
def clustering_download(job_id: str):
    """Download clustered CSV file."""
    output_path = Path(OUTPUTS_DIR) / "clustering" / job_id / "clustered_output.csv"
    if not output_path.exists():
        raise HTTPException(404, "Clustered file not found")
    return FileResponse(str(output_path), media_type="text/csv", filename=f"clustered_{job_id}.csv")


@app.get("/api/clustering/download/{job_id}/plot")
def clustering_download_plot(job_id: str):
    """Download cluster plot PNG file."""
    plot_path = Path(OUTPUTS_DIR) / "clustering" / job_id / "cluster_plot.png"
    if not plot_path.exists():
        raise HTTPException(404, "Cluster plot not found")
    return FileResponse(str(plot_path), media_type="image/png", filename=f"cluster_plot_{job_id}.png")


# PCA endpoints
@app.post("/api/pca/run")
async def pca_run(file: UploadFile = File(...), n_components: int = 2):
    """Perform PCA on uploaded CSV/Excel file."""
    allowed_extensions = {".csv", ".xlsx", ".xls"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return JSONResponse(
            {"error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"},
            status_code=400
        )
    
    if not (1 <= n_components <= 5):
        return JSONResponse({"error": "n_components must be between 1 and 5"}, status_code=400)
    
    try:
        job_id = str(uuid.uuid4())
        job_dir = Path(OUTPUTS_DIR) / "pca" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        input_path = job_dir / f"input{file_ext}"
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        metadata = process_pca_job(input_path, job_dir, n_components)
        
        return {
            "job_id": job_id,
            "download_scores": f"/api/pca/download/{job_id}/scores",
            "download_loadings": f"/api/pca/download/{job_id}/loadings",
            "download_plot": f"/api/pca/download/{job_id}/plot",
            "plot_base64": metadata["plot_base64"],
            "metadata": {
                "n_components": metadata["n_components"],
                "n_samples": metadata["n_samples"],
                "explained_variance": metadata["explained_variance"],
                "cumulative_variance": metadata["cumulative_variance"],
                "columns_used": metadata["columns_used"]
            }
        }
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Processing failed: {str(e)}"}, status_code=500)


@app.get("/api/pca/download/{job_id}/scores")
def pca_download_scores(job_id: str):
    """Download PCA scores CSV."""
    output_path = Path(OUTPUTS_DIR) / "pca" / job_id / "pca_scores.csv"
    if not output_path.exists():
        raise HTTPException(404, "PCA scores file not found")
    return FileResponse(str(output_path), media_type="text/csv", filename=f"pca_scores_{job_id}.csv")


@app.get("/api/pca/download/{job_id}/loadings")
def pca_download_loadings(job_id: str):
    """Download PCA loadings CSV."""
    output_path = Path(OUTPUTS_DIR) / "pca" / job_id / "pca_loadings.csv"
    if not output_path.exists():
        raise HTTPException(404, "PCA loadings file not found")
    return FileResponse(str(output_path), media_type="text/csv", filename=f"pca_loadings_{job_id}.csv")


@app.get("/api/pca/download/{job_id}/plot")
def pca_download_plot(job_id: str):
    """Download PCA plot PNG."""
    output_path = Path(OUTPUTS_DIR) / "pca" / job_id / "pca_plot.png"
    if not output_path.exists():
        raise HTTPException(404, "PCA plot file not found")
    return FileResponse(str(output_path), media_type="image/png", filename=f"pca_plot_{job_id}.png")

# TERNARY PLOTTING ENDPOINTS
@app.post("/api/plotting/run")
async def plotting_run(
    file: UploadFile = File(...),
    mode: str = "auto",
    system: str = None,
    include_k2o: bool = True
):
    """Perform ternary plotting on uploaded CSV/Excel file."""
    allowed_extensions = {".csv", ".xlsx", ".xls"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return JSONResponse(
            {"error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"},
            status_code=400
        )
    
    if mode not in ["auto", "manual"]:
        return JSONResponse({"error": "mode must be 'auto' or 'manual'"}, status_code=400)
    
    if mode == "manual" and (not system or system not in TERNARY_MAP):
        return JSONResponse(
            {"error": f"system must be one of: {', '.join(TERNARY_MAP.keys())}"},
            status_code=400
        )
    
    try:
        job_id = str(uuid.uuid4())
        job_dir = Path(OUTPUTS_DIR) / "plotting" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        input_path = job_dir / f"input{file_ext}"
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        metadata = process_plotting_job(
            input_path, 
            job_dir, 
            mode=mode,
            system=system,
            include_k2o=include_k2o
        )
        
        return {
            "job_id": job_id,
            "download_plot": f"/api/plotting/download/{job_id}/plot",
            "download_data": f"/api/plotting/download/{job_id}/data",
            "plot_base64": metadata["plot_base64"],
            "metadata": {
                "system": metadata["system"],
                "n_samples": metadata["n_samples"],
                "mode": metadata["mode"],
                "include_k2o": metadata["include_k2o"],
                "corner_names": metadata["corner_names"]
            }
        }
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Processing failed: {str(e)}"}, status_code=500)


@app.get("/api/plotting/download/{job_id}/plot")
def plotting_download_plot(job_id: str):
    """Download ternary plot PNG."""
    plot_path = Path(OUTPUTS_DIR) / "plotting" / job_id / "ternary_plot.png"
    if not plot_path.exists():
        raise HTTPException(404, "Ternary plot not found")
    return FileResponse(str(plot_path), media_type="image/png", filename=f"ternary_plot_{job_id}.png")


@app.get("/api/plotting/download/{job_id}/data")
def plotting_download_data(job_id: str):
    """Download normalized data CSV."""
    data_path = Path(OUTPUTS_DIR) / "plotting" / job_id / "normalized_data.csv"
    if not data_path.exists():
        raise HTTPException(404, "Normalized data not found")
    return FileResponse(str(data_path), media_type="text/csv", filename=f"normalized_data_{job_id}.csv")


@app.get("/api/plotting/systems")
def plotting_systems():
    """Return available ternary systems."""
    return {"systems": list(TERNARY_MAP.keys())}


app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static_root")
