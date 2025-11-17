from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sqlite3, os, uuid
from datetime import datetime
from pathlib import Path
from pdf_service import extract_pdf_to_outputs 

#Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "uploads.db")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

#App 
app = FastAPI(title="ArchaeoDB (Simple)")

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


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db(); c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS uploads(
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            pdf_path TEXT NOT NULL,
            excel_path TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT,
            error_msg TEXT
        )
    """)
    conn.commit(); conn.close()

init_db()

#Routes 
@app.get("/")
def root():
    return RedirectResponse(url="/index.html")

@app.get("/health")
def health():
    try:
        conn = db(); conn.execute("SELECT 1"); conn.close()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def update_status(job_id: str, status: str, excel_path: str | None = None, error: str | None = None):
    conn = db(); c = conn.cursor()
    c.execute("UPDATE uploads SET status=?, excel_path=?, error_msg=? WHERE id=?",
              (status, excel_path, error, job_id))
    conn.commit(); conn.close()  

from pathlib import Path
from pdf_service import extract_pdf_with_ocr   # add this import

def worker(job_id: str, pdf_path: str):
    try:
        job_dir = Path(OUTPUTS_DIR) / job_id
        # choose result mode; zip is efficient for large outputs
        result = extract_pdf_with_ocr(
            job_dir=job_dir,
            pdf_path=Path(pdf_path),
            mode="excel",     # or "zip"
            pages="all",
        )
        update_status(job_id, "verified", excel_path=result["result_path"])
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
    conn = db(); c = conn.cursor()
    c.execute("INSERT INTO uploads(id, filename, pdf_path, status, created_at) VALUES (?, ?, ?, ?, ?)",
              (job_id, file.filename, pdf_path, "pending", datetime.now().isoformat()))
    conn.commit(); conn.close()
    if background_tasks:
        background_tasks.add_task(worker, job_id, pdf_path)
    return {"job_id": job_id, "status": "pending", "message": "Processing started"}

@app.get("/api/pdf/jobs/{job_id}/status")
async def status(job_id: str):
    conn = db(); c = conn.cursor()
    c.execute("SELECT * FROM uploads WHERE id = ?", (job_id,))
    row = c.fetchone(); conn.close()
    if not row:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return {"job_id": job_id, "status": row["status"], "filename": row["filename"], "error": row["error_msg"]}

@app.get("/api/pdf/jobs/{job_id}/download")
async def download(job_id: str):
    conn = db(); c = conn.cursor()
    c.execute("SELECT * FROM uploads WHERE id = ?", (job_id,))
    row = c.fetchone(); conn.close()
    if not row or row["status"] != "verified":
        return JSONResponse({"error": "File not ready"}, status_code=400)
    xlsx_path = row["excel_path"]
    if not xlsx_path or not os.path.exists(xlsx_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(xlsx_path, filename=f"{job_id}.xlsx")

@app.get("/api/pdf/uploads")
async def recent():
    conn = db(); c = conn.cursor()
    c.execute("SELECT * FROM uploads ORDER BY created_at DESC LIMIT 10")
    rows = c.fetchall(); conn.close()
    return {"uploads": [dict(r) for r in rows]}


app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static_root")
