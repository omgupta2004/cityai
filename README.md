# ArQave - Archaeological Data Analysis Platform

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Technology Stack](#technology-stack)
4. [Features](#features)
5. [Installation & Setup](#installation--setup)
6. [How the Code Works](#how-the-code-works)
7. [API Endpoints](#api-endpoints)
8. [Frontend Pages](#frontend-pages)
9. [Database Schema](#database-schema)
10. [Deployment](#deployment)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

**ArQave** is a web-based platform for analyzing archaeological data, specifically designed for processing slag compositional data and extracting information from PDF documents.

### What Does It Do?

1. **PDF Data Extraction** - Extracts tables from PDF documents using OCR and table detection
2. **Clustering Analysis** - Groups similar data points using KMeans or Agglomerative clustering
3. **PCA (Principal Component Analysis)** - Reduces data dimensionality and visualizes patterns
4. **Ternary Plotting** - Plots slag compositional data on ternary phase diagrams

### Who Is It For?

Archaeologists and researchers working with:
- Slag compositional data (oxide percentages)
- PDF documents containing tabular data
- Large datasets requiring pattern analysis

---

## Project Structure

```
CITYAI/
├── app.py                      # Main FastAPI application (backend)
├── pdf_service.py              # PDF extraction logic
├── clustering_service.py       # Clustering analysis logic
├── pca_service.py              # PCA analysis logic
├── plotting_service.py         # Ternary plotting logic
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in git)
├── .gitignore                  # Git ignore rules
├── uploads.db                  # SQLite database
│
├── index.html                  # Landing page
├── phase1.html                 # Main navigation page
│
├── pages/                      # HTML pages for each feature
│   ├── pdf-extract.html
│   ├── clustering.html
│   ├── pca.html
│   ├── plotting.html
│   ├── groups.html
│   └── image-analysis.html
│
├── pages-css/                  # CSS stylesheets
│   ├── pdf-extract.css
│   ├── clustering.css
│   └── ...
│
├── pages-js/                   # JavaScript files
│   ├── pdf-extract.js
│   ├── clustering.js
│   └── ...
│
├── ternary/                    # Ternary diagram background images
│   ├── CaAlSi Ox clean str.jpg
│   ├── KAlSiOx clean.jpg
│   └── ... (5 images total)
│
├── uploads/                    # User uploaded files (not in git)
│   ├── Ceramics/
│   ├── Glass/
│   └── Slag/
│
└── outputs/                    # Generated results (not in git)
    ├── clustering/
    ├── pca/
    └── plotting/
```

---

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Python 3.12** - Programming language
- **SQLite** - Database for tracking uploads
- **Uvicorn** - ASGI server

### Data Processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning (clustering, PCA)
- **Matplotlib** - Plotting and visualization
- **Pillow** - Image processing

### PDF Processing
- **Camelot** - Table extraction from PDFs
- **Tabula** - Alternative table extraction
- **OpenCV** - Image processing for OCR

### Frontend
- **HTML/CSS/JavaScript** - User interface
- **Fetch API** - AJAX requests to backend

---

## Features

### 1. PDF Data Extraction

**What it does:**
- Uploads PDF files
- Detects and extracts tables using OCR
- Converts tables to Excel/CSV format
- Categorizes data (Ceramics, Glass, Slag)

**How it works:**
1. User uploads PDF via `pages/pdf-extract.html`
2. Backend (`pdf_service.py`) processes PDF:
   - Uses Camelot to detect tables
   - Falls back to Tabula if Camelot fails
   - Applies OCR if needed
3. Extracted tables saved to `outputs/<job_id>/`
4. User downloads Excel file

**Key Files:**
- `pdf_service.py` - Extraction logic
- `pages/pdf-extract.html` - Upload interface
- API: `POST /api/upload`, `GET /api/status/{job_id}`

---

### 2. Clustering Analysis

**What it does:**
- Groups similar data points into clusters
- Supports KMeans and Agglomerative algorithms
- Generates scatter plot visualization
- Exports clustered data with cluster labels

**How it works:**
1. User uploads CSV/Excel via `pages/clustering.html`
2. Selects algorithm (KMeans/Agglomerative) and number of clusters
3. Backend (`clustering_service.py`):
   - Detects numeric columns
   - Normalizes and scales data
   - Applies clustering algorithm
   - Uses PCA to reduce to 2D for visualization
   - Creates scatter plot with cluster colors
4. Returns plot (base64) and clustered CSV

**Key Files:**
- `clustering_service.py` - Clustering logic
- `pages/clustering.html` - Interface
- API: `POST /api/clustering/run`

**Algorithms:**
- **KMeans**: Partitions data into k clusters by minimizing variance
- **Agglomerative**: Hierarchical clustering, builds tree of clusters

---

### 3. PCA (Principal Component Analysis)

**What it does:**
- Reduces high-dimensional data to 2-3 principal components
- Visualizes data variance and patterns
- Exports scores and loadings

**How it works:**
1. User uploads CSV/Excel via `pages/pca.html`
2. Selects number of components (2 or 3)
3. Backend (`pca_service.py`):
   - Normalizes and scales data
   - Applies PCA transformation
   - Generates scatter plot (2D or 3D)
   - Calculates explained variance
4. Returns plot, scores, and loadings

**Key Files:**
- `pca_service.py` - PCA logic
- `pages/pca.html` - Interface
- API: `POST /api/pca/run`

**Output Files:**
- `scores.csv` - Transformed data in PC space
- `loadings.csv` - Variable contributions to PCs
- `pca_plot.png` - Visualization

---

### 4. Ternary Plotting

**What it does:**
- Plots slag compositional data on ternary phase diagrams
- Automatically selects best diagram based on oxide composition
- Overlays data points on phase diagram backgrounds

**How it works:**
1. User uploads slag data (CSV/Excel) via `pages/plotting.html`
2. Selects mode:
   - **Auto**: System chooses diagram based on oxide dominance
   - **Manual**: User selects specific ternary system
3. Backend (`plotting_service.py`):
   - Reads oxide data (CaO, SiO2, Al2O3, FeO, etc.)
   - Groups oxides (e.g., SiO2 + TiO2)
   - Normalizes to 100%
   - Selects ternary system using rules
   - Converts ternary coords to cartesian (x, y)
   - Overlays points on background diagram
4. Returns plot with background and data points

**Key Files:**
- `plotting_service.py` - Ternary logic
- `pages/plotting.html` - Interface
- `ternary/` - Background diagram images
- API: `POST /api/plotting/run`

**Ternary Systems:**
1. CaO–SiO2–Al2O3
2. K2O–SiO2–Al2O3
3. FeO–SiO2–Al2O3
4. MnO–SiO2–Al2O3
5. FeO–CaO–SiO2

**Selection Rules:**
- If Al₂O₃ ≤ 5 and CaO > 1 and FeO > 1 → FeO–CaO–SiO2
- If K₂O > CaO or K₂O > FeO → K2O–SiO2–Al2O3
- If CaO dominant → CaO–SiO2–Al2O3
- If FeO dominant → FeO–SiO2–Al2O3
- If MnO ≥ FeO → MnO–SiO2–Al2O3

---

## Installation & Setup

### Prerequisites
- Python 3.12+
- pip (Python package manager)
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/omgupta2004/cityai.git
cd cityai
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
```
fastapi
uvicorn
python-multipart
camelot-py[cv]
tabula-py
pandas
openpyxl
scikit-learn
matplotlib
python-ternary
pillow
```

### Step 3: Create Required Directories
```bash
mkdir -p uploads outputs
```

### Step 4: Initialize Database
The SQLite database (`uploads.db`) is created automatically on first run.

### Step 5: Run Server
```bash
uvicorn app:app --reload
```

Server runs at: `http://localhost:8000`

---

## How the Code Works

### Backend Architecture (app.py)

**1. Imports and Setup**
```python
from fastapi import FastAPI, UploadFile, File
import sys
from pathlib import Path

# Add current directory to Python path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import service modules
from pdf_service import extract_pdf_with_ocr
from clustering_service import process_clustering_job
from pca_service import process_pca_job
from plotting_service import process_plotting_job, TERNARY_MAP
```

**Why sys.path?** Ensures service modules can be imported regardless of working directory (fixes production deployment issues).

**2. FastAPI App Initialization**
```python
app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Mount static files
app.mount("/pages", StaticFiles(directory="pages"), name="pages")
app.mount("/ternary", StaticFiles(directory="ternary"), name="ternary")
```

**3. Database Connection**
```python
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
```

**4. API Endpoints**

Each endpoint follows this pattern:
```python
@app.post("/api/feature/run")
async def feature_run(file: UploadFile = File(...), param: str = "default"):
    # 1. Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # 2. Save uploaded file
    job_dir = Path(OUTPUTS_DIR) / "feature" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / f"input{file_ext}"
    
    # 3. Process file (call service module)
    result = process_feature_job(input_path, job_dir, param)
    
    # 4. Return JSON response
    return JSONResponse({
        "job_id": job_id,
        "download": f"/api/feature/download/{job_id}",
        "metadata": result
    })
```

---

### Service Modules

#### clustering_service.py

**Main Function:**
```python
def process_clustering_job(input_file, output_dir, n_clusters=3, algorithm='KMeans'):
    # 1. Read file (CSV or Excel)
    df = pd.read_csv(input_file) or pd.read_excel(input_file)
    
    # 2. Detect numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 3. Normalize and scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])
    
    # 4. Apply clustering
    if algorithm == 'KMeans':
        model = KMeans(n_clusters=n_clusters)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)
    
    clusters = model.fit_predict(X_scaled)
    
    # 5. Create visualization using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
    
    # 6. Save plot and clustered data
    # Returns metadata dict
```

**Key Concepts:**
- **StandardScaler**: Normalizes features to mean=0, std=1
- **PCA for visualization**: Reduces to 2D for plotting
- **Cluster labels**: Added as new column to original data

---

#### pca_service.py

**Main Function:**
```python
def process_pca_job(input_file, output_dir, n_components=2):
    # 1. Read and prepare data
    df = pd.read_csv(input_file) or pd.read_excel(input_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 2. Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])
    
    # 3. Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # 4. Calculate explained variance
    explained_var = pca.explained_variance_ratio_
    
    # 5. Create visualization
    if n_components == 2:
        plt.scatter(X_pca[:, 0], X_pca[:, 1])
    else:  # 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])
    
    # 6. Save scores, loadings, plot
```

**Key Concepts:**
- **Scores**: Transformed data in PC space
- **Loadings**: Original variable contributions to PCs
- **Explained variance**: How much variance each PC captures

---

#### plotting_service.py

**Main Function:**
```python
def process_plotting_job(input_file, output_dir, mode='auto', system=None, include_k2o=True):
    # 1. Read oxide data
    df = pd.read_csv(input_file) or pd.read_excel(input_file)
    
    # 2. Group oxides
    C_SiO2_Group = SiO2 + TiO2
    C_Al2O3_Group = Al2O3
    minor_basics = MnO + MgO + K2O_val
    
    # 3. Select ternary system (auto or manual)
    if mode == 'auto':
        selected_system = choose_ternary(avg_row, include_k2o, TERNARY_MAP)
    else:
        selected_system = system
    
    # 4. Calculate normalized coordinates for each sample
    for row in df:
        C1, C2, C3 = calculate_components(row, selected_system)
        total = C1 + C2 + C3
        Norm_Right = 100 * C2 / total
        Norm_Top = 100 * C1 / total
        Norm_Left = 100 * C3 / total
    
    # 5. Plot on ternary diagram
    plot_ternary(results, base_dir)
```

**Coordinate Transformation:**
```python
# Ternary to Cartesian conversion
x = right + top / 2
y = top * sqrt(3) / 2
```

**Why this formula?**
- Ternary diagrams are equilateral triangles
- Standard triangle: base = 100, height = 86.6 (100 * sqrt(3)/2)
- Formula maps (right, top, left) to (x, y) coordinates

---

### Frontend Architecture

Each page follows this structure:

**1. HTML Structure**
```html
<div class="card">
  <input id="file" type="file">
  <select id="param">...</select>
  <button id="run">Run Analysis</button>
  <div id="msg"></div>
  <div id="out"></div>
</div>
```

**2. JavaScript Logic**
```javascript
$('#run').onclick = async () => {
  // 1. Get file and parameters
  const file = $('#file').files[0];
  const param = $('#param').value;
  
  // 2. Create FormData
  const fd = new FormData();
  fd.append('file', file);
  
  // 3. Send POST request
  const response = await fetch(`/api/feature/run?param=${param}`, {
    method: 'POST',
    body: fd
  });
  
  // 4. Parse JSON response
  const data = await response.json();
  
  // 5. Display results
  $('#out').innerHTML = `
    <img src="data:image/png;base64,${data.plot_base64}">
    <a href="${data.download}">Download</a>
  `;
};
```

---

## API Endpoints

### PDF Extraction
- `POST /api/upload` - Upload PDF for extraction
- `GET /api/status/{job_id}` - Check extraction status
- `GET /api/download/{job_id}` - Download extracted Excel
- `GET /api/files/{category}/jobs` - List jobs by category

### Clustering
- `POST /api/clustering/run?n_clusters=3&algo=KMeans` - Run clustering
- `GET /api/clustering/download/{job_id}/csv` - Download clustered data
- `GET /api/clustering/download/{job_id}/plot` - Download plot PNG

### PCA
- `POST /api/pca/run?n_components=2` - Run PCA
- `GET /api/pca/download/{job_id}/scores` - Download scores CSV
- `GET /api/pca/download/{job_id}/loadings` - Download loadings CSV
- `GET /api/pca/download/{job_id}/plot` - Download plot PNG

### Ternary Plotting
- `POST /api/plotting/run?mode=auto&include_k2o=true` - Generate plot
- `GET /api/plotting/download/{job_id}/plot` - Download plot PNG
- `GET /api/plotting/download/{job_id}/data` - Download normalized CSV
- `GET /api/plotting/systems` - List available ternary systems

### Utility
- `GET /health` - Health check
- `GET /` - Redirect to index.html

---

## Database Schema

**Table: uploads**
```sql
CREATE TABLE uploads (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    category TEXT NOT NULL,
    upload_time TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    output_file TEXT
);
```

**Fields:**
- `id`: Unique job ID (UUID)
- `filename`: Original uploaded filename
- `category`: Ceramics, Glass, or Slag
- `upload_time`: ISO timestamp
- `status`: pending, processing, completed, failed
- `output_file`: Path to extracted Excel file

---

## Deployment

### Local Development
```bash
uvicorn app:app --reload
```

### Production (VPS/Server)

**Option 1: Direct Deployment**
```bash
# 1. Upload files via FTP
# 2. SSH into server
cd /path/to/CITYAI
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Option 2: GitHub Deployment (Recommended)**
```bash
# On server:
git clone https://github.com/omgupta2004/cityai.git
cd cityai
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000

# To update:
git pull origin main
# Restart server
```

**Using systemd (Linux)**
Create `/etc/systemd/system/arqave.service`:
```ini
[Unit]
Description=ArQave FastAPI Application
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/CITYAI
ExecStart=/usr/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable arqave
sudo systemctl start arqave
```

---

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'clustering_service'**

**Cause:** Python can't find service modules

**Fix:** The sys.path fix in `app.py` should resolve this. Ensure you're running from the CITYAI directory.

```bash
cd /path/to/CITYAI
uvicorn app:app
```

---

**2. 404 Not Found on API endpoints**

**Cause:** Server not restarted after code changes

**Fix:** Restart uvicorn server
```bash
# Kill old process
ps aux | grep uvicorn
kill -9 <PID>

# Start new process
uvicorn app:app
```

---

**3. "No module named 'sklearn'"**

**Cause:** Dependencies not installed

**Fix:**
```bash
pip install scikit-learn
# or
pip install -r requirements.txt
```

---

**4. Ternary plot points in wrong positions**

**Cause:** Coordinate transformation issue

**Fix:** Already fixed in current code. Uses formula:
```python
x = right + top / 2
y = top * sqrt(3) / 2
```

---

**5. Database locked error**

**Cause:** Multiple processes accessing SQLite

**Fix:** Use connection pooling or switch to PostgreSQL for production

---

**6. CORS errors in browser**

**Cause:** Frontend and backend on different domains

**Fix:** Already configured in `app.py`:
```python
app.add_middleware(CORSMiddleware, allow_origins=["*"])
```

---

## Development Tips

### Adding a New Feature

1. **Create service module** (`new_feature_service.py`)
2. **Add endpoint in app.py**
3. **Create HTML page** (`pages/new-feature.html`)
4. **Add to navigation** (update `phase1.html`)
5. **Test locally**
6. **Commit and push to GitHub**

### Code Style

- Use descriptive variable names
- Add docstrings to functions
- Follow PEP 8 for Python
- Keep functions under 50 lines
- Comment complex logic

### Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test clustering
curl -X POST -F "file=@data.csv" \
  "http://localhost:8000/api/clustering/run?n_clusters=3&algo=KMeans"
```

---

## Project History

- **Initial Version**: PDF extraction only
- **v1.1**: Added clustering analysis
- **v1.2**: Added PCA analysis
- **v1.3**: Added ternary plotting
- **v1.4**: Fixed production deployment issues
- **v1.5**: Pushed to GitHub

---

## Credits

**Developer:** Om Gupta  
**GitHub:** https://github.com/omgupta2004/cityai  
**Production:** https://arqave.cityai.space

---

## License

[Add your license here]

---

## Contact

For questions or issues, contact: [your email]

---

**Last Updated:** November 28, 2024
