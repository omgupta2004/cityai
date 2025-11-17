import os
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import camelot
import pdfplumber

import os, json, shutil, subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import camelot


def has_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def run_ocrmypdf(in_pdf: Path, out_pdf: Path) -> None:
    
    cmd = [
        "ocrmypdf",
        "--deskew",
        "--rotate-pages",
        "--remove-background",
        "--optimize", "3",
        "--force-ocr",
        str(in_pdf), str(out_pdf)
    ]
    subprocess.run(cmd, check=True)

def is_numeric_like(df: pd.DataFrame) -> bool:
    num_ratio = df.apply(lambda s: pd.to_numeric(s, errors="coerce")).notna().mean().mean()
    return num_ratio >= 0.30

def passes_basic_table_filters(df: pd.DataFrame) -> Tuple[bool, str]:
    if df is None or df.empty:
        return False, "empty"
    rows, cols = df.shape
    if rows < 2 or cols < 2:
        return False, "too_small"
    empty_ratio = df.replace("", None).isna().mean().mean()
    if empty_ratio > 0.40:
        return False, "too_sparse"
    header_mean_len = df.iloc[0].fillna("").astype(str).str.len().mean()
    if header_mean_len < 2:
        return False, "short_headers"
    return True, "ok"


def tabula_extract_csvs(pdf_path: Path, out_dir: Path, pages: str = "all") -> List[Path]:
    """
    Uses tabula-py CLI (Java) to export a single CSV; split into per-table later if needed.
    For more control, you can run multiple convert_into calls with areas.
    """
    created: List[Path] = []
    if not has_cmd("java"):
        return created
    out_csv = out_dir / "tabula_all.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import tabula
        tabula.convert_into(str(pdf_path), str(out_csv), output_format="csv", pages=pages, guess=False)
        if out_csv.exists() and out_csv.stat().st_size > 0:
            created.append(out_csv)
    except Exception:
        
        try:
            subprocess.run(
                ["java", "-jar", "tabula.jar", "-p", pages, "-f", "CSV", "-o", str(out_csv), str(pdf_path)],
                check=True
            )
            if out_csv.exists() and out_csv.stat().st_size > 0:
                created.append(out_csv)
        except Exception:
            pass
    return created


def camelot_lattice_tables(pdf_path: Path, pages: str = "all", line_scale: Optional[int] = 50) -> List[pd.DataFrame]:
    kwargs = dict(pages=pages, flavor="lattice")
    if line_scale is not None:
        kwargs["line_scale"] = line_scale
    try:
        tables = camelot.read_pdf(str(pdf_path), **kwargs)
        return [t.df for t in tables] if tables and tables.n > 0 else []
    except Exception:
        return []


def extract_pdf_with_ocr(
    job_dir: Path,
    pdf_path: Path,
    *,
    mode: str = "excel",     
    pages: str = "all",
) -> dict:
    job_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = job_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    manifest = {
        "mode": mode,
        "steps": [],
        "tables_kept": 0,
        "tables_dropped": 0,
        "drop_reasons": {},
        "paths": {}
    }

    #OCR
    ocred = job_dir / "ocred.pdf"
    if has_cmd("ocrmypdf"):
        run_ocrmypdf(pdf_path, ocred)
        manifest["steps"].append("ocr")
        input_for_extract = ocred
    else:
        manifest["steps"].append("no_ocr_binary_on_path")
        input_for_extract = pdf_path

    #Tabula stream first (guess=False)
    kept: List[pd.DataFrame] = []
    dropped = 0
    drop_reasons = {}

    tabula_csvs = tabula_extract_csvs(input_for_extract, tables_dir, pages=pages)
    if tabula_csvs:
       
        for p in tabula_csvs:
            try:
                df = pd.read_csv(p)
                ok, reason = passes_basic_table_filters(df)
                if not ok:
                    dropped += 1
                    drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                else:
                    kept.append(df)
            except Exception:
                dropped += 1
                drop_reasons["read_error"] = drop_reasons.get("read_error", 0) + 1

    #Camelot lattice fallback 
    if not kept:
        manifest["steps"].append("camelot_lattice_fallback")
        cdfs = camelot_lattice_tables(input_for_extract, pages=pages, line_scale=50)
        for df in cdfs:
            ok, reason = passes_basic_table_filters(df)
            if not ok:
                dropped += 1
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            else:
                kept.append(df)

    
    tables_dir.mkdir(exist_ok=True)
    csv_paths: List[Path] = []
    for i, df in enumerate(kept, start=1):
        cp = tables_dir / f"table_{i:04d}.csv"
        df.to_csv(cp, index=False)
        csv_paths.append(cp)

    manifest["tables_kept"] = len(kept)
    manifest["tables_dropped"] = dropped
    manifest["drop_reasons"] = drop_reasons

    result_path: Optional[Path] = None
    if mode == "excel":
        out_xlsx = job_dir / "extracted.xlsx"
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            for i, df in enumerate(kept, start=1):
                df.to_excel(writer, index=False, sheet_name=f"Table_{i}")
            if kept:
                pd.concat(kept, ignore_index=True).to_excel(writer, index=False, sheet_name="Combined")
        result_path = out_xlsx
    else:
        import zipfile
        zip_path = job_dir / "tables.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in csv_paths:
                z.write(p, arcname=p.name)
        result_path = zip_path

    manifest["paths"]["result"] = str(result_path)
    manifest["paths"]["tables_dir"] = str(tables_dir)
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"tables": len(kept), "result_path": str(result_path), "mode": mode}




def _camelot_read(pdf_path: str, pages: str, flavor: str, line_scale: Optional[int] = None, table_areas: Optional[List[str]] = None):
    kwargs = dict(pages=pages, flavor=flavor)
    if line_scale is not None:
        kwargs["line_scale"] = line_scale
    if table_areas:
        kwargs["table_areas"] = table_areas
    return camelot.read_pdf(pdf_path, **kwargs)

def extract_tables_batch(
    pdf_path: str,
    pages: str = "all",
    flavor_order: Tuple[str, ...] = ("lattice", "stream"),
    line_scale: Optional[int] = None,
    table_areas: Optional[List[str]] = None,
) -> List[pd.DataFrame]:
    """
    Camelot with given flavors; if none found, fallback to pdfplumber for the same pages.
    Returns list of DataFrames (can be large).
    """
    dfs: List[pd.DataFrame] = []

    
    for flavor in flavor_order:
        try:
            tables = _camelot_read(pdf_path, pages, flavor, line_scale, table_areas)
            if tables and tables.n > 0:
                dfs.extend([t.df for t in tables])
                return dfs  
        except Exception:
            
            pass

    
    try:
        if pages == "all":
            page_ranges = None
        else:
            
            page_ranges = _parse_pages(pages)

        with pdfplumber.open(pdf_path) as pdf:
            idxs = page_ranges if page_ranges else list(range(1, len(pdf.pages) + 1))
            for pno in idxs:
                if 1 <= pno <= len(pdf.pages):
                    page = pdf.pages[pno - 1]
                    tbl = page.extract_table()
                    if tbl:
                        dfs.append(pd.DataFrame(tbl[1:], columns=tbl[0]))
    except Exception:
        pass

    return dfs

def _parse_pages(pages: str) -> List[int]:
    nums: List[int] = []
    for part in pages.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            nums.extend(list(range(a, b + 1)))
        else:
            nums.append(int(part))
    return nums



def extract_pdf_to_outputs(
    job_dir: Path,
    pdf_path: Path,
    *,
    mode: str = "excel",                  
    pages: str = "all",
    batch_size: int = 25,
    flavor_order: Tuple[str, ...] = ("lattice", "stream"),
    line_scale: Optional[int] = None,
    table_areas: Optional[List[str]] = None,
) -> dict:
    """
    Scalable extractor:
    - Reads PDF in batches of pages.
    - Writes per-table CSV files into job_dir/tables/.
    - If mode == "excel", also writes a sheet-per-table Excel and a Combined sheet.
    - Writes manifest.json with stats and parameters.
    Returns dict with {"tables": N, "result_path": str, "mode": mode}
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = job_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    
    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)

    if pages == "all":
        page_list = list(range(1, total_pages + 1))
    else:
        page_list = _parse_pages(pages)
        page_list = [p for p in page_list if 1 <= p <= total_pages]

    table_count = 0
    combined_frames: List[pd.DataFrame] = []

    # Process in batches
    for i in range(0, len(page_list), batch_size):
        batch_pages = page_list[i : i + batch_size]
        page_spec = ",".join(map(str, batch_pages))

        dfs = extract_tables_batch(
            str(pdf_path),
            pages=page_spec,
            flavor_order=flavor_order,
            line_scale=line_scale,
            table_areas=table_areas,
        )
        # write per-table CSVs
        for j, df in enumerate(dfs, start=1):
            table_count += 1
            out_csv = tables_dir / f"table_{table_count:05d}_p{batch_pages[0]}-{batch_pages[-1]}.csv"
            df.to_csv(out_csv, index=False)
            combined_frames.append(df)

    result_path: Optional[Path] = None

    if mode == "excel":
        
        max_sheets = 300  
        out_xlsx = job_dir / "extracted.xlsx"
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:

            for n, df in enumerate(combined_frames[:max_sheets], start=1):
                sheet = f"Table_{n}"
                df.to_excel(writer, index=False, sheet_name=sheet)
            
            if combined_frames:
                pd.concat(combined_frames, ignore_index=True).to_excel(writer, index=False, sheet_name="Combined")
        result_path = out_xlsx
    else:
        
        import zipfile
        zip_path = job_dir / "tables.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in tables_dir.glob("*.csv"):
                z.write(p, arcname=p.name)
        result_path = zip_path

    
    manifest = {
        "mode": mode,
        "total_pages": total_pages,
        "processed_pages": len(page_list),
        "tables_found": table_count,
        "params": {
            "pages": pages,
            "batch_size": batch_size,
            "flavor_order": list(flavor_order),
            "line_scale": line_scale,
            "table_areas": table_areas or [],
        },
        "paths": {
            "result": str(result_path),
            "tables_dir": str(tables_dir),
        },
    }
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return {"tables": table_count, "result_path": str(result_path), "mode": mode}
