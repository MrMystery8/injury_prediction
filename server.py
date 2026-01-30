from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import pandas as pd
import json
import logging
from pathlib import Path
import sys

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

app = FastAPI()

# CORS for React Dev Server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
DATA_MAP = {
    "raw_injuries": "data/raw/irrazional:transfermarkt-injuries.csv", # Validated via list_dir
    "clean_panel": "data/processed/panel.parquet",
    # Detailed raw files
    "raw_profiles": "data/raw/xfkzujqjvx97n:football-datasets/player_profiles/player_profiles.csv",
    "raw_transfers": "data/raw/xfkzujqjvx97n:football-datasets/transfer_history/transfer_history.csv",
}

SCRIPT_MAP = {
    # Using python -m to run modules/scripts ensuring sys.path is correct
    "step_1_load": [sys.executable, "src/data/loader.py"], 
    "step_2_process": [sys.executable, "make_panel.py"],
    "step_3_train": [sys.executable, "train_model.py"],
    "step_4_eval": [sys.executable, "quick_eval.py"],
    
    # Granular Notebook Steps
    "step_1_1": [sys.executable, "interactive_notebook.py", "--step", "step_1_1"],
    "step_1_2": [sys.executable, "interactive_notebook.py", "--step", "step_1_2"],
    "step_1_3": [sys.executable, "interactive_notebook.py", "--step", "step_1_3"],
    "step_1_4": [sys.executable, "interactive_notebook.py", "--step", "step_1_4"],
    
    "step_2_1": [sys.executable, "interactive_notebook.py", "--step", "step_2_1"],
    "step_2_2": [sys.executable, "interactive_notebook.py", "--step", "step_2_2"],
    "step_2_3": [sys.executable, "interactive_notebook.py", "--step", "step_2_3"],
    "step_2_4": [sys.executable, "interactive_notebook.py", "--step", "step_2_4"],
    "step_2_5": [sys.executable, "interactive_notebook.py", "--step", "step_2_5"],
    
    "step_3_1": [sys.executable, "interactive_notebook.py", "--step", "step_3_1"],
    "step_3_2": [sys.executable, "interactive_notebook.py", "--step", "step_3_2"],
    "step_3_3": [sys.executable, "interactive_notebook.py", "--step", "step_3_3"],
    
    "step_4_1": [sys.executable, "interactive_notebook.py", "--step", "step_4_1"],
    "step_4_2": [sys.executable, "interactive_notebook.py", "--step", "step_4_2"],
    "step_4_3": [sys.executable, "interactive_notebook.py", "--step", "step_4_3"],
    "step_4_4": [sys.executable, "interactive_notebook.py", "--step", "step_4_4"],
    
    "step_5_1": [sys.executable, "interactive_notebook.py", "--step", "step_5_1"],
    "step_5_3": [sys.executable, "interactive_notebook.py", "--step", "step_5_3"],
}

# --- Routes ---

@app.get("/api/status")
def get_status():
    """Checks existence of key files to determine pipeline state."""
    status = {}
    for key, path_str in DATA_MAP.items():
        p = Path(path_str)
        status[key] = {
            "exists": p.exists(),
            "updated": p.stat().st_mtime if p.exists() else None
        }
    return status

@app.get("/api/data/{dataset_id}")
def get_data(dataset_id: str, limit: int = 100):
    """Returns head of dataset as JSON."""
    if dataset_id not in DATA_MAP:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    path = Path(DATA_MAP[dataset_id])
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not configured or missing")
    
    try:
        if path.suffix == '.csv':
            df = pd.read_csv(path, nrows=limit)
        elif path.suffix == '.parquet':
            # Parquet doesn't support 'nrows' efficiently like CSV, 
            # but for local files read_table -> slice -> to_pandas is okay
            df = pd.read_parquet(path)
            df = df.head(limit)
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
            
        # Convert NaN to None for valid JSON
        data = json.loads(df.to_json(orient="records", date_format="iso"))
        columns = [{"name": c, "type": str(df[c].dtype)} for c in df.columns]
        
        return {"columns": columns, "data": data}
        
    except Exception as e:
        logger.error(f"Error reading data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ScriptRequest(BaseModel):
    step_id: str
    params: dict = {}

@app.post("/api/run_step")
def run_step(req: ScriptRequest):
    """Executes a pipeline script synchronously (blocking for now)."""
    if req.step_id not in SCRIPT_MAP:
        raise HTTPException(status_code=404, detail="Step not found")
    
    cmd = SCRIPT_MAP[req.step_id].copy()
    
    # Inject params if step is training
    if req.step_id == 'step_3_train' and req.params:
        if 'learningRate' in req.params:
            cmd.extend(['--learning_rate', str(req.params['learningRate'])])
        if 'maxIter' in req.params:
            cmd.extend(['--max_iter', str(req.params['maxIter'])])
        if 'maxDepth' in req.params:
            cmd.extend(['--max_depth', str(req.params['maxDepth'])])

    # In a real app, use Celery/Redis Queue. 
    # Here, we use simple subprocess and capture output.
    try:
        logger.info(f"Running command: {cmd}")
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=Path.cwd()
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
