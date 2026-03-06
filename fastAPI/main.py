from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from config import embed_model, collection, venv_vits, script_vits, folder_vits
import json
import subprocess


app = FastAPI(
    title="KeepCoding",
    description="Prueba Final keepcoding :)",
    version="1.0.0"
)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    
    orders = []
    with open("database/orders.json", "r") as f:
        orders = json.load(f)

    print(orders)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "orders": orders
    })


@app.get("/knowledge")
def read_knowledge(query: str, request: Request):

    query_embedding = embed_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    return {
        "query": query,
        "results": results
    }


from fastapi import HTTPException, Request
import subprocess

@app.get("/generate_speech")
def generate_speech(query: str, request: Request):

    result = subprocess.run(
        [
            str(venv_vits),
            str(script_vits),
            "--text", query
        ],
        capture_output=True,
        text=True,
        cwd=str(folder_vits)
    )

    if result.returncode != 0:
        print(result.stderr)
        raise HTTPException(status_code=500, detail="Error al generar el audio")

    path = None
    for line in result.stdout.splitlines():
        if line.startswith("RESULT_PATH="):
            path = line.replace("RESULT_PATH=", "").strip()
            break

    if not path:
        raise HTTPException(status_code=500, detail="No se pudo obtener la ruta del audio")

    return {
        "query": query,
        "path": path
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)