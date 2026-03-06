from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import json

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)