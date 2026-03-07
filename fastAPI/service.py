from config import venv_vits, script_vits, folder_vits, venv_llama, script_llama, folder_llama
import subprocess
from fastapi import HTTPException
import json


def call_model_vits(query: str):
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
    
    return path


def call_model_llama(memory_llama: list, message: str):
    memory_llama.append({"role": "user", "content": message})
    result = subprocess.run(
        [
            str(venv_llama),
            str(script_llama),
            "--messages", json.dumps(memory_llama)
        ],
        capture_output=True,
        text=True,
        cwd=str(folder_llama)
    )

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise HTTPException(status_code=500, detail="Error al generar el texto")

    text = None
    for line in result.stdout.splitlines():
        if line.startswith("RESULT_TEXT="):
            text = line.replace("RESULT_TEXT=", "").strip()
            break

    if not text:
        raise HTTPException(status_code=500, detail="No se pudo obtener el texto")

    memory_llama.append({"role": "assistant", "content": text})
    return text