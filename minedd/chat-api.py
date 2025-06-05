from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from minedd.pdf_manager import DcoumentPDF

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/pdf_as_markdown")
def load_pdf_as_markdown(pdf_path: str):
    pdf_manager = DcoumentPDF(pdf_path=pdf_path)
    try:
        markdown_text = pdf_manager.convert_to_markdown()
        return {"markdown": markdown_text}
    except Exception as e:
        return {"error": str(e)}