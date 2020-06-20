import os, io
from fmnist import FMNIST
from style_transfer import do
from PIL import Image
from fastapi import FastAPI, Request, File, UploadFile
from typing import List
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

model = FMNIST()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/files", StaticFiles(directory="files"), name="files")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_item(request: Request):
    return templates.TemplateResponse("style_home.html", {"request": request})

@app.post("/style_transfer")
def predict(request: Request, files:List[UploadFile]=File(...)):
    print(len(files))
    assert len(files) == 2, 'excpecting two images uploaded'
    images = []
    file1, file2 = files
    for file in files:
        content = file.file.read()
        images.append(Image.open(io.BytesIO(content)).convert('RGB'))
    output_image = do(images)
    for f,im in zip(['src.png','style.png','output.png'], [images[0], images[1], output_image]):
        im.save(f'./files/{f}')
    payload = {'request': request, 'source': 'src.png', 'style': 'style.png', 'output': 'output.png'}
    return templates.TemplateResponse("style_home.html", payload)
