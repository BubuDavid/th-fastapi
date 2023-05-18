from fastapi import FastAPI, UploadFile, File
from ai_tools.classificator import classificate
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
async def root():
    return {"message": "Hello World"}

@app.post("/cat-vs-dog/")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        classification = classificate(contents)
        return {"filename": file.filename, "classification": classification, "status_code": 200}
    except Exception as e:
        return {"error": f"An error occurred: {e}", "status_code": 400}