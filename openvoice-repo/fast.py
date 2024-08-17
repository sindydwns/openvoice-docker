from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return JSONResponse({"meta": "ai"})

@app.post("/video2text")
async def video2text():
    with open("temp/stt_file.txt", "r") as file:
        return JSONResponse({"text": file.read()})

class SummarizeModel(BaseModel):
    text: str

@app.get("/summarize")
async def summarize(data: SummarizeModel):
    return JSONResponse({"product_name": "item", "text": "summarized content"})

@app.get("/hightlight")
async def hightlight():
    return JSONResponse({"data": [ 
        {"time": 30, "heart": 10, "chat_cnt":14},
        {"time": 42, "heart": 7, "chat_cnt": 20}]})

class ChatModel(BaseModel):
    type: int
    text: str
    
@app.post("/chat")
async def chat(data: ChatModel):
    return JSONResponse({"text": "return chat"})



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
