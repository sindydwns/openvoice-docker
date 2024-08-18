from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from chatbot import RecommendChatbot, SummarizeChatbot
from summarize import Summarizer
import voice

app = FastAPI()

summarizer = Summarizer()
tone_color_converter = voice.create_tone_color_converter("checkpoints_v2")
voice_model = voice.Model(tone_color_converter)
voice_model.load("checkpoints_v2/base_speakers/ses/kr.pth", "temp/pretrained.pth")

### audio test ###
audio1_path = "temp/output_n.wav"
audio2_path = "temp/output_c.wav"
voice_model.tts("hello world", audio1_path)
voice_model.tone_color(audio1_path, audio2_path)
##################

file = open("temp/stt_file.txt", "r")
file_content = file.read()
file.close()

recommend_chatbot = RecommendChatbot(file_content)
summarize_chatbot = SummarizeChatbot(file_content)

@app.get("/")
async def root():
    return JSONResponse({"meta": "ai"})

@app.post("/video2text")
async def video2text():
    return JSONResponse({"text": file_content})

class SummarizeModel(BaseModel):
    text: str

@app.get("/summarize")
async def summarize(data: SummarizeModel):
    res = summarizer.summarize_text(data.text)
    return JSONResponse(res)

@app.get("/highlight")
async def hightlight():
    return JSONResponse({"data": [ 
        {"time": 32, "heart": 30, "chat_cnt":0},
        {"time": 47, "heart": 32, "chat_cnt": 0},
        {"time": 88, "heart": 32, "chat_cnt": 2},
        {"time": 187, "heart": 31, "chat_cnt": 2},
        {"time": 324, "heart": 49, "chat_cnt": 10}, 
        {"time": 429, "heart": 30, "chat_cnt":2},
        {"time": 1320, "heart": 58, "chat_cnt": 10},
        {"time": 1705, "heart": 34, "chat_cnt": 12},
        {"time": 2287, "heart": 81, "chat_cnt": 31},
        {"time": 2308, "heart": 62, "chat_cnt": 25},
        {"time": 2731, "heart": 44, "chat_cnt": 9}, 
        {"time": 3632, "heart": 120, "chat_cnt":35},
        {"time": 3699, "heart": 64, "chat_cnt":16},
    ]})

class ChatModel(BaseModel):
    type: int
    text: str
    
@app.post("/chat")
async def chat(data: ChatModel):
    if data.type == 0:
        res = recommend_chatbot.pred(data.text)
    else:
        res = summarize_chatbot.pred(data.text)
    return JSONResponse({"text": res})

from uuid import uuid4
from datetime import datetime

class HostVoiceModel(BaseModel):
    text: str
    use_openai: bool=False
    speed: float=1.0

def generate_random_name(version="", ext=None):
    path = datetime.now().strftime(f"v_%m%d_%H%M%S_{version}")
    path += f"{str(uuid4()).split('-')[0]}"
    if ext is not None:
        path += f"{path}.{ext}"
    return path

@app.post("/hostvoice")
async def hostvoice(data: HostVoiceModel):
    audio1_path = "temp/" + generate_random_name("n_", "wav")
    audio2_path = "temp/" + generate_random_name("c_", "wav")
    voice_model.tts(data.text, audio1_path, use_openai=data.use_openai, speed=data.speed)
    voice_model.tone_color(audio1_path, audio2_path)
    return FileResponse(audio2_path, filename="output.wav", media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
