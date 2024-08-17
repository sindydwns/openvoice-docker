import torch
from moviepy.editor import VideoFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from langchain.document_loaders import TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from openai import OpenAI
from uuid import uuid4
from datetime import datetime
import os

map_prompt_template = '''다음 텍스트에서 제품 정보를 추출해주세요:
- 제품명
- 주요 성분
- 제품 효과
- 사용 방법
- 프로모션 정보
- 배송일 정보
- 가격 정보
- 고객 후기
- 유통기한/보관 정보
- 구성품 정보
- 대상 고객
- 특별 주의사항
- 배송비 및 조건
- 제품의 세부 사용법
- 성분의 상세 설명과 효과
- 스트리머의 사용 후기
텍스트: {text}
'''
combine_prompt_template = '''아래의 텍스트에서 추출된 제품 정보를 종합해 주세요:
{text}
'''

class Summarizer():
    def __init__(self, device="auto", use_openai_stt=False):
        load_dotenv()
        if device == "auto":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        
        # audio
        self.audio_model_id = "openai/whisper-large-v3"
        self.use_openai_stt = use_openai_stt
        if use_openai_stt:
            self.client = OpenAI()
        else:
            self.audio_pipe = self.__load_transcribe_audio_pipe()
        
        # summarize
        self.chat_model_id = "gpt-4o-mini"
        self.summarize_chain = load_summarize_chain(
                llm=ChatOpenAI(temperature=0, model_name=self.chat_model_id),
                chain_type="map_reduce",
                return_intermediate_steps=True,
                map_prompt=PromptTemplate(template=map_prompt_template, input_variables=["text"]),
                combine_prompt=PromptTemplate(template=combine_prompt_template, input_variables=["text"])
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n"],
                chunk_size=3000,
                chunk_overlap=300
        )
        
        # etc
        self.temp_folder = "temp"
        os.makedirs(self.temp_folder, exist_ok=True)

    def __generate_random_name(self, ext=None):
        path = datetime.now().strftime(f"m_%m%d_%H%M%S")
        path += f"{str(uuid4()).split('-')[0]}"
        if ext is not None:
            path += f"{path}.{ext}"
        return path

    def __load_transcribe_audio_pipe(self):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.audio_model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(self.device)
        processor = AutoProcessor.from_pretrained(self.audio_model_id)
        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens = 128,
            chunk_length_s = 30,
            batch_size = 16,
            return_timestamps = True,
            torch_dtype=self.torch_dtype,
            device=self.device
        )
    
    def __openai_stt(self, audio_file_path):
        audio_file = open(audio_file_path, "rb")
        return self.client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="text",
            temperature=0.0,
        )
    
    def stt(self, audio_file_path):
        if self.use_openai_stt:
            return self.__openai_stt(audio_file_path)
        return self.audio_pipe(audio_file_path)["text"]

    def summarize_text_file(self, text_file):
        document = TextLoader(text_file, encoding="utf-8").load()
        return self.summarize_text(document[0].page_content)

    def summarize_text(self, text):
        docs = self.text_splitter.create_documents([text])
        extracted_info = self.summarize_chain({"input_documents": docs}, return_only_outputs=True)
        product_name = self.__extract_product_name(extracted_info["output_text"])
        return {"product_name": product_name, "text": extracted_info["output_text"]}

    def __extract_product_name(self, text):
        for line in text.split("\n"):
            if "제품명" in line:
                return line.split(": ")[1]
        return ""

    def video2voice(self, video_path, output_audio_path):
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio_path)
    
    def summarize_audio_file(self, audio_file):
        audio_output_file = self.__generate_random_name(ext="mp3")
        audio_output_path = os.path.join(self.temp_folder, audio_output_file)
        self.video2voice(audio_file, audio_output_path)
        text = self.stt(audio_output_path)
        summ = self.summarize_text(text)
        return summ
