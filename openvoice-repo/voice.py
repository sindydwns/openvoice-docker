import sys, os
import torch
from openvoice.api import ToneColorConverter
import openvoice.se_extractor as se_extractor
from faster_whisper import WhisperModel
import nltk
from melo.api import TTS

nltk.download('averaged_perceptron_tagger_eng')
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def create_tone_color_converter(checkpoint_folder: str):
    checkpoint_path = os.path.join(checkpoint_folder, "converter", "config.json")
    converter = ToneColorConverter(checkpoint_path, device=device)
    trained_weight = os.path.join(checkpoint_folder, "converter", "checkpoint.pth")
    converter.load_ckpt(trained_weight)
    return converter


class Model():
    
    def __init__(self, tone_color_converter: ToneColorConverter, lang: str="KR"):
        self.tone_color_converter = tone_color_converter
        self.lang = lang
        self.model = TTS(language=lang, device=device)
        self.speaker_id = self.model.hps.data.spk2id["KR"]
        self.src_se = None
        self.target_se = None


    def train(self, speaker_path, output_path):
        se_extractor.model = WhisperModel("medium", device="cpu", compute_type="float32")
        target_se, _ = se_extractor.get_se(speaker_path, self.tone_color_converter, vad=False)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(target_se.cpu(), output_path)
        del se_extractor.model


    def load(self, src_se_path: str, target_se_path: str):
        if self.src_se is not None:
            del self.src_se
        if self.target_se is not None:
            del self.target_se
        print(f"load src: {src_se_path}")
        print(f"load target: {target_se_path}")
        self.src_se = torch.load(src_se_path, map_location=device)
        self.target_se = torch.load(target_se_path, map_location=device)


    def tts(self, text: str, output_path: str):
        self.model.tts_to_file(text, self.speaker_id, output_path)


    def tone_color(self, src_path: str, output_path: str):
        encode_message = "meta"
        self.tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=self.src_se, 
            tgt_se=self.target_se, 
            output_path=output_path,
            message=encode_message)