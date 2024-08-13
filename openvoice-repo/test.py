import os
import torch
import nltk
import sys
from melo.api import TTS
from openvoice.api import ToneColorConverter
import openvoice.se_extractor as se_extractor
from faster_whisper import WhisperModel

if __name__ == "__main__":
    reference_speaker = sys.argv[1]
    output_file = sys.argv[2]

ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

se_extractor.model = WhisperModel("medium", device="cpu", compute_type="float32")
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)

os.makedirs(output_dir, exist_ok=True)


nltk.download('averaged_perceptron_tagger_eng')


texts = {
    'KR': "안녕하세요! 오늘은 날씨가 정말 좋네요.",
}

src_path = f'{output_dir}/tmp.wav'

# Speed is adjustable
speed = 1.0

for language, text in texts.items():
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    
    for speaker_key in speaker_ids.keys():
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace('_', '-')
        
        source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
        model.tts_to_file(text, speaker_id, src_path, speed=speed)

        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=output_file,
            message=encode_message)
