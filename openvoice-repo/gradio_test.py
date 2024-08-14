import gradio as gr
from glob import glob
from scipy.io.wavfile import write
import os
import shutil
import facade
base_speaker_se_path = "checkpoints_v2/base_speakers/ses/kr.pth"
converter = facade.create_tone_color_converter("checkpoints_v2")
train_model = facade.Model(converter)
model_dropdown = None
test_models = {}

def train(audio: str):
    files = glob("resources/*")
    res = list(filter(lambda x: os.path.basename(x).startswith("voice_"), files))
    res = list(map(lambda x: x.split("_")[1], res))
    res = list(map(lambda x: int(x.split(".")[0]), res))
    if len(res) == 0:
        res = [0]
    new_num = max(res) + 1
    new_file_name = f"voice_{new_num}"
    origin_voice_path = f"resources/{new_file_name}.{audio.split('.')[-1]}"
    train_voice_path = f"voice/{new_file_name}.pth"
    shutil.copyfile(audio, origin_voice_path)
    train_model.train(origin_voice_path, train_voice_path)
    return new_file_name

def tts(target_model: str, text: str):
    src_output_path = "outputs_v2/output1.wav"
    target_output_path = "outputs_v2/output2.wav"
    if target_model not in test_models:
        model = facade.Model(converter)
        model.load(base_speaker_se_path, target_model)
        test_models[target_model] = model
    model = test_models[target_model]
    model.tts(text, src_output_path)
    model.tone_color(src_output_path, target_output_path)
    return src_output_path, target_output_path

with gr.Blocks(theme=gr.themes.Default()) as demo:
    with gr.Tab("(임시)TTS"):
        with gr.Row():
            gr.Interface(fn=train, inputs=[gr.Audio(type="filepath")], outputs=[gr.Text()])
        with gr.Row():
            files = glob("voice/*")
            choices = list(filter(lambda x: os.path.basename(x).startswith("voice_"), files))
            model_dropdown = gr.Dropdown(choices=choices, label="Tone color", info="select voice tone color")
            gr.Interface(fn=tts, inputs=[model_dropdown, gr.Text()], outputs=[gr.Audio(), gr.Audio()])
        
        
demo.launch()
