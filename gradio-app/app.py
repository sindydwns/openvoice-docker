import requests
import gradio as gr
import pandas as pd
from gradio_toggle import Toggle

host = "http://openvoice:8080"

def post_video2text():
    data = {"text" : "concreat content"}
    response = requests.post(f"{host}/video2text", json=data)
    return response.json().get('text')
    
def get_text_summarize():
    text = post_video2text()
    params = {"text" : text}
    response = requests.get(f"{host}/summarize", json=params)
    response_data = response.json()
    item = response_data.get('product_name')
    content = response_data.get('text')
    return item, content

def get_highlight():
    response = requests.get(f"{host}/highlight")
    response_data = response.json()
    data_list = response_data.get('data', [])
    return data_list

def post_chat(chat_type, chat_input):
    data = {
        "type":chat_type,
        "text":chat_input
    }
    response = requests.post(f"{host}/chat", json=data)
    response_data = response.json()
    return response_data.get('text')

def hostvoice(voice_text):
    data={
        "text": voice_text,
        "speed": 1.4
    }
    response = requests.post(f"{host}/hostvoice", json=data)
    return response.content

def toggle_update(input):
    output = input
    return output

def chat_display(message, history):
    if message=='':
        return '', history
    else:
        response_text = post_chat(0,message)
        history.append([message, response_text])
    return '', history

def chat_reset(history):
    history = []
    return history

def handdle_button_click(message, history, toggle_state):
    response_text = chat_display(message, history)
    if toggle_state == True:
        response_voice = hostvoice(history[-1][-1])
        return '', history, response_voice
    else:
        return '', history, None
    
def video_display(video):
    return video

def summarize_display():
    summarize = post_chat(1,'')
    return summarize

def highlight_graph():
    data_list = get_highlight()
    df = pd.DataFrame(data_list)
    df["x"] = df["time"]
    df["y"] = df["heart"] + df["chat_cnt"]
    return df
    
    
def play_streaming(video):
    summary = summarize_display()
    video_output = video_display(video)
    chatbot_msg = [(None, summary)]
    df = highlight_graph()
    return chatbot_msg, video_output, df

def upload_cancel():
    return None

def chat_before(input_text, history):
    response_text = post_chat(1,input_text)
    history.append([input_text, response_text])
    return "", history


with gr.Blocks(theme=gr.themes.Default()) as demo:
    with gr.Tab('이전 방송 요약'):
        with gr.Row():
            with gr.Column():
                video_uploader = gr.File(label='방송선택', file_types=['video'])
                streaming = gr.Video(label='방송보기',height=500)
                highlight_area = gr.LinePlot(
                    x="x",
                    y="y",
                    x_label="Time",
                    y_label="Pop",
                    height=300
                )
                with gr.Row():
                    clear_btn = gr.Button(value='다른방송보기')
                    play_btn = gr.Button(value='방송시청')
            with gr.Column():
                bef_sum_chat = gr.Chatbot(
                    height=550,
                    show_label=False
                    )
                with gr.Row():
                    chat_input = gr.Text(lines=1, container=False, label='입력', scale=3)
                    submit_btn = gr.Button(value='보내기', scale=1)

        play_btn.click(fn=play_streaming, inputs=video_uploader, outputs=[bef_sum_chat, streaming, highlight_area])
        clear_btn.click(fn=upload_cancel, inputs=None, outputs=video_uploader)
 
        submit_btn.click(fn=chat_before, inputs=[chat_input, bef_sum_chat], outputs=[chat_input, bef_sum_chat])
        chat_input.submit(fn=chat_before, inputs=[chat_input, bef_sum_chat], outputs=[chat_input, bef_sum_chat])
    
    with gr.Tab('AI 제품 추천 채팅봇'):
        title = gr.HTML("<h1><center>AI 제품 추천 챗봇</center></h1>")
   
        with gr.Row():
            with gr.Column():
                toggle = Toggle(
                    label="Host voice",
                    value=False,
                    info="호스트의 목소리를 들을 수 있습니다.",
                    interactive=True,
                )
        chatbot = gr.Chatbot()
        
        with gr.Row():
            chat_input = gr.Text(placeholder="어떤 제품이 궁금하신가요?")
        with gr.Row():
            voice = gr.Audio(visible=False, autoplay=True)
        
            clear_btn= gr.ClearButton(value='취소', components=[chat_input])
            reset_btn = gr.Button(value='초기화')
            run_btn = gr.Button(value='보내기')

            reset_btn.click(fn=chat_reset, inputs=[chatbot], outputs=[chatbot])
            run_btn.click(fn=handdle_button_click,inputs=[chat_input,chatbot,toggle], outputs=[chat_input, chatbot, voice]) 
            chat_input.submit(fn=handdle_button_click, inputs=[chat_input,chatbot,toggle], outputs=[chat_input,chatbot, voice])

demo.launch()