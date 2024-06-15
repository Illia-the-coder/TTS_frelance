import torch
from TTS.api import TTS
import os
import docx2txt
import gradio as gr
import json

folder = './TTS_frelance'


os.environ["COQUI_TOS_AGREED"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1").to(device)




with open(f"{folder}/config.json") as f:
    config_settings = json.load(f)

# all fileanmes in voices folder
voices = [f.name.split(".")[0] for f in os.scandir(f"{folder}/voices") if f.is_file()]
voices_in_config = [f.name for f in os.scandir(f"{folder}/voices") if f.is_file()]

if set(voices_in_config) != set(config_settings["voices"].keys()):
    # print which file makes voices not in sync
    print(set(voices_in_config).difference(config_settings["voices"].keys()))
    print("You should update https://github.com/Illia-the-coder/TTS_frelance/blob/main/config.json and voices folder")
    raise ValueError("config.json and voices folder are not in sync!")  


def generate_voiceover(text, voice, n, name):
    if not os.path.exists(f"{folder}/Result/{voice}/{name}"):
        os.makedirs(f"{folder}/Result/{voice}/{name}")
    file_path = f"{folder}/Result/{voice}/{name}/{n}.mp3"

        tts.tts_to_file(text=text,  speaker_wav=f'{folder}/voices/{voice}.mp3', language=config_settings["voices"][f'{voice}.mp3'], file_path= file_path)

    # speed up for 1.2
    # os.system("play " +file_path+" tempo {}".format(config_settings["speed"]))

    return file_path



def process_file(file, voice, progress=gr.Progress()):
    if file is None:
        return None
    if voice is None:
        return None
    file.name = file.name.replace(" ", "_")

    file_type = file.name.split(".")[-1]

    if file_type == "txt":
        with open(file) as file_:
          text = file_.read()
    elif file_type == "docx":
        text  = docx2txt.process(file)
    else:
        return None

    if not text.strip():
        return None

    paragraphs = text.split("\n\n")
    print(paragraphs)
    audio_outputs = []
    n=1
    for paragraph in paragraphs:
        if paragraph.strip():
            audio_placeholder = generate_voiceover(paragraph, voice, n, file.name.split("/")[-1].split(".")[0])
            audio_outputs.append(audio_placeholder)
            progress(1/len(paragraphs))
            n+=1


    # create zip of all audio files
    os.system(f"zip -r {folder}/Result/result.zip {folder}/Result/{voice}/{file.name.split('/')[-1].split('.')[0]}")

    # return a zip of all audio files
    return f"{folder}/Result/result.zip"



def main():
    gr.Interface(
        fn=process_file,
        live=True,
        inputs=[
            gr.File(file_types=['.txt', '.docx'], label="Upload a file"),
            gr.Dropdown(voices, label="Select a language")
        ],
        outputs=[
            # return a zip of all audio files
            gr.File(label="Generated Audio Files"),
        ],
        title="Text to Speech App",
        description="Upload a .txt or .docx file, select a language, and generate voiceovers for the content."
    ).launch(debug=True, share=True)

if __name__ == "__main__":
    main()