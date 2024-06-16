import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import docx2txt
import gradio as gr
import json
import shutil

folder = './TTS_frelance'
model_file = 'XTTS-v2'
# Initialize the TTS model with specified configuration and checkpoint
def initialize_model():
    print("Loading model...")
    config = XttsConfig()
    config.load_json(f"{model_file}/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=f"{model_file}/model.pth", use_deepspeed=True)
    model.cuda()
    return model

MODEL = initialize_model()

# Generate voiceover for a given text
def generate_voiceover(model, text, voice_file, language, n, name):
    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[voice_file])
    
    print("Inference...")
    out = model.inference(
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7
    )

    output_folder = f"{folder}/Result/{os.path.splitext(name)[0]}"
    os.makedirs(output_folder, exist_ok=True)
    file_path_wav = f"{output_folder}/{n}.wav"
    file_path_mp3 = f"{output_folder}/{n}.mp3"
    
    torchaudio.save(file_path_wav, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    
    # Convert WAV to MP3
    os.system(f"ffmpeg -i {file_path_wav} {file_path_mp3}")
    os.remove(file_path_wav)  # Remove the WAV file after conversion

    return file_path_mp3

# Process the uploaded file and generate audio files
def process_file(file, voice, progress=gr.Progress()):
    if file is None or voice is None:
        return None

    file.name = file.name.replace(" ", "_")
    file_type = file.name.split(".")[-1]

    if file_type == "txt":
        with open(file) as file_:
            text = file_.read()
    elif file_type == "docx":
        text = docx2txt.process(file)
    else:
        return None

    if not text.strip():
        return None

    paragraphs = text.split("\n\n")
    print(paragraphs)

    audio_outputs = []
    n = 1
    
    voice_file = f'{folder}/voices/{voice}.mp3'
    language = config_settings["voices"][f'{voice}.mp3']

    for paragraph in paragraphs:
        if paragraph.strip():
            audio_placeholder = generate_voiceover(MODEL, paragraph, voice_file, language, n, file.name)
            audio_outputs.append(audio_placeholder)
            progress(1 / len(paragraphs))
            n += 1

    # Create a zip of all audio files
    output_zip = f"{folder}/Result/result.zip"
    shutil.make_archive(output_zip.replace('.zip', ''), 'zip', f"{folder}/Result/{os.path.splitext(file.name)[0]}")

    # Return a zip of all audio files
    return output_zip

# Main function to launch the Gradio interface
def main():
    gr.Interface(
        fn=process_file,
        live=True,
        inputs=[
            gr.File(file_types=['.txt', '.docx'], label="Upload a file"),
            gr.Dropdown(voices, label="Select a language")
        ],
        outputs=[
            gr.File(label="Generated Audio Files"),
        ],
        title="Text to Speech App",
        description="Upload a .txt or .docx file, select a language, and generate voiceovers for the content."
    ).launch(debug=True, share=True)

if __name__ == "__main__":
    main()
