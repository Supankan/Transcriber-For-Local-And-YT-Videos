import streamlit as st
import moviepy.editor as mp
import librosa
import soundfile as sf
import requests
import json
import os
import time

# Replace with your Hugging Face API URL and token
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": f"Bearer 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"}


def extract_audio(video_path, audio_path):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)


def split_audio(audio_path, chunk_duration=30):
    y, sr = librosa.load(audio_path, sr=16000)
    chunk_size = chunk_duration * sr
    chunks = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]
    return chunks, sr


def convert_audio_chunk_to_text(audio_chunk, sr, max_retries=5, retry_delay=5):
    temp_audio_path = "files/temp_audio_chunk.wav"
    sf.write(temp_audio_path, audio_chunk, sr)

    with open(temp_audio_path, "rb") as f:
        data = f.read()

    retries = 0
    while retries < max_retries:
        response = requests.post(API_URL, headers=headers, data=data)

        if response.status_code == 200:
            try:
                result = json.loads(response.content.decode("utf-8"))
                return result["text"]
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Response content: {response.content}")
                return None
        elif response.status_code == 503:
            print(f"Error: Received status code 503 (Service Unavailable). Retrying in {retry_delay} seconds...")
            retries += 1
            time.sleep(retry_delay)
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response content: {response.content}")
            return None

    print("Max retries reached. Failed to transcribe the audio chunk.")
    return None


def save_output(text, output_path):
    with open(output_path, "w") as f:
        f.write(text)


def main():
    st.title("Video to Text Converter")

    if "transcribed_text" not in st.session_state:
        st.session_state.transcribed_text = None
    if "audio_path" not in st.session_state:
        st.session_state.audio_path = None

    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if video_file is not None:
        if st.button("Convert"):
            with st.spinner("Converting to audio..."):
                st.session_state.audio_path = "files/extracted_audio.wav"
                os.makedirs("files", exist_ok=True)
                with open("files/temp_video.mp4", "wb") as f:
                    f.write(video_file.getbuffer())
                extract_audio("files/temp_video.mp4", st.session_state.audio_path)
                st.success("✅ Converted to audio")

            st.write("Converting to text...")
            progress_bar = st.progress(0)
            chunks, sr = split_audio(st.session_state.audio_path)
            transcribed_text = ""

            for i, chunk in enumerate(chunks):
                progress_bar.progress((i + 1) / len(chunks))
                text = convert_audio_chunk_to_text(chunk, sr)
                if text:
                    transcribed_text += text + " "

            st.success("✅ Converted to text")

            output_path = "files/transcribed_text.txt"
            save_output(transcribed_text, output_path)
            st.session_state.transcribed_text = transcribed_text

    if st.session_state.transcribed_text is not None:
        st.subheader("Transcribed Text")
        with st.expander("Show Text"):
            st.write(st.session_state.transcribed_text)

        st.subheader("Extracted Audio")
        st.audio(st.session_state.audio_path)

        st.subheader("Download")
        with open("files/transcribed_text.txt", "rb") as f:
            st.download_button("Download Text", f, file_name="transcribed_text.txt")
        with open(st.session_state.audio_path, "rb") as f:
            st.download_button("Download Audio", f, file_name="extracted_audio.wav")

    if st.button("Reset"):
        st.session_state.transcribed_text = None
        st.session_state.audio_path = None
        st.rerun()


if __name__ == "__main__":
    main()
