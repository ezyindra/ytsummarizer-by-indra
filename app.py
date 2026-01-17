import re
import subprocess
import requests
import gradio as gr

HF_API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"


def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None


def get_transcript(video_url):
    try:
        command = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "-o", "subtitle.%(ext)s",
            video_url
        ]

        subprocess.run(command, check=True, capture_output=True)

        # Read generated subtitle file
        with open("subtitle.en.vtt", "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Remove timestamps and metadata
        text_lines = []
        for line in lines:
            line = line.strip()
            if not line or "-->" in line or line.startswith("WEBVTT"):
                continue
            text_lines.append(line)

        return " ".join(text_lines), None

    except Exception as e:
        return None, str(e)


def summarize_text(text):
    payload = {"inputs": text}
    response = requests.post(HF_API_URL, json=payload, timeout=120)

    if response.status_code != 200:
        return f"Hugging Face API error: {response.text}"

    result = response.json()
    return result[0]["summary_text"]


def summarize_youtube(url):
    transcript, error = get_transcript(url)
    if error:
        return f"Transcript error: {error}"

    return summarize_text(transcript)


demo = gr.Interface(
    fn=summarize_youtube,
    inputs=gr.Textbox(label="Enter YouTube Video URL"),
    outputs=gr.Textbox(label="Video Summary"),
    title="YouTube Video Summarizer",
    description="Summarizes YouTube videos using yt-dlp and Hugging Face AI."
)

demo.launch(server_name="0.0.0.0", server_port=7860)
