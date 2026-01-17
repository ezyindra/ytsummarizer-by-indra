import re
import requests
import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi

# Hugging Face Inference API endpoint (same model you used before)
HF_API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"


def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None


def get_transcript(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return None, "Invalid YouTube URL"

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([item["text"] for item in transcript])
        return full_text, None
    except Exception as e:
        return None, str(e)


def summarize_text(text):
    payload = {"inputs": text}

    response = requests.post(HF_API_URL, json=payload, timeout=120)

    if response.status_code != 200:
        return f"Error from Hugging Face API: {response.text}"

    result = response.json()
    if isinstance(result, list) and "summary_text" in result[0]:
        return result[0]["summary_text"]

    return "Failed to generate summary."


def summarize_youtube(video_url):
    transcript, error = get_transcript(video_url)
    if error:
        return f"Error fetching transcript: {error}"

    return summarize_text(transcript)


demo = gr.Interface(
    fn=summarize_youtube,
    inputs=gr.Textbox(label="Enter YouTube Video URL"),
    outputs=gr.Textbox(label="Video Summary"),
    title="YouTube Video Summarizer",
    description="Paste a YouTube URL to get a summarized version of the video transcript."
)

demo.launch(server_name="0.0.0.0", server_port=7860)
