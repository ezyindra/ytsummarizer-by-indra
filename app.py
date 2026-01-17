import os
import requests
import gradio as gr

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

def get_transcript(video_url):
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]

        url = "https://youtube-transcript.p.rapidapi.com/youtube-transcript"
        querystring = {"id": video_id}

        headers = {
            "X-RapidAPI-Key": os.environ.get("RAPIDAPI_KEY"),
            "X-RapidAPI-Host": "youtube-transcript.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers, params=querystring, timeout=15)

        if response.status_code != 200:
            return None

        data = response.json()
        transcript = " ".join([item["text"] for item in data])
        return transcript

    except Exception as e:
        return None


def summarize(text):
    headers = {
        "Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"
    }

    payload = {"inputs": text[:3000]}

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)

    result = response.json()

    if isinstance(result, list):
        return result[0]["summary_text"]
    else:
        return "Error from Hugging Face API"


def process(video_url):
    if not video_url:
        return "Please enter a YouTube URL."

    transcript = get_transcript(video_url)

    if not transcript:
        return "Could not fetch transcript. This video may not have captions."

    summary = summarize(transcript)
    return summary


demo = gr.Interface(
    fn=process,
    inputs=gr.Textbox(label="Enter YouTube Video URL"),
    outputs=gr.Textbox(label="Video Summary", lines=10),
    title="YouTube Video Summarizer",
    description="Summarizes YouTube videos using Transcript API + Hugging Face AI"
)

demo.launch(server_name="0.0.0.0", server_port=8080)
