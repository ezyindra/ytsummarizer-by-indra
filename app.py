import requests
import gradio as gr

# CONFIG (we'll set these on Render)
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

def get_transcript(video_url):
    video_id = video_url.split("v=")[-1].split("&")[0]

    url = "https://youtube-transcript.p.rapidapi.com/youtube-transcript"
    querystring = {"id": video_id}

    headers = {
        "X-RapidAPI-Key": os.environ["RAPIDAPI_KEY"],
        "X-RapidAPI-Host": "youtube-transcript.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    if response.status_code != 200:
        return None

    data = response.json()
    transcript = " ".join([item["text"] for item in data])
    return transcript


def summarize(text):
    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}"
    }

    payload = {"inputs": text[:3000]}

    response = requests.post(HF_API_URL, headers=headers, json=payload)
    result = response.json()

    if isinstance(result, list):
        return result[0]["summary_text"]
    else:
        return "Error from HuggingFace API"


def process(video_url):
    transcript = get_transcript(video_url)
    if not transcript:
        return "Could not fetch transcript."

    summary = summarize(transcript)
    return summary


demo = gr.Interface(
    fn=process,
    inputs=gr.Textbox(label="Enter YouTube Video URL"),
    outputs=gr.Textbox(label="Video Summary"),
    title="YouTube Video Summarizer",
    description="Summarizes YouTube videos using Transcript API + HuggingFace AI"
)

demo.launch(server_name="0.0.0.0", server_port=8080)
