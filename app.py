import re
import gradio as gr
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter


# Load summarizer (CPU)
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1
)

api = YouTubeTranscriptApi()


def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None


def chunk_text(text, max_words=400):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])


def summarize_text(text):
    summaries = []

    for chunk in chunk_text(text):
        result = summarizer(chunk, max_length=130, min_length=50, do_sample=False)
        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)


def get_youtube_summary(url):
    video_id = extract_video_id(url)

    if not video_id:
        return "❌ Invalid YouTube URL"

    try:
        transcript = api.fetch(video_id)

        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript)

        if len(transcript_text.split()) < 50:
            return "❌ Transcript too short to summarize."

        return summarize_text(transcript_text)

    except Exception as e:
        return f"❌ Error: {e}"


gr.close_all()

demo = gr.Interface(
    fn=get_youtube_summary,
    inputs=gr.Textbox(label="Input YouTube URL to summarize"),
    outputs=gr.Textbox(label="Summarized text", lines=8),
    title="Indrajeet Project 2: YouTube Script Summarizer",
    description="THIS APPLICATION WILL BE USED TO SUMMARIZE THE YOUTUBE VIDEO SCRIPT."
)

demo.launch()
