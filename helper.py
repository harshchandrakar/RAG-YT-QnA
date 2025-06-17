from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from dotenv import load_dotenv
import os

# Load environment variables for local development
load_dotenv()

# Function to get environment variables that works both locally and on Streamlit Cloud
def get_env_variable(key):
    """Get environment variable that works both locally and on Streamlit Cloud"""
    try:
        # Try to get from Streamlit secrets first (for deployed app)
        import streamlit as st
        return st.secrets.get(key)
    except:
        # Fallback to regular environment variables (for local development)
        return os.getenv(key)

def extract_youtube_video_id(url):
    parsed_url = urlparse(url)
    
    # Handle youtu.be URLs
    if parsed_url.netloc == 'youtu.be':
        return parsed_url.path[1:]  # Remove leading slash
    
    # Handle youtube.com URLs
    if 'youtube.com' in parsed_url.netloc:
        query_params = parse_qs(parsed_url.query)
        return query_params.get('v', [None])[0]
    
    return None

common_language_codes = {
    'en': 'English',
    'hi': 'Hindi',
    'es': 'Spanish', 
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'ml': 'Malayalam',
    'kn': 'Kannada',
    'gu': 'Gujarati',
    'pa': 'Punjabi',
    'mr': 'Marathi',
    'ur': 'Urdu'
}

def transcribe_extractor(url,language):
    video_id = extract_youtube_video_id(url)
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])

        transcript = " ".join(chunk["text"] for chunk in transcript_list)

        return transcript

    except TranscriptsDisabled:
        return "No captions available for this video."
    
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text