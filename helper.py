from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv
import os
import requests
import re
import time
import random

# Load environment variables for local development
load_dotenv()

def get_env_variable(key):
    """Get environment variable that works both locally and on Streamlit Cloud"""
    try:
        import streamlit as st
        return st.secrets.get(key)
    except:
        return os.getenv(key)

def extract_youtube_video_id(url):
    parsed_url = urlparse(url)
    
    if parsed_url.netloc == 'youtu.be':
        return parsed_url.path[1:]
    
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

def get_available_transcripts(video_id):
    """Get list of available transcript languages for a video"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available_languages = []
        
        for transcript in transcript_list:
            available_languages.append(transcript.language_code)
            
        return available_languages
    except Exception as e:
        return []

def transcribe_with_fallback_languages(video_id, preferred_language):
    """Try to get transcript with fallback to other available languages"""
    try:
        # First, get available languages
        available_languages = get_available_transcripts(video_id)
        
        if not available_languages:
            raise Exception("No transcripts available for this video")
        
        # Try preferred language first
        if preferred_language in available_languages:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[preferred_language])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)
            return transcript, preferred_language
        
        # Try English if not preferred and available
        if 'en' in available_languages and preferred_language != 'en':
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)
            return transcript, 'en'
        
        # Try any available language
        for lang in available_languages:
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                transcript = " ".join(chunk["text"] for chunk in transcript_list)
                return transcript, lang
            except:
                continue
        
        raise Exception("Could not retrieve transcript in any available language")
        
    except Exception as e:
        raise Exception(f"Transcript extraction failed: {str(e)}")

def scrape_youtube_captions(video_id, language='en'):
    """
    Alternative method to scrape captions using a different approach
    This uses a public endpoint that doesn't require authentication
    """
    try:
        # Add random delay to avoid rate limiting
        time.sleep(random.uniform(0.5, 2.0))
        
        # Custom headers to mimic a regular browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Try to get the video page first
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        response = requests.get(video_url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"Could not access video page: {response.status_code}")
        
        html_content = response.text
        
        # Look for caption tracks in the HTML
        caption_pattern = r'"captionTracks":\[(.*?)\]'
        caption_match = re.search(caption_pattern, html_content)
        
        if not caption_match:
            raise Exception("No caption tracks found in video page")
        
        caption_data = caption_match.group(1)
        
        # Extract caption URL
        url_pattern = r'"baseUrl":"(.*?)"'
        url_matches = re.findall(url_pattern, caption_data)
        
        if not url_matches:
            raise Exception("No caption URLs found")
        
        # Try the first caption URL
        caption_url = url_matches[0].replace('\\u0026', '&').replace('\\', '')
        
        # Download the caption
        caption_response = requests.get(caption_url, headers=headers, timeout=10)
        
        if caption_response.status_code != 200:
            raise Exception(f"Could not download captions: {caption_response.status_code}")
        
        # Parse the XML content
        caption_xml = caption_response.text
        transcript = parse_youtube_xml_captions(caption_xml)
        
        return transcript
        
    except Exception as e:
        raise Exception(f"Caption scraping failed: {str(e)}")

def parse_youtube_xml_captions(xml_content):
    """Parse YouTube's XML caption format"""
    import xml.etree.ElementTree as ET
    
    try:
        root = ET.fromstring(xml_content)
        texts = []
        
        for text_elem in root.findall('.//text'):
            if text_elem.text:
                # Clean up the text
                clean_text = text_elem.text.strip()
                # Remove HTML entities
                clean_text = clean_text.replace('&amp;', '&')
                clean_text = clean_text.replace('&lt;', '<')
                clean_text = clean_text.replace('&gt;', '>')
                clean_text = clean_text.replace('&quot;', '"')
                clean_text = clean_text.replace('&#39;', "'")
                
                if clean_text:
                    texts.append(clean_text)
        
        return ' '.join(texts)
        
    except Exception as e:
        raise Exception(f"XML parsing failed: {str(e)}")

def transcribe_extractor(url, language):
    video_id = extract_youtube_video_id(url)
    
    if not video_id:
        raise Exception("Invalid YouTube URL")
    
    print(f"Attempting to extract transcript for video ID: {video_id}")
    
    # Method 1: Try youtube-transcript-api with fallback languages
    try:
        print("Trying youtube-transcript-api with fallback languages...")
        transcript, used_language = transcribe_with_fallback_languages(video_id, language)
        
        if used_language != language:
            print(f"Note: Transcript retrieved in '{used_language}' instead of requested '{language}'")
        
        return transcript
        
    except Exception as api_error:
        print(f"youtube-transcript-api failed: {api_error}")
        
        # Method 2: Try caption scraping
        try:
            print("Trying caption scraping method...")
            return scrape_youtube_captions(video_id, language)
            
        except Exception as scrape_error:
            print(f"Caption scraping failed: {scrape_error}")
            
            # Method 3: Final fallback - try original API with different approach
            try:
                print("Trying final fallback with any available transcript...")
                # Get any available transcript without language specification
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript = " ".join(chunk["text"] for chunk in transcript_list)
                return transcript
                
            except Exception as final_error:
                # Provide detailed error information
                available_langs = get_available_transcripts(video_id)
                if available_langs:
                    raise Exception(f"All methods failed. Available languages for this video: {available_langs}. "
                                  f"API error: {api_error}. Scraping error: {scrape_error}. Final error: {final_error}")
                else:
                    raise Exception("No captions/transcripts are available for this video. "
                                  "The video either has no captions or they are disabled.")

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text