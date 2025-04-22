from google import genai
import json
import time
import os

# Gemini configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not set. Video analysis will not work.")

client = genai.Client(api_key=GEMINI_API_KEY)

def analyze_video_with_gemini(file_path):
    """Analyze video using Gemini Vision model."""
    try:
        if not GEMINI_API_KEY:
            return {"error": "Gemini API key not configured"}
        
        # For Gemini Pro Vision
        model = genai.GenerativeModel('gemini-pro-vision')
        
        # Send video file for analysis - need to open in binary mode
        with open(file_path, 'rb') as f:
            video_data = f.read()
        
        prompt = """
        Analyze this interview video. Please provide feedback on:
        1. Body language and posture
        2. Voice tone and clarity
        3. Confidence level
        4. Overall presentation
        Keep your response concise and professional.
        """
        
        response = model.generate_content([prompt, {"mime_type": "video/mp4", "data": video_data}])
        
        return {
            "analysis": response.text,
            "timestamp": time.time()
        }
    except google_exceptions.GoogleAPIError as e:
        return {"error": f"Gemini API error: {str(e)}"}
    except Exception as e:
        return {"error": f"Error analyzing video: {str(e)}"}