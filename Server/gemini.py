from google import genai
import json
import time
import os
from typing import Dict, Any, Optional, List, BinaryIO, Union

# Gemini configuration
GEMINI_API_KEY: str = os.environ.get(
    "GEMINI_API_KEY", "AIzaSyCNSuwC8afGO2DAFBLgFJUVsYorYGPH-3o"
)
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not set. Video analysis will not work.")

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY)


def analyze_video_with_gemini(
    file_path: str,
    interview_question: str,
    transcript: str,
    audio_metrics_str: str,
    video_metrics_str: str,
) -> Dict[str, Any]:
    """Analyze video using Gemini Vision model."""
    try:
        if not GEMINI_API_KEY:
            return {"error": "Gemini API key not configured"}

        # Use the latest Gemini model
        model = genai.GenerativeModel("gemini-pro-vision")

        # Send video file for analysis - need to open in binary mode
        with open(file_path, "rb") as f:
            video_data: bytes = f.read()

        prompt: str = """You are an expert AI Interview Coach. Your task is to analyze a candidate's response to an interview question based on the provided transcript and objective delivery metrics (audio/video analysis).

**Interview Context:**
*   **Question Asked:** "{interview_question}"

**Candidate's Response:**
*   **Transcript:** "{transcript}"

**Objective Delivery Analysis Data:**
{audio_metrics_str}
{video_metrics_str}

**Analysis Instructions:**
Evaluate the candidate's performance comprehensively based *only* on the provided data. Consider both the *content* of the answer (using the transcript) and the *delivery* (using the provided audio/video metrics).

*   **Content & Delivery Integration:** Assess the following aspects, assigning a score from 1 (Poor) to 5 (Excellent) for each:
    *   **Relevance:** How directly and effectively does the answer address the specific question asked? (1=Irrelevant, 5=Highly Relevant)
    *   **Clarity:** How clear, concise, and easy to understand is the response, considering both the language used and the delivery (e.g., pace, fillers)? (1=Unclear/Rambling, 5=Very Clear/Concise)
    *   **Tone:** How appropriate and effective is the perceived tone for an interview? Consider confidence, professionalism, and engagement, inferred from vocal cues (pitch/volume variation), facial expressions (dominant emotion), and language. (1=Inappropriate/Disengaged/Unconfident, 5=Confident/Professional/Engaging)
    *   **Vocabulary:** How appropriate, professional, and articulate is the language used? (1=Inappropriate/Informal/Unclear, 5=Highly Professional/Articulate)
    *   **STAR Format Adhesion:** If the question is behavioral, how well does the answer adhere to the STAR method (Situation, Task, Action, Result)? Are all components present and distinct? (1=No Adherence/Not Applicable, 3=Partial Adherence, 5=Excellent Adherence - All parts clear). Assign 1 if not a behavioral question or if format is totally absent.

**Output Format:**
Provide your analysis *strictly* in JSON format. The JSON object should have the following keys ONLY:

*   `relevance_score`: (Integer) Score from 1-5.
*   `clarity_score`: (Integer) Score from 1-5.
*   `tone_score`: (Integer) Score from 1-5 assessing perceived tone's effectiveness.
*   `vocabulary_score`: (Integer) Score from 1-5 assessing language use.
*   `star_format_score`: (Integer) Score from 1-5 assessing STAR method adhesion (1 if N/A or completely missing).
*   `strengths`: (List of strings) 2-3 bullet points highlighting specific positive aspects related to the scored criteria (e.g., "Strong relevance (Score: 5).", "Tone perceived as confident (Score: 4).").
*   `areas_for_improvement`: (List of strings) 2-4 specific, actionable feedback points related to the scored criteria, referencing metrics or transcript parts where possible (e.g., "Improve STAR adhesion (Score: 2) by explicitly stating the Result.", "Reduce filler word count (count: Y) to enhance clarity (Score: 3).", "Work on varying vocal pitch (Std Dev: Z Hz) to improve tone perception (Score: 2).").

**Example JSON Structure:**
```json
{{
  "relevance_score": 4,
  "clarity_score": 3,
  "tone_score": 4,
  "vocabulary_score": 5,
  "star_format_score": 3,
  "strengths": [
    "Excellent vocabulary use, very professional (Score: 5).",
    "Answer was highly relevant to the question asked (Score: 4).",
    "Tone came across as generally confident (Score: 4)."
  ],
  "areas_for_improvement": [
    "Improve clarity (Score: 3) by structuring points more logically and reducing minor rambling.",
    "STAR format adhesion was partial (Score: 3); ensure the 'Result' is clearly articulated.",
    "Consider increasing eye contact (estimated Z%) to further enhance engagement aspect of tone.",
    "Slightly high filler word count (count: Y) impacted clarity."
  ]
}}"""

        # Generate content with the updated API
        response = model.generate_content(
            [prompt, {"mime_type": "video/mp4", "data": video_data}]
        )

        return {"analysis": response.text, "timestamp": time.time()}
    except Exception as e:
        if "GoogleAPIError" in str(type(e)):
            return {"error": f"Gemini API error: {str(e)}"}
        return {"error": f"Error analyzing video: {str(e)}"}
