from flask import Flask, request, jsonify
import os
import json
from typing import Dict, List, Union, Optional, Any, Tuple
from werkzeug.utils import secure_filename

from gemini import GEMINI_API_KEY, analyze_video_with_gemini
from audio import (
    OUTPUT_AUDIO_FILENAME,
    analyze_audio_features,
    extract_audio,
    transcribe_audio,
)
from video import analyze_video_features

app: Flask = Flask(__name__)

# Configuration
UPLOAD_FOLDER: str = "uploads"
ALLOWED_EXTENSIONS: set = {"mp4", "avi", "mov", "wmv", "mkv"}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max-limit

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/process", methods=["POST"])
def process_video_complete() -> Tuple[Any, int]:
    """
    Process a video file completely:
    1. Extract audio
    2. Transcribe audio
    3. Analyze audio features
    4. Analyze video features
    5. Send to Gemini for complete analysis

    Returns:
        Tuple[Any, int]: JSON response and HTTP status code
    """
    # Check if the post request has the video part
    if 'video' not in request.files:
        return jsonify({'error': 'No video part in the request'}), 400

    # Check if interview_question is provided
    interview_question: str = request.form.get("interview_question", "")
    if not interview_question:
        return jsonify({"error": "No interview_question provided"}), 400

    file = request.files['video']

    # If user does not select file, browser might submit an empty file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        if not file.filename:
            return jsonify({'error': 'Filename missing'}), 400

        # Save uploaded video
        filename: str = secure_filename(file.filename)
        file_path: str = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        results: Dict[str, Any] = {
            "video_file": filename,
            "interview_question": interview_question,
            "status": "processing",
        }

        try:
            # 1. Extract Audio
            if not extract_audio(file_path, OUTPUT_AUDIO_FILENAME):
                return (
                    jsonify(
                        {"error": "Audio extraction failed", "video_file": filename}
                    ),
                    500,
                )

            # 2. Transcribe Audio
            transcript: Optional[str] = transcribe_audio(OUTPUT_AUDIO_FILENAME)
            if transcript is None:
                # Clean up audio file before returning error
                if os.path.exists(OUTPUT_AUDIO_FILENAME):
                    os.remove(OUTPUT_AUDIO_FILENAME)
                return (
                    jsonify({"error": "Transcription failed", "video_file": filename}),
                    500,
                )

            results["transcript"] = transcript

            # 3. Analyze Audio Features
            audio_metrics: Optional[Dict[str, Any]] = analyze_audio_features(
                OUTPUT_AUDIO_FILENAME, transcript
            )
            if audio_metrics:
                results["audio_metrics"] = audio_metrics

            # 4. Analyze Video Features
            video_metrics: Optional[Dict[str, Any]] = analyze_video_features(file_path)
            if video_metrics and "error" not in video_metrics:
                results["video_metrics"] = video_metrics

            # 5. Format metrics for Gemini analysis
            audio_metrics_str: str = ""
            if audio_metrics:
                audio_metrics_str = "Audio Analysis:\n"
                for key, value in audio_metrics.items():
                    audio_metrics_str += f"- {key.replace('_', ' ').title()}: {value}\n"

            video_metrics_str: str = ""
            if video_metrics and "error" not in video_metrics:
                video_metrics_str = "Video Analysis:\n"
                for key, value in video_metrics.items():
                    if key == "emotion_distribution":
                        video_metrics_str += f"- {key.replace('_', ' ').title()}:\n"
                        for emotion, count in value.items():
                            video_metrics_str += f"  - {emotion}: {count}\n"
                    else:
                        video_metrics_str += (
                            f"- {key.replace('_', ' ').title()}: {value}\n"
                        )

            # 6. Send to Gemini for analysis if API key is configured
            if GEMINI_API_KEY:
                gemini_result: Dict[str, Any] = analyze_video_with_gemini(
                    file_path,
                    interview_question,
                    transcript,
                    audio_metrics_str,
                    video_metrics_str,
                )

                if "analysis" in gemini_result:
                    # Try to parse the JSON response from Gemini
                    try:
                        # Find JSON within the text if it exists
                        analysis_text: str = gemini_result["analysis"]
                        json_start: int = analysis_text.find("{")
                        json_end: int = analysis_text.rfind("}") + 1

                        if json_start >= 0 and json_end > json_start:
                            json_str: str = analysis_text[json_start:json_end]
                            gemini_analysis: Dict[str, Any] = json.loads(json_str)
                            results["gemini_analysis"] = gemini_analysis
                        else:
                            results["gemini_analysis"] = analysis_text
                    except json.JSONDecodeError:
                        # If parsing fails, use the raw text
                        results["gemini_analysis"] = gemini_result["analysis"]
                else:
                    results["gemini_error"] = gemini_result.get(
                        "error", "Unknown Gemini error"
                    )
            else:
                results["gemini_error"] = "Gemini API key not configured"

            # 7. Clean up temporary audio file
            if os.path.exists(OUTPUT_AUDIO_FILENAME):
                try:
                    os.remove(OUTPUT_AUDIO_FILENAME)
                except OSError:
                    pass  # Ignore deletion errors

            results["status"] = "complete"
            return jsonify(results), 200

        except Exception as e:
            # Clean up temporary files in case of error
            if os.path.exists(OUTPUT_AUDIO_FILENAME):
                try:
                    os.remove(OUTPUT_AUDIO_FILENAME)
                except OSError:
                    pass

            return (
                jsonify(
                    {
                        "error": f"Error processing video: {str(e)}",
                        "video_file": filename,
                    }
                ),
                500,
            )
    else:
        return jsonify({'error': 'File type not allowed'}), 400


@app.route('/', methods=['GET'])
def index() -> str:
    return """
    <!doctype html>
    <title>Upload Video File</title>
    <h1>Upload Video File</h1>
    <form method=post action=/process enctype=multipart/form-data>
      <input type=file name=video>
      <input type=text name=interview_question placeholder="Interview Question">
      <input type=submit value=Process>
    </form>
    """


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
