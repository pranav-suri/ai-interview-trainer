from audio import OUTPUT_AUDIO_FILENAME, analyze_audio_features, extract_audio, transcribe_audio
from video import analyze_video_features
import os


args = {'video_file': 'confident_interview.mp4', 'question': 'Tell me about yourself'}

print("-" * 30)
print("Starting Interview Analysis")
print(f"Video File: {args['question']}")
print(f"Interview Question: {args['video_file']}")
print("-" * 30)
# 1. Extract Audio
if not extract_audio(args['video_file'], OUTPUT_AUDIO_FILENAME):
    print("Analysis aborted due to audio extraction failure.")
    exit(1)

# 2. Transcribe Audio
transcript = transcribe_audio(OUTPUT_AUDIO_FILENAME)
if transcript is None:
    print("Analysis aborted due to transcription failure.")
        # Clean up audio file before exiting
    if os.path.exists(OUTPUT_AUDIO_FILENAME):
        os.remove(OUTPUT_AUDIO_FILENAME)
    exit(1)

print("\n--- Transcription ---")
print(transcript)
print("-" * 20)

# 3. Analyze Audio Features
print("\n--- Audio Delivery Analysis ---")
audio_metrics = analyze_audio_features(OUTPUT_AUDIO_FILENAME, transcript)
if audio_metrics is None:
    print("Could not perform detailed audio analysis.") 
elif audio_metrics:
    for key, value in audio_metrics.items():
        print(f"- {key.replace('_', ' ').title()}: {value}")
print("-" * 20)

# 4. Analyze Video Features
print("\n--- Video Delivery Analysis ---")
video_metrics = analyze_video_features(args['video_file'])
if video_metrics is None:
    print("Could not perform video analysis.")
elif video_metrics and 'error' not in video_metrics :
    for key, value in video_metrics.items():
            print(f"- {key.replace('_', ' ').title()}: {value}")
elif video_metrics and 'error' in video_metrics:
        print(f"- Error: {video_metrics['error']}")
else:
    print("N/A")
print("-" * 20)

# 5. Clean up temporary audio file
if os.path.exists(OUTPUT_AUDIO_FILENAME):
    try:
        os.remove(OUTPUT_AUDIO_FILENAME)
        print(f"\nTemporary audio file {OUTPUT_AUDIO_FILENAME} deleted.")
    except OSError as e:
        print(f"\nWarning: Could not delete temporary audio file {OUTPUT_AUDIO_FILENAME}: {e}")

print("\n--- Analysis Summary ---")
print(f"Question: {args['question']}")
print(f"Video File: {args['video_file']}")
print("\nTranscript:")
print(transcript if transcript else "N/A")
print("\nAudio Metrics:")
if audio_metrics:
    for key, value in audio_metrics.items():
        print(f"- {key.replace('_', ' ').title()}: {value}")
else:
    print("N/A")
print("\nVideo Metrics:")
if video_metrics and 'error' not in video_metrics :
    for key, value in video_metrics.items():
            print(f"- {key.replace('_', ' ').title()}: {value}")
elif video_metrics and 'error' in video_metrics:
        print(f"- Error: {video_metrics['error']}")
else:
    print("N/A")
print("-" * 30)
print("Analysis Complete.")

# 5. Clean up temporary audio file
if os.path.exists(OUTPUT_AUDIO_FILENAME):
    try:
        os.remove(OUTPUT_AUDIO_FILENAME)
        print(f"\nTemporary audio file {OUTPUT_AUDIO_FILENAME} deleted.")
    except OSError as e:
        print(f"\nWarning: Could not delete temporary audio file {OUTPUT_AUDIO_FILENAME}: {e}")

# 6. Print Summary
print("\n--- Analysis Summary ---")
print(f"Question: {args['question']}")
print(f"Video File: {args['video_file']}")
print("\nTranscript:")
print(transcript if transcript else "N/A")
print("\nAudio Metrics:")
if audio_metrics:
    for key, value in audio_metrics.items():
        print(f"- {key.replace('_', ' ').title()}: {value}")
else:
    print("N/A")
print("\nVideo Metrics:")
if video_metrics and 'error' not in video_metrics :
    for key, value in video_metrics.items():
            print(f"- {key.replace('_', ' ').title()}: {value}")
elif video_metrics and 'error' in video_metrics:
        print(f"- Error: {video_metrics['error']}")
else:
    print("N/A")
print("-" * 30)
print("Analysis Complete.")