import os
import numpy as np
from moviepy import VideoFileClip
import whisper # OpenAI's Whisper for STT
import librosa # For audio analysis
import time

# --- Configuration ---
OUTPUT_AUDIO_FILENAME = "temp_audio.wav"

def extract_audio(video_path, audio_output_path):
    """Extracts audio from video file and saves as WAV."""
    print(f"Extracting audio from {video_path}...")
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        if audio_clip is None:
             print(f"Error: No audio track found in {video_path}")
             return False
        audio_clip.write_audiofile(audio_output_path, codec='pcm_s16le') # Standard WAV codec
        audio_clip.close()
        video_clip.close()
        print(f"Audio extracted successfully to {audio_output_path}")
        return True
    except Exception as e:
        print(f"Error extracting audio: {e}")
        # Clean up if partial file exists
        if os.path.exists(audio_output_path):
            os.remove(audio_output_path)
        return False
    
WHISPER_MODEL = "base" # Options: "tiny", "base", "small", "medium", "large". Larger = more accurate but slower/more resource intensive.

def transcribe_audio(audio_path):
    """
    Transcribes audio using Whisper, attempting to retain filler words
    by requesting word-level timestamps and reconstructing the transcript.
    """
    print(f"Loading Whisper model ({WHISPER_MODEL})...")
    try:
        # Load the model (consider doing this once outside the function if calling repeatedly)
        model = whisper.load_model(WHISPER_MODEL)
        print(f"Transcribing audio file: {audio_path} with word timestamps (this may take a while)...")

        # Key change: Set word_timestamps=True
        start_time = time.time()
        result = model.transcribe(audio_path, word_timestamps=True, fp16=False) # fp16=False might improve stability/accuracy on some systems
        end_time = time.time()
        print(f"Transcription complete in {end_time - start_time:.2f} seconds.")

        # Reconstruct the transcript from word segments
        # This ensures we capture words that might be filtered in the basic 'text' output
        full_transcript = ""
        if 'segments' in result:
            all_words = []
            for segment in result['segments']:
                if 'words' in segment:

                    for word_info in segment['words']: # type: ignore
                        # word_info is a dict like {'word': ' Hello', 'start': 0.0, 'end': 0.5, 'probability': 0.9}
                        # Note: Whisper often includes leading/trailing spaces in word_info['word']
                        all_words.append(word_info['word']) # type: ignore

            # Join the words carefully. Using strip() on each word and joining with a single space
            # handles cases where Whisper includes spaces and avoids double spacing.
            full_transcript = " ".join(word.strip() for word in all_words).strip()

            # Alternative simpler join (might have occasional extra spaces if whisper includes them):
            # full_transcript = "".join(word_info['word'] for segment in result['segments'] if 'words' in segment for word_info in segment['words']).strip()

        else:
            # Fallback if segments/words aren't available (shouldn't happen with word_timestamps=True)
            print("Warning: Word segments not found in Whisper result. Falling back to basic text.")
            full_transcript = result.get('text', "") # Use basic text if structure is unexpected

        if not full_transcript:
             print("Warning: Transcription resulted in empty text.")
             return "" # Return empty string instead of None for consistency downstream

        return full_transcript

    except Exception as e:
        print(f"Error during transcription with word timestamps: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None # Return None on error

def analyze_audio_features(audio_path, transcript):
    """Analyzes audio features like pace, pauses, pitch, volume."""
    print("Analyzing audio features...")
    try:
        y, sr = librosa.load(audio_path, sr=None) # Load audio with its original sample rate
        duration = librosa.get_duration(y=y, sr=sr)

        analysis_results = {}

        # 1. Pace (Words Per Minute)
        word_count = len(transcript.split())
        if duration > 0:
            wpm = int((word_count / duration) * 60)
            analysis_results['pace_wpm'] = wpm
            print(f"- Pace: {wpm} WPM")
        else:
             analysis_results['pace_wpm'] = 0
             print("- Pace: N/A (duration is zero)")


        # 2. Filler Words (Simple Count)
        # More sophisticated filler word detection is complex and often requires specific acoustic models
        fillers = ["um", "uh", "like", "you know", "so", "actually", "basically"]
        filler_count = sum(transcript.lower().count(f) for f in fillers)
        analysis_results['filler_count'] = filler_count
        print(f"- Filler Words Count (basic): {filler_count}")

        # 3. Volume Analysis (RMS Energy)
        rms = librosa.feature.rms(y=y)[0]
        avg_volume = np.mean(rms)
        std_volume = np.std(rms)
        analysis_results['avg_volume_rms'] = float(avg_volume)
        analysis_results['std_volume_rms'] = float(std_volume)
        print(f"- Average Volume (RMS): {avg_volume:.4f}")
        print(f"- Volume Variation (Std Dev RMS): {std_volume:.4f}")


        # 4. Pitch Analysis (Fundamental Frequency - F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')) # type: ignore
        valid_f0 = f0[voiced_flag] # Consider only voiced segments for pitch stats
        if len(valid_f0) > 0:
            avg_pitch = np.mean(valid_f0)
            std_pitch = np.std(valid_f0)
            analysis_results['avg_pitch_hz'] = float(avg_pitch)
            analysis_results['std_pitch_hz'] = float(std_pitch)
            print(f"- Average Pitch (F0): {avg_pitch:.2f} Hz")
            print(f"- Pitch Variation (Std Dev F0): {std_pitch:.2f} Hz")
        else:
            analysis_results['avg_pitch_hz'] = 0
            analysis_results['std_pitch_hz'] = 0
            print("- Pitch: Could not reliably detect pitch.")


        # 5. Pause Analysis (Simple Silence Detection)
        # Use librosa's split based on RMS energy threshold
        # top_db=40 means consider anything 40dB below the max RMS as silence
        non_silent_intervals = librosa.effects.split(y, top_db=40)
        pauses = []
        last_end = 0
        for start, end in non_silent_intervals:
            pause_duration = (start / sr) - (last_end / sr)
            if pause_duration > 0.2: # Consider pauses longer than 200ms
                pauses.append(pause_duration)
            last_end = end
        # Check pause after last segment until end of audio
        final_pause = duration - (last_end / sr)
        if final_pause > 0.2:
             pauses.append(final_pause)

        analysis_results['num_pauses'] = len(pauses)
        analysis_results['avg_pause_duration_s'] = float(np.mean(pauses)) if pauses else 0
        print(f"- Number of Pauses (>0.2s): {len(pauses)}")
        if pauses:
            print(f"- Average Pause Duration: {np.mean(pauses):.2f} s")

        return analysis_results

    except Exception as e:
        print(f"Error during audio feature analysis: {e}")
        return None