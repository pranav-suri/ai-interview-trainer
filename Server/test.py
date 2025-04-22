import requests
import os
import json
import time
from typing import Dict, Any, Optional

# Server URL - change this if your server is running on a different host/port
SERVER_URL = "http://localhost:5000/process"


def test_interview_video(video_path: str, question: str) -> Optional[Dict[str, Any]]:
    """
    Tests the /process route with a video file and interview question.

    Args:
        video_path: Path to the video file
        question: The interview question

    Returns:
        Dict containing the server response or None if an error occurred
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        return None

    print(f"Testing with video: {video_path}")
    print(f"Interview question: {question}")

    # Prepare the multipart form data
    files = {
        "video": (os.path.basename(video_path), open(video_path, "rb"), "video/mp4")
    }
    data = {"interview_question": question}

    try:
        # Send the POST request
        print("Sending request to server...")
        start_time = time.time()
        response = requests.post(SERVER_URL, files=files, data=data)
        end_time = time.time()

        print(f"Request completed in {end_time - start_time:.2f} seconds.")
        print(f"Status code: {response.status_code}")

        # Parse the response
        if response.status_code == 200:
            result = response.json()
            print("Success! Processing completed.")
            return result
        else:
            print(f"Error: {response.text}")
            return None

    except Exception as e:
        print(f"Error sending request: {e}")
        return None
    finally:
        # Make sure to close the file
        files["video"][1].close()


def pretty_print_results(results: Dict[str, Any]) -> None:
    """Prints the results in a readable format."""
    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 50)
    print(f"Results for video: {results.get('video_file', 'Unknown')}")
    print(f"Interview question: {results.get('interview_question', 'Unknown')}")
    print("=" * 50)

    # Print transcript
    if "transcript" in results:
        print("\nTranscript:")
        print("-" * 50)
        print(results["transcript"])
        print("-" * 50)

    # Print audio metrics
    if "audio_metrics" in results:
        print("\nAudio Metrics:")
        print("-" * 50)
        for key, value in results["audio_metrics"].items():
            print(f"- {key.replace('_', ' ').title()}: {value}")
        print("-" * 50)

    # Print video metrics
    if "video_metrics" in results:
        print("\nVideo Metrics:")
        print("-" * 50)
        for key, value in results["video_metrics"].items():
            if key == "emotion_distribution":
                print(f"- Emotion Distribution:")
                for emotion, count in value.items():
                    print(f"  - {emotion}: {count}")
            else:
                print(f"- {key.replace('_', ' ').title()}: {value}")
        print("-" * 50)

    # Print Gemini analysis if available
    if "gemini_analysis" in results:
        print("\nGemini Analysis:")
        print("-" * 50)
        if isinstance(results["gemini_analysis"], dict):
            # If it's a parsed JSON, print it nicely
            for key, value in results["gemini_analysis"].items():
                print(f"- {key}:")
                if isinstance(value, dict):
                    for k, v in value.items():
                        print(f"  - {k}: {v}")
                else:
                    print(f"  {value}")
        else:
            # If it's a string, print it as-is
            print(results["gemini_analysis"])
        print("-" * 50)

    # Print any errors
    if "error" in results:
        print("\nError:")
        print("-" * 50)
        print(results["error"])
        print("-" * 50)

    if "gemini_error" in results:
        print("\nGemini Error:")
        print("-" * 50)
        print(results["gemini_error"])
        print("-" * 50)


def main():
    """Main function to run tests."""
    # Define test cases
    test_cases = [
        {
            "video_path": "confident_interview.mp4",
            "question": "Tell me about yourself",
        },
        {
            "video_path": "nervous_interview.mp4",
            "question": "What is your greatest weakness?",
        },
    ]

    # Check if server is running
    try:
        requests.get("http://localhost:5000/")
        print("Server is running. Starting tests...")
    except requests.ConnectionError:
        print("ERROR: Server is not running. Please start the Flask server first.")
        print("Run: python Server/server.py")
        return

    # Run tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nRunning test {i}/{len(test_cases)}...")
        results = test_interview_video(test_case["video_path"], test_case["question"])

        if results:
            # Save results to file
            output_file = f"test_results_{os.path.basename(test_case['video_path']).split('.')[0]}.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")

            # Print results
            pretty_print_results(results)

        print("\n" + "=" * 50)

    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
