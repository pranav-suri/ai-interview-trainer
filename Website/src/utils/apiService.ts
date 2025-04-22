interface InterviewReturnedData {
    audio_metrics: any;
    video_metrics: any;
    interview_question: string;
    status: string;
    transcript: string;
    video_file: string;
    relevance_score: number;
    clarity_score: number;
    tone_score: number;
    vocabulary_score: number;
    star_format_score: number;
    strengths: string[];
    areas_for_improvement: string[];
}

interface InterviewResponse {
    audio_metrics: any;
    video_metrics: any;
    gemini_analysis: GeminiAnalysis;
    interview_question: string;
    status: string;
    transcript: string;
    video_file: string;
}

interface GeminiAnalysis {
    relevance_score: number;
    clarity_score: number;
    tone_score: number;
    vocabulary_score: number;
    star_format_score: number;
    strengths: string[];
    areas_for_improvement: string[];
}

/**
 * Sends recorded interview video and question to the processing API
 * @param videoBlob The recorded video blob
 * @param question The interview question text
 * @returns API response with scores and feedback
 */
export async function processInterviewVideo(
    videoBlob: Blob,
    question: string
): Promise<InterviewReturnedData> {
    try {
        // Create a form data object
        const formData = new FormData();
        formData.append("video", videoBlob, "interview-recording.webm");
        formData.append("interview_question", question);

        // Send the request to the API
        const response = await fetch("/api/process", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`API responded with status: ${response.status}`);
        }

        const jsonResponse = (await response.json()) as InterviewResponse;
        console.log(jsonResponse);
        const geminiAnalysis: GeminiAnalysis = jsonResponse.gemini_analysis;

        // Return the response in the desired format
        return {
            ...jsonResponse,
            ...geminiAnalysis,
        };
    } catch (error) {
        console.error("Error processing interview video:", error);
        throw error;
    }
}
