// Website\src\utils\api.ts

export interface BackendAnalysisResult {
    relevance_score: number; // (1-5)
    clarity_score: number; // (1-5)
    tone_score: number; // (1-5)
    vocabulary_score: number; // (1-5)
    star_format_score: number; // (1-5)
    strengths: string[];
    areas_for_improvement: string[];
}

const API_URL = "/api/process";

export const sendVideoToServer = async (
    videoBlob: Blob,
    question: string
): Promise<BackendAnalysisResult> => {
    const formData = new FormData();
    // Use a timestamp or unique ID for the filename if needed, otherwise a generic name is fine
    formData.append(
        "video",
        videoBlob,
        `interview_response_${Date.now()}.webm`
    );
    formData.append("interview_question", question);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData,
            // Add headers if required by your backend, e.g., Authorization
            // headers: { 'Authorization': 'Bearer YOUR_TOKEN' },
        });

        if (!response.ok) {

            const errorText = await response.text();
            throw new Error(
                `API Error (${response.status}): ${
                    errorText || response.statusText
                }`
            );
        }

        const result: BackendAnalysisResult = await response.json();
        return result;
    } catch (error) {
        console.error("Error sending video to server:", error);
  
        // Re-throw the error so the caller can handle it
        throw error;
    }
};
