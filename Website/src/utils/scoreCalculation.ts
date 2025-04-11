
import { FaceAnalysisResult } from './faceAnalysis';
import { VoiceAnalysisResult } from './voiceAnalysis';

export interface InterviewScore {
  overall: number;
  confidence: number;
  engagement: number;
  communication: number;
  bodyLanguage: number;
  eyeContact: number;
  feedback: string[];
}

export const calculateInterviewScore = (
  faceResults: FaceAnalysisResult,
  voiceResults: VoiceAnalysisResult
): InterviewScore => {
  // Calculate individual scores
  const confidenceScore = (faceResults.confidence * 0.6) + (voiceResults.toneConfidence * 0.4);
  const engagementScore = (faceResults.engagement * 0.7) + (voiceResults.volume * 0.3);
  const communicationScore = (voiceResults.clarity * 0.5) + 
                           (voiceResults.speakingPace * 0.3) + 
                           ((1 - voiceResults.fillerWordPercentage * 10) * 0.2);
  const bodyLanguageScore = faceResults.confidence * 0.8;
  const eyeContactScore = faceResults.eyeContact;
  
  // Calculate overall score (weighted average)
  const overall = (
    confidenceScore * 0.25 +
    engagementScore * 0.2 +
    communicationScore * 0.25 +
    bodyLanguageScore * 0.15 +
    eyeContactScore * 0.15
  );
  
  // Generate feedback based on scores
  const feedback = generateFeedback(
    confidenceScore,
    engagementScore,
    communicationScore,
    bodyLanguageScore,
    eyeContactScore,
    voiceResults.fillerWordCount
  );
  
  return {
    overall: Number((overall * 100).toFixed(1)),
    confidence: Number((confidenceScore * 100).toFixed(1)),
    engagement: Number((engagementScore * 100).toFixed(1)),
    communication: Number((communicationScore * 100).toFixed(1)),
    bodyLanguage: Number((bodyLanguageScore * 100).toFixed(1)),
    eyeContact: Number((eyeContactScore * 100).toFixed(1)),
    feedback
  };
};

const generateFeedback = (
  confidence: number,
  engagement: number,
  communication: number,
  bodyLanguage: number,
  eyeContact: number,
  fillerWordCount: number
): string[] => {
  const feedback: string[] = [];
  
  // Confidence feedback
  if (confidence < 0.6) {
    feedback.push("Work on your confidence by practicing power poses before interviews and speaking with a more authoritative tone.");
  } else if (confidence >= 0.8) {
    feedback.push("Great job showing confidence! Your self-assured presence is a strong point.");
  }
  
  // Engagement feedback
  if (engagement < 0.6) {
    feedback.push("Try to appear more engaged by showing enthusiasm in your voice and facial expressions.");
  } else if (engagement >= 0.8) {
    feedback.push("You demonstrated excellent engagement and enthusiasm throughout the interview.");
  }
  
  // Communication feedback
  if (communication < 0.6) {
    feedback.push("Focus on clearer communication by speaking at a moderate pace and eliminating filler words.");
  } else if (communication >= 0.8) {
    feedback.push("Your communication skills are strong, with clear articulation and good pacing.");
  }
  
  // Body language feedback
  if (bodyLanguage < 0.6) {
    feedback.push("Be mindful of your body language by sitting up straight and using appropriate hand gestures.");
  } else if (bodyLanguage >= 0.8) {
    feedback.push("Your body language effectively communicated openness and professionalism.");
  }
  
  // Eye contact feedback
  if (eyeContact < 0.6) {
    feedback.push("Work on maintaining more consistent eye contact to build rapport with interviewers.");
  } else if (eyeContact >= 0.8) {
    feedback.push("Excellent eye contact throughout the interview, which helps establish trust.");
  }
  
  // Filler word feedback
  if (fillerWordCount > 5) {
    feedback.push(`You used approximately ${fillerWordCount} filler words (like 'um', 'uh', 'like'). Try to reduce these for clearer communication.`);
  } else if (fillerWordCount <= 2) {
    feedback.push("You used very few filler words, which made your responses sound polished and prepared.");
  }
  
  // If no specific feedback, provide a general positive comment
  if (feedback.length === 0) {
    feedback.push("Overall, you performed well in this mock interview. Continue practicing to enhance your skills further.");
  }
  
  return feedback;
};

// Function to determine the color based on score
export const getScoreColor = (score: number): string => {
  if (score >= 80) return 'text-green-600';
  if (score >= 70) return 'text-green-500';
  if (score >= 60) return 'text-yellow-500';
  if (score >= 50) return 'text-yellow-600';
  return 'text-red-500';
};

// Function to generate improvement suggestions
export const generateImprovementSuggestions = (score: InterviewScore): string[] => {
  const suggestions: string[] = [];
  
  // Add custom suggestions based on lowest scoring areas
  const scores = [
    { name: 'confidence', value: score.confidence },
    { name: 'engagement', value: score.engagement },
    { name: 'communication', value: score.communication },
    { name: 'bodyLanguage', value: score.bodyLanguage },
    { name: 'eyeContact', value: score.eyeContact }
  ];
  
  // Sort areas by score (lowest first)
  scores.sort((a, b) => a.value - b.value);
  
  // Add suggestions for the two lowest scoring areas
  for (let i = 0; i < Math.min(2, scores.length); i++) {
    const area = scores[i];
    
    if (area.value < 70) {
      switch (area.name) {
        case 'confidence':
          suggestions.push("Practice power poses for 2 minutes before interviews to boost confidence");
          suggestions.push("Record yourself answering common questions and review to build confidence");
          break;
        case 'engagement':
          suggestions.push("Incorporate more vocal variety by practicing emphasizing key words");
          suggestions.push("Show enthusiasm by smiling appropriately and using positive language");
          break;
        case 'communication':
          suggestions.push("Practice speaking more slowly and deliberately to improve clarity");
          suggestions.push("Record yourself and count filler words to become more aware of them");
          break;
        case 'bodyLanguage':
          suggestions.push("Practice interviews in front of a mirror to be aware of your posture and gestures");
          suggestions.push("Maintain an open posture with uncrossed arms and occasional hand gestures");
          break;
        case 'eyeContact':
          suggestions.push("When practicing, place a sticker near your camera to remind you to look at it");
          suggestions.push("In virtual interviews, look directly at the camera instead of the screen");
          break;
      }
    }
  }
  
  // Add a general suggestion if no specific ones were generated
  if (suggestions.length === 0) {
    suggestions.push("Continue building your interview skills by practicing regularly with different types of questions");
  }
  
  return suggestions;
};
