
export interface VoiceAnalysisResult {
  fillerWordCount: number;
  fillerWordPercentage: number;
  toneConfidence: number;
  speakingPace: number;
  clarity: number;
  volume: number;
}

// List of common filler words to detect
const fillerWords = [
  'um', 'uh', 'er', 'ah', 'like', 'you know', 'so', 'actually',
  'basically', 'literally', 'I mean', 'right', 'kind of', 'sort of'
];

// Speech Recognition setup
class SpeechRecognitionService {
  recognition: SpeechRecognition | null = null;
  isListening: boolean = false;
  transcript: string = '';
  wordCount: number = 0;
  startTime: number = 0;
  
  constructor() {
    this.initRecognition();
  }
  
  initRecognition() {
    // Check if browser supports SpeechRecognition
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      this.recognition = new SpeechRecognition();
      
      if (this.recognition) {
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US';
        
        this.recognition.onresult = (event) => {
          let interimTranscript = '';
          
          for (let i = event.resultIndex; i < event.results.length; i++) {
            if (event.results[i].isFinal) {
              this.transcript += event.results[i][0].transcript + ' ';
            } else {
              interimTranscript += event.results[i][0].transcript;
            }
          }
          
          // Count words in the transcript
          this.wordCount = this.transcript.trim().split(/\s+/).length;
        };
        
        this.recognition.onerror = (event) => {
          console.error('Speech recognition error:', event.error);
        };
      }
    } else {
      console.warn('Speech Recognition API not supported in this browser');
    }
  }
  
  start() {
    if (this.recognition && !this.isListening) {
      this.transcript = '';
      this.wordCount = 0;
      this.startTime = Date.now();
      
      try {
        this.recognition.start();
        this.isListening = true;
      } catch (error) {
        console.error('Error starting speech recognition:', error);
      }
    }
  }
  
  stop() {
    if (this.recognition && this.isListening) {
      try {
        this.recognition.stop();
        this.isListening = false;
      } catch (error) {
        console.error('Error stopping speech recognition:', error);
      }
    }
  }
  
  getTranscript() {
    return this.transcript;
  }
  
  getDurationSeconds() {
    return (Date.now() - this.startTime) / 1000;
  }
}

// Create a singleton instance
const speechRecognitionService = new SpeechRecognitionService();

// Analyze voice using speech recognition
export const analyzeVoice = (audioData: any = null): VoiceAnalysisResult => {
  const transcript = speechRecognitionService.getTranscript();
  const duration = speechRecognitionService.getDurationSeconds();
  const wordCount = speechRecognitionService.wordCount;
  
  // Detect filler words
  const detectedFillerWords = detectFillerWords(transcript);
  const fillerWordCount = detectedFillerWords.length;
  
  // Calculate filler word percentage
  const totalWords = wordCount || 1; // Avoid division by zero
  const fillerWordPercentage = fillerWordCount / totalWords;
  
  // Calculate speaking pace
  const speakingPaceWPM = calculateSpeakingPace(wordCount, duration);
  
  // Normalize speaking pace to a 0-1 scale (assuming 120-160 WPM is optimal)
  let normalizedPace = 0.8; // Default to a good score
  if (speakingPaceWPM > 0) {
    if (speakingPaceWPM < 100) {
      // Too slow, scale from 0.4-0.7 based on how close to 100
      normalizedPace = 0.4 + (speakingPaceWPM / 100) * 0.3;
    } else if (speakingPaceWPM > 180) {
      // Too fast, scale from 0.4-0.7 based on how far above 180
      normalizedPace = 0.7 - Math.min(0.3, ((speakingPaceWPM - 180) / 60) * 0.3);
    } else if (speakingPaceWPM >= 100 && speakingPaceWPM <= 180) {
      // Ideal range, higher score the closer to 140
      const distanceFromIdeal = Math.abs(140 - speakingPaceWPM);
      normalizedPace = 1.0 - (distanceFromIdeal / 80) * 0.3;
    }
  }
  
  // Estimate clarity based on speaking pace (too fast or too slow reduces clarity)
  // and filler word percentage (more filler words reduces clarity)
  const clarityFromPace = normalizedPace * 0.6;
  const clarityFromFillerWords = (1 - Math.min(1, fillerWordPercentage * 5)) * 0.4;
  const clarity = clarityFromPace + clarityFromFillerWords;
  
  // We don't have actual volume analysis through Web Speech API,
  // so we'll provide a reasonable default
  const volume = 0.75;
  
  // For tone confidence, we can use a combination of speaking pace
  // and filler word usage as a proxy
  const toneConfidence = 0.8 - (fillerWordPercentage * 0.5);
  
  return {
    fillerWordCount,
    fillerWordPercentage,
    toneConfidence: Math.max(0.3, Math.min(1, toneConfidence)),
    speakingPace: normalizedPace,
    clarity: Math.max(0.3, Math.min(1, clarity)),
    volume
  };
};

// Start speech recognition
export const startVoiceAnalysis = () => {
  speechRecognitionService.start();
};

// Stop speech recognition and return results
export const stopVoiceAnalysis = () => {
  speechRecognitionService.stop();
  return analyzeVoice();
};

// Detect filler words in text
export const detectFillerWords = (text: string): string[] => {
  if (!text) return [];
  
  // Convert text to lowercase for case-insensitive matching
  const lowerText = text.toLowerCase();
  
  // Find all filler words in the text
  return fillerWords.filter(word => {
    const regex = new RegExp(`\\b${word}\\b`, 'gi');
    return lowerText.match(regex);
  });
};

// Calculate speaking pace in words per minute
export const calculateSpeakingPace = (wordCount: number, durationSeconds: number): number => {
  if (durationSeconds === 0 || durationSeconds < 5) return 0;
  return (wordCount / durationSeconds) * 60;
};

// Generate feedback based on voice analysis
export const generateVoiceFeedback = (analysis: VoiceAnalysisResult): string => {
  let feedback = '';
  
  if (analysis.fillerWordPercentage > 0.05) {
    feedback += "Try to reduce filler words like 'um', 'uh', and 'like'. ";
  }
  
  if (analysis.toneConfidence < 0.6) {
    feedback += "Speak with more confidence and authority. ";
  }
  
  if (analysis.speakingPace > 0.8) {
    feedback += "Consider slowing down your speaking pace a bit. ";
  } else if (analysis.speakingPace < 0.4) {
    feedback += "Try to speak a bit faster to maintain engagement. ";
  }
  
  if (analysis.clarity < 0.7) {
    feedback += "Focus on articulating your words more clearly. ";
  }
  
  return feedback || "Your voice sounds good! Keep up the good work.";
};
