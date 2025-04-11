
import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import WebcamFeed from '@/components/interview/WebcamFeed';
import InterviewControls from '@/components/interview/InterviewControls';
import InterviewQuestion from '@/components/interview/InterviewQuestion';
import { getRandomQuestion, getInterviewSet, InterviewQuestion as IQuestion } from '@/utils/interviewQuestions';
import { analyzeFace, FaceAnalysisResult } from '@/utils/faceAnalysis';
import { analyzeVoice, VoiceAnalysisResult, startVoiceAnalysis, stopVoiceAnalysis } from '@/utils/voiceAnalysis';
import { calculateInterviewScore, InterviewScore, getScoreColor, generateImprovementSuggestions } from '@/utils/scoreCalculation';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Check, AlertCircle, Info } from 'lucide-react';
import { useToast } from '@/components/ui/use-toast';
import { Progress } from '@/components/ui/progress';

const Practice = () => {
  const { toast } = useToast();
  
  // Interview state
  const [isRecording, setIsRecording] = useState(false);
  const [questions, setQuestions] = useState<IQuestion[]>([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [timeRemaining, setTimeRemaining] = useState(60); // 60 seconds per question
  const [analysisResults, setAnalysisResults] = useState<FaceAnalysisResult[]>([]);
  const [voiceResults, setVoiceResults] = useState<VoiceAnalysisResult | null>(null);
  const [score, setScore] = useState<InterviewScore | null>(null);
  const [hasCompletedInterview, setHasCompletedInterview] = useState(false);
  
  // Timer refs
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  
  // Initialize interview questions
  useEffect(() => {
    const randomQuestions = getInterviewSet(5);
    setQuestions(randomQuestions);
  }, []);
  
  // Handle interview start
  const handleStart = () => {
    setIsRecording(true);
    setAnalysisResults([]);
    setVoiceResults(null);
    setScore(null);
    setHasCompletedInterview(false);
    setTimeRemaining(60);
    
    // Start voice analysis
    startVoiceAnalysis();
    
    toast({
      title: "Interview started",
      description: "Answer the question clearly and confidently.",
    });
    
    // Start timer
    timerRef.current = setInterval(() => {
      setTimeRemaining((prev) => {
        if (prev <= 1) {
          // Move to next question or end interview
          if (currentQuestionIndex < questions.length - 1) {
            setCurrentQuestionIndex((prev) => prev + 1);
            return 60; // Reset timer for next question
          } else {
            // End interview if on last question
            handleStop();
            return 0;
          }
        }
        return prev - 1;
      });
    }, 1000);
  };
  
  // Handle interview stop
  const handleStop = () => {
    setIsRecording(false);
    
    // Clear timer
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    
    // Stop voice analysis and get results
    const voiceResult = stopVoiceAnalysis();
    setVoiceResults(voiceResult);
    
    // Calculate average face analysis results
    const avgFaceResults = analysisResults.length > 0
      ? {
          confidence: analysisResults.reduce((acc, curr) => acc + curr.confidence, 0) / analysisResults.length,
          engagement: analysisResults.reduce((acc, curr) => acc + curr.engagement, 0) / analysisResults.length,
          nervousness: analysisResults.reduce((acc, curr) => acc + curr.nervousness, 0) / analysisResults.length,
          smile: analysisResults.reduce((acc, curr) => acc + curr.smile, 0) / analysisResults.length,
          eyeContact: analysisResults.reduce((acc, curr) => acc + curr.eyeContact, 0) / analysisResults.length,
        }
      : {
          confidence: 0.5,
          engagement: 0.5,
          nervousness: 0.5,
          smile: 0.5,
          eyeContact: 0.5,
        };
    
    // Calculate interview score
    const interviewScore = calculateInterviewScore(avgFaceResults, voiceResult);
    setScore(interviewScore);
    setHasCompletedInterview(true);
    
    toast({
      title: "Interview completed",
      description: "View your results and feedback below.",
    });
  };
  
  // Handle interview reset
  const handleReset = () => {
    setCurrentQuestionIndex(0);
    setTimeRemaining(60);
    setAnalysisResults([]);
    setVoiceResults(null);
    setScore(null);
    setHasCompletedInterview(false);
    
    // Generate new questions
    const randomQuestions = getInterviewSet(5);
    setQuestions(randomQuestions);
    
    toast({
      title: "Interview reset",
      description: "Ready for a new practice session.",
    });
  };
  
  // Video frame analysis handler
  const handleVideoFrame = (videoElement: HTMLVideoElement) => {
    const analysis = analyzeFace(videoElement);
    setAnalysisResults((prev) => [...prev, analysis]);
  };
  
  // Prepare data for the score chart
  const prepareScoreData = () => {
    if (!score) return [];
    
    return [
      { name: 'Overall', score: score.overall },
      { name: 'Confidence', score: score.confidence },
      { name: 'Engagement', score: score.engagement },
      { name: 'Communication', score: score.communication },
      { name: 'Body Language', score: score.bodyLanguage },
      { name: 'Eye Contact', score: score.eyeContact },
    ];
  };
  
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Header />
      
      <main className="flex-grow container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-6">Practice Interview</h1>
        
        {hasCompletedInterview && score ? (
          <Tabs defaultValue="results" className="w-full">
            <TabsList className="mb-4">
              <TabsTrigger value="results">Results</TabsTrigger>
              <TabsTrigger value="feedback">Feedback & Improvements</TabsTrigger>
              <TabsTrigger value="practice">Practice Again</TabsTrigger>
            </TabsList>
            
            <TabsContent value="results">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <Card className="lg:col-span-1">
                  <CardHeader>
                    <CardTitle>Your Interview Score</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-col items-center justify-center p-6">
                      <div className={`score-indicator w-32 h-32 ${getScoreColor(score.overall)} bg-gray-50 border-4 border-current mb-4`}>
                        {score.overall}%
                      </div>
                      <p className="text-center text-gray-600">Overall Score</p>
                      
                      <div className="grid grid-cols-2 gap-4 mt-8 w-full">
                        <div className="text-center">
                          <div className={`text-lg font-semibold ${getScoreColor(score.confidence)}`}>
                            {score.confidence}%
                          </div>
                          <p className="text-sm text-gray-600">Confidence</p>
                        </div>
                        <div className="text-center">
                          <div className={`text-lg font-semibold ${getScoreColor(score.engagement)}`}>
                            {score.engagement}%
                          </div>
                          <p className="text-sm text-gray-600">Engagement</p>
                        </div>
                        <div className="text-center">
                          <div className={`text-lg font-semibold ${getScoreColor(score.communication)}`}>
                            {score.communication}%
                          </div>
                          <p className="text-sm text-gray-600">Communication</p>
                        </div>
                        <div className="text-center">
                          <div className={`text-lg font-semibold ${getScoreColor(score.eyeContact)}`}>
                            {score.eyeContact}%
                          </div>
                          <p className="text-sm text-gray-600">Eye Contact</p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                <Card className="lg:col-span-2">
                  <CardHeader>
                    <CardTitle>Performance Breakdown</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={prepareScoreData()}
                          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" />
                          <YAxis domain={[0, 100]} />
                          <Tooltip />
                          <Bar dataKey="score" fill="#3B82F6" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    
                    <div className="mt-8 space-y-4">
                      {voiceResults && (
                        <div>
                          <h3 className="font-semibold text-lg mb-2">Speech Analysis</h3>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                              <p className="text-sm text-gray-600 mb-1">Filler Words</p>
                              <Progress value={voiceResults.fillerWordCount * 10} className="h-2" />
                              <p className="text-xs text-gray-500 mt-1">
                                {voiceResults.fillerWordCount} detected
                              </p>
                            </div>
                            <div>
                              <p className="text-sm text-gray-600 mb-1">Speaking Clarity</p>
                              <Progress value={voiceResults.clarity * 100} className="h-2" />
                              <p className="text-xs text-gray-500 mt-1">
                                {Math.round(voiceResults.clarity * 100)}% clear
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
            
            <TabsContent value="feedback">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Info className="h-5 w-5 text-interview-primary" />
                      Personalized Feedback
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-4">
                      {score.feedback.map((feedback, index) => (
                        <li key={index} className="flex gap-3">
                          <Check className="h-5 w-5 text-green-500 shrink-0 mt-0.5" />
                          <p>{feedback}</p>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <AlertCircle className="h-5 w-5 text-interview-accent" />
                      Areas to Improve
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-4">
                      {generateImprovementSuggestions(score).map((suggestion, index) => (
                        <li key={index} className="flex gap-3">
                          <div className="bg-blue-100 text-interview-primary rounded-full w-6 h-6 flex items-center justify-center shrink-0 mt-0.5">
                            {index + 1}
                          </div>
                          <p>{suggestion}</p>
                        </li>
                      ))}
                    </ul>
                    
                    <div className="mt-8 p-4 bg-gray-50 rounded-lg border border-gray-200">
                      <h4 className="font-semibold mb-2">Pro Tip</h4>
                      <p className="text-sm text-gray-600">
                        Research shows that practicing interviews at least 5 times can improve 
                        your performance by up to 60% in real interviews.
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </div>
              
              <div className="mt-6 text-center">
                <Button 
                  onClick={handleReset}
                  className="bg-interview-primary hover:bg-interview-primary/90"
                >
                  Practice Again
                </Button>
              </div>
            </TabsContent>
            
            <TabsContent value="practice">
              <div className="mb-6">
                <h2 className="text-2xl font-bold mb-4">Ready for Another Practice Session?</h2>
                <p className="text-gray-600 mb-8">
                  Keep practicing to improve your interview skills. Each session will give you new 
                  insights and help you build confidence.
                </p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <Card>
                    <CardHeader>
                      <CardTitle>Continue with Similar Questions</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-gray-600 mb-6">
                        Practice with the same type of questions to master your responses.
                      </p>
                      <Button onClick={handleReset} className="w-full bg-interview-primary hover:bg-interview-primary/90">
                        Start New Session
                      </Button>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardHeader>
                      <CardTitle>Focus on Improvement Areas</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-gray-600 mb-6">
                        Get specialized practice focusing on your lowest-scoring areas.
                      </p>
                      <Button onClick={handleReset} className="w-full">
                        Targeted Practice
                      </Button>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <WebcamFeed 
                onVideoFrame={handleVideoFrame} 
                isRecording={isRecording} 
              />
              
              <div className="mt-6">
                <InterviewControls 
                  isRecording={isRecording}
                  onStart={handleStart}
                  onStop={handleStop}
                  onReset={handleReset}
                  disabled={questions.length === 0}
                />
              </div>
            </div>
            
            <div className="space-y-6">
              {questions.length > 0 && (
                <InterviewQuestion 
                  question={questions[currentQuestionIndex]?.question || "Loading question..."}
                  category={questions[currentQuestionIndex]?.category || "General"}
                  timeRemaining={isRecording ? timeRemaining : undefined}
                />
              )}
              
              <Card>
                <CardHeader>
                  <CardTitle>Interview Tips</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-3 text-sm">
                    <li className="flex gap-2">
                      <Check className="h-4 w-4 text-green-500 shrink-0 mt-0.5" />
                      <p>Maintain eye contact with the camera</p>
                    </li>
                    <li className="flex gap-2">
                      <Check className="h-4 w-4 text-green-500 shrink-0 mt-0.5" />
                      <p>Speak clearly and at a moderate pace</p>
                    </li>
                    <li className="flex gap-2">
                      <Check className="h-4 w-4 text-green-500 shrink-0 mt-0.5" />
                      <p>Use the STAR method for behavioral questions</p>
                    </li>
                    <li className="flex gap-2">
                      <Check className="h-4 w-4 text-green-500 shrink-0 mt-0.5" />
                      <p>Avoid filler words (um, uh, like)</p>
                    </li>
                    <li className="flex gap-2">
                      <Check className="h-4 w-4 text-green-500 shrink-0 mt-0.5" />
                      <p>Show enthusiasm and positive body language</p>
                    </li>
                  </ul>
                </CardContent>
              </Card>
              
              {isRecording && analysisResults.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Live Feedback</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div>
                        <p className="text-sm text-gray-600 mb-1">Confidence</p>
                        <Progress 
                          value={analysisResults[analysisResults.length - 1]?.confidence * 100 || 0} 
                          className="h-2" 
                        />
                      </div>
                      <div>
                        <p className="text-sm text-gray-600 mb-1">Engagement</p>
                        <Progress 
                          value={analysisResults[analysisResults.length - 1]?.engagement * 100 || 0} 
                          className="h-2" 
                        />
                      </div>
                      <div>
                        <p className="text-sm text-gray-600 mb-1">Nervousness</p>
                        <Progress 
                          value={analysisResults[analysisResults.length - 1]?.nervousness * 100 || 0} 
                          className="h-2" 
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        )}
      </main>
      
      <Footer />
    </div>
  );
};

export default Practice;
