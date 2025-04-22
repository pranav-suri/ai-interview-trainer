import React, { useState, useCallback } from 'react'; // Import useCallback
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import WebcamFeed from '@/components/interview/WebcamFeed';
import { processInterviewVideo } from '@/utils/apiService';
import { Loader2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { Progress } from '@/components/ui/progress';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend
} from 'recharts';

// Sample/mock data (use to display when API fails)
const SAMPLE_API_RESULTS = {
  relevance_score: 4,
  clarity_score: 3,
  tone_score: 4,
  vocabulary_score: 5,
  star_format_score: 3,
  strengths: [
    'Clear and confident voice',
    'Maintained good eye contact',
    'Used specific examples'
  ],
  areas_for_improvement: [
    'Could be more concise',
    'Expand on results achieved'
  ]
};

// Utility for transforming scores to chart-friendly format
function getChartData(apiResults: any) {
  return [
    { name: 'Relevance', value: apiResults.relevance_score ?? 0 },
    { name: 'Clarity', value: apiResults.clarity_score ?? 0 },
    { name: 'Tone', value: apiResults.tone_score ?? 0 },
    { name: 'Vocabulary', value: apiResults.vocabulary_score ?? 0 },
    { name: 'STAR', value: apiResults.star_format_score ?? 0 }
  ];
}

const Practice = () => {
  const { toast } = useToast();

  const [isRecording, setIsRecording] = useState(false);
  const [recordedVideo, setRecordedVideo] = useState<Blob | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [apiResults, setApiResults] = useState<any>(null);
  const [usedSample, setUsedSample] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState(
    'Describe a time when you had to overcome a significant challenge at work.'
  );

  // UI state: clear all when starting new
  const handleStart = () => {
    setIsRecording(true);
    setApiResults(null);
    setUsedSample(false);
    setRecordedVideo(null);
    toast({
      title: "Recording started",
      description: "Please answer the interview question.",
    });
  };

  const handleStop = () => {
    setIsRecording(false);
    // Note: The actual processing starts when the WebcamFeed calls onRecordingReady
    // This toast might be slightly premature if there's a delay in the stop event.
    toast({
      title: "Recording stopped",
      description: "Processing your video response.",
    });
  };

  // Main video processing & API fetch logic - WRAPPED IN useCallback
  const handleRecordingReady = useCallback(async (videoBlob: Blob) => {
    // Check if we are already processing to prevent potential double triggers
    // although useCallback should primarily fix this.
    // Also check if we intended to stop recording.
    // console.log("isProcessing:", isProcessing, "isRecording:", isRecording);
    // if (isProcessing || isRecording) {
    //   console.warn('Already processing or recording in progress. Ignoring new recording.');
    //   return
    // };

    setRecordedVideo(videoBlob);
    setIsProcessing(true);
    setUsedSample(false); // Reset sample flag
    setApiResults(null); // Clear previous results before fetching new ones

    try {
      toast({
        title: "Processing video",
        description: "Sending your interview to be analyzed...",
      });
      const result = await processInterviewVideo(videoBlob, currentQuestion);
      setApiResults(result);
      toast({
        title: "Analysis complete",
        description: "Your interview has been processed successfully.",
      });
    } catch (error) {
      console.error('Error processing video:', error);
      setApiResults(SAMPLE_API_RESULTS);
      setUsedSample(true);
      toast({
        title: "Processing error",
        description: "Could not analyze video via API. Displaying example results instead.",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
    // Dependencies for useCallback: Include everything from the outer scope that the function uses.
    // State setters (like setIsProcessing) are guaranteed stable by React and technically don't need to be listed,
    // but including them is safer and often required by ESLint rules.
  }, [currentQuestion, toast, isProcessing, isRecording]); // Added dependencies

  // Prepare chart data (empty by default)
  const chartData = apiResults ? getChartData(apiResults) : [];

  // For radar chart (show distribution in categories)
  const radarData = [
    { category: 'Relevance', Score: apiResults?.relevance_score ?? 0, fullMark: 5 },
    { category: 'Clarity', Score: apiResults?.clarity_score ?? 0, fullMark: 5 },
    { category: 'Tone', Score: apiResults?.tone_score ?? 0, fullMark: 5 },
    { category: 'Vocabulary', Score: apiResults?.vocabulary_score ?? 0, fullMark: 5 },
    { category: 'STAR', Score: apiResults?.star_format_score ?? 0, fullMark: 5 }
  ];

  // Determine title based on whether sample data is used
  const resultTitle = usedSample ? "Practice Interview (Sample Data)" : apiResults ? "Practice Interview (API Results)" : "Practice Interview";

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <main className="flex-grow container mx-auto px-4 py-8 max-w-2xl">
        {/* Updated Title Logic */}
        <h1 className="text-3xl font-bold mb-6">
          {resultTitle}
        </h1>
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Interview Question</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-lg">{currentQuestion}</p>
          </CardContent>
        </Card>
        <WebcamFeed
          isRecording={isRecording}
          onRecordingReady={handleRecordingReady}
        />
        <div className="flex items-center gap-4 my-6">
          <Button
            onClick={handleStart}
            disabled={isRecording || isProcessing}
            className="bg-interview-primary hover:bg-interview-primary/90"
          >
            Start Recording
          </Button>
          <Button
            onClick={handleStop}
            disabled={!isRecording || isProcessing}
          >
            Stop Recording
          </Button>
        </div>
        {isProcessing && (
          <div className="flex items-center justify-center py-6">
            <Loader2 className="h-8 w-8 animate-spin text-interview-primary mr-2" />
            <p>Processing your interview video...</p>
          </div>
        )}
        {/* Conditionally render results only when not processing and results exist */}
        {!isProcessing && apiResults && (
          <div className="space-y-8">
            {/* Main scores as charts */}
            <Card>
              <CardHeader>
                <CardTitle>Interview Scores Overview</CardTitle>
              </CardHeader>
              <CardContent>
                {/* Rest of the chart/results rendering logic remains the same */}
                <div className="w-full max-w-xl mx-auto">
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis domain={[0, 5]} ticks={[0,1,2,3,4,5]} />
                      <Tooltip />
                      <Bar dataKey="value" fill="#0ea5e9" barSize={32} radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-6 w-full max-w-xl mx-auto">
                  <ResponsiveContainer width="100%" height={250}>
                    <RadarChart data={radarData}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="category" />
                      <PolarRadiusAxis angle={30} domain={[0, 5]} tickCount={6} /> {/* Adjusted tickCount */}
                      <Radar name="Score" dataKey="Score" stroke="#0ea5e9" fill="#0ea5e9" fillOpacity={0.5} />
                      {/* <Legend /> Optionally add legend back */}
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
                 <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-4 mt-8">
                  {/* Progress Bars for each section */}
                  {chartData.map(d => (
                    <Card key={d.name} className="bg-gray-50 flex flex-col items-center">
                      <CardContent className="w-full pt-6">
                        <h4 className="font-medium mb-2 text-center">{d.name}</h4>
                        <div className="flex items-center justify-center">
                          <div className={`text-2xl font-bold
                            ${d.value >= 4 ? 'text-green-600'
                            : d.value >= 3 ? 'text-yellow-600'
                            : 'text-red-500'}`}>
                            {d.value}/5
                          </div>
                        </div>
                        <Progress value={d.value * 20} className="h-2 mt-3" />
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Strengths</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="list-disc pl-6 space-y-1">
                  {apiResults.strengths?.length
                    ? apiResults.strengths.map((strength: string, idx: number) => (
                        <li key={idx}>{strength}</li>
                      ))
                    : <li>No strengths detected.</li>}
                </ul>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Areas for Improvement</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="list-disc pl-6 space-y-1">
                  {apiResults.areas_for_improvement?.length
                    ? apiResults.areas_for_improvement.map((area: string, idx: number) => (
                        <li key={idx}>{area}</li>
                      ))
                    : <li>No improvement areas detected.</li>}
                </ul>
              </CardContent>
            </Card>
          </div>
        )}
      </main>
    </div>
  );
};

export default Practice;