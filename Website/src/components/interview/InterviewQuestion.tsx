
import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface InterviewQuestionProps {
  question: string;
  category: string;
  timeRemaining?: number;
}

const InterviewQuestion: React.FC<InterviewQuestionProps> = ({
  question,
  category,
  timeRemaining
}) => {
  // Format time remaining into MM:SS
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs < 10 ? '0' + secs : secs}`;
  };

  return (
    <Card className="shadow-md">
      <CardContent className="pt-6">
        <div className="flex justify-between items-center mb-3">
          <Badge variant="outline" className="bg-interview-background text-interview-primary">
            {category}
          </Badge>
          {timeRemaining !== undefined && (
            <div className="text-sm font-medium text-gray-500">
              Time: {formatTime(timeRemaining)}
            </div>
          )}
        </div>
        <h3 className="text-xl font-semibold leading-tight">{question}</h3>
      </CardContent>
    </Card>
  );
};

export default InterviewQuestion;
