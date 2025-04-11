
import React from 'react';
import { Button } from '@/components/ui/button';
import { 
  Play, 
  Square, 
  RefreshCw,
  ChevronRight,
  RotateCcw
} from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

interface InterviewControlsProps {
  isRecording: boolean;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
  disabled?: boolean;
}

const InterviewControls: React.FC<InterviewControlsProps> = ({
  isRecording,
  onStart,
  onStop,
  onReset,
  disabled = false
}) => {
  return (
    <div className="flex items-center justify-center gap-4">
      <TooltipProvider>
        {!isRecording ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button 
                onClick={onStart} 
                disabled={disabled}
                className="bg-interview-primary hover:bg-interview-primary/90 text-white"
                size="lg"
              >
                <Play className="mr-2 h-4 w-4" />
                Start Interview
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Begin recording your mock interview</p>
            </TooltipContent>
          </Tooltip>
        ) : (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button 
                onClick={onStop} 
                variant="destructive"
                size="lg"
              >
                <Square className="mr-2 h-4 w-4" />
                End Interview
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Stop recording and analyze results</p>
            </TooltipContent>
          </Tooltip>
        )}
        
        <Tooltip>
          <TooltipTrigger asChild>
            <Button 
              onClick={onReset} 
              variant="outline"
              size="icon"
              className="ml-2"
              disabled={isRecording}
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Reset interview session</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  );
};

export default InterviewControls;
