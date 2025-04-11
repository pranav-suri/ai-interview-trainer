
import React, { useRef, useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Mic, MicOff, Video, VideoOff } from 'lucide-react';

interface WebcamFeedProps {
  onVideoFrame?: (videoEl: HTMLVideoElement) => void;
  isRecording: boolean;
}

const WebcamFeed: React.FC<WebcamFeedProps> = ({ onVideoFrame, isRecording }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isMicOn, setIsMicOn] = useState(true);
  const [isVideoOn, setIsVideoOn] = useState(true);
  const [permissionDenied, setPermissionDenied] = useState(false);
  const [micPermissionDenied, setMicPermissionDenied] = useState(false);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const constraints = {
          video: true,
          audio: true,
        };
        
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        
        setPermissionDenied(false);
        setMicPermissionDenied(false);
      } catch (err) {
        console.error('Error accessing camera or microphone:', err);
        // Check if it's a permission error
        if (err instanceof DOMException && 
            (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError')) {
          if (err.message.includes('audio')) {
            setMicPermissionDenied(true);
          } else {
            setPermissionDenied(true);
          }
        }
      }
    };

    startCamera();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    if (!isRecording || !videoRef.current || !onVideoFrame) return;
    
    const interval = setInterval(() => {
      if (videoRef.current) {
        onVideoFrame(videoRef.current);
      }
    }, 200); // Analyze every 200ms
    
    return () => clearInterval(interval);
  }, [isRecording, onVideoFrame]);

  const toggleMicrophone = () => {
    if (streamRef.current) {
      streamRef.current.getAudioTracks().forEach(track => {
        track.enabled = !isMicOn;
      });
      setIsMicOn(!isMicOn);
    }
  };

  const toggleVideo = () => {
    if (streamRef.current) {
      streamRef.current.getVideoTracks().forEach(track => {
        track.enabled = !isVideoOn;
      });
      setIsVideoOn(!isVideoOn);
    }
  };

  return (
    <Card className="overflow-hidden">
      <div className="video-container aspect-video bg-gray-900">
        {permissionDenied ? (
          <div className="flex items-center justify-center h-full text-white">
            <p>Camera access denied. Please allow camera access and reload.</p>
          </div>
        ) : (
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            muted 
            className="w-full h-full object-cover"
          />
        )}
        
        {micPermissionDenied && (
          <div className="absolute top-4 right-4 bg-yellow-600 text-white px-3 py-1 rounded-full text-sm font-medium">
            Microphone access denied
          </div>
        )}
        
        {isRecording && (
          <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-500 text-white px-3 py-1 rounded-full text-sm font-medium">
            <div className="w-2 h-2 bg-white rounded-full animate-pulse-subtle"></div>
            Recording
          </div>
        )}
        
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex gap-2">
          <Button 
            size="icon" 
            variant="secondary" 
            className="rounded-full bg-black/70 text-white hover:bg-black/90"
            onClick={toggleMicrophone}
          >
            {isMicOn ? <Mic size={18} /> : <MicOff size={18} />}
          </Button>
          <Button 
            size="icon" 
            variant="secondary" 
            className="rounded-full bg-black/70 text-white hover:bg-black/90"
            onClick={toggleVideo}
          >
            {isVideoOn ? <Video size={18} /> : <VideoOff size={18} />}
          </Button>
        </div>
      </div>
    </Card>
  );
};

export default WebcamFeed;
