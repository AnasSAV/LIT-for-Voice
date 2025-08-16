import { useEffect, useRef, useState } from "react";
import { Card } from "@/components/ui/card";
import WaveSurfer from "wavesurfer.js";

interface WaveformViewerProps {
  audioUrl?: string;
  isPlaying?: boolean;
  onReady?: (wavesurfer: WaveSurfer) => void;
  onProgress?: (currentTime: number, duration: number) => void;
}

export const WaveformViewer = ({ audioUrl, isPlaying, onReady, onProgress }: WaveformViewerProps) => {
  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (waveformRef.current && !wavesurferRef.current) {
      // Initialize wavesurfer
      wavesurferRef.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#6366f1',
        progressColor: '#4338ca',
        cursorColor: '#818cf8',
        height: 60,
        normalize: true,
        backend: 'WebAudio',
        responsive: true,
      });

      // Set up event listeners
      wavesurferRef.current.on('ready', () => {
        setIsLoading(false);
        setError(null);
        if (onReady && wavesurferRef.current) {
          onReady(wavesurferRef.current);
        }
      });

      wavesurferRef.current.on('audioprocess', (currentTime) => {
        if (onProgress && wavesurferRef.current) {
          onProgress(currentTime, wavesurferRef.current.getDuration());
        }
      });

      wavesurferRef.current.on('error', (err) => {
        setError('Failed to load audio file');
        setIsLoading(false);
        console.error('WaveSurfer error:', err);
      });
    }

    return () => {
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
        wavesurferRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (wavesurferRef.current && audioUrl) {
      setIsLoading(true);
      setError(null);
      wavesurferRef.current.load(audioUrl);
    }
  }, [audioUrl]);

  useEffect(() => {
    if (wavesurferRef.current) {
      if (isPlaying) {
        wavesurferRef.current.play();
      } else {
        wavesurferRef.current.pause();
      }
    }
  }, [isPlaying]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Card className="p-3">
      <div 
        ref={waveformRef}
        className="h-16 bg-muted/30 rounded relative"
      >
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-xs text-muted-foreground">Loading waveform...</div>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-xs text-red-500">{error}</div>
          </div>
        )}
        {!audioUrl && !isLoading && !error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-xs text-muted-foreground">Select an audio file to view waveform</div>
          </div>
        )}
      </div>
      
      {/* Time markers - will be updated by wavesurfer */}
      <div className="flex justify-between text-xs text-muted-foreground mt-1">
        <span>0:00</span>
        <span className="text-center flex-1">Click waveform to seek</span>
        <span>{wavesurferRef.current ? formatTime(wavesurferRef.current.getDuration() || 0) : '0:00'}</span>
      </div>
    </Card>
  );
};