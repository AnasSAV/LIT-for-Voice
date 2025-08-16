import { useEffect, useRef, useState } from "react";
import { Card } from "@/components/ui/card";
import { useAudio } from "@/contexts/AudioContext";
import WaveSurfer from "wavesurfer.js";

export const WaveformViewer = () => {
  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const { currentAudio, isPlaying, currentTime, seekTo } = useAudio();
  const [isWaveformReady, setIsWaveformReady] = useState(false);

  // Initialize WaveSurfer
  useEffect(() => {
    if (!waveformRef.current) return;

    // Clean up previous instance
    if (wavesurferRef.current) {
      wavesurferRef.current.destroy();
    }

    // Create new WaveSurfer instance
    const wavesurfer = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: 'hsl(210 40% 70%)',
      progressColor: 'hsl(210 40% 40%)',
      cursorColor: 'hsl(210 40% 20%)',
      barWidth: 2,
      barRadius: 1,
      height: 60,
      normalize: true,
      backend: 'WebAudio',
      interact: true
    });

    // Handle click events for seeking
    wavesurfer.on('click', (progress) => {
      if (currentAudio && currentAudio.duration) {
        const newTime = progress * currentAudio.duration;
        seekTo(newTime);
      }
    });

    wavesurfer.on('ready', () => {
      setIsWaveformReady(true);
    });

    wavesurfer.on('error', (error) => {
      console.error('WaveSurfer error:', error);
      setIsWaveformReady(false);
    });

    wavesurferRef.current = wavesurfer;

    return () => {
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
        wavesurferRef.current = null;
      }
    };
  }, []);

  // Load audio when currentAudio changes
  useEffect(() => {
    if (!wavesurferRef.current || !currentAudio) {
      setIsWaveformReady(false);
      return;
    }

    setIsWaveformReady(false);
    wavesurferRef.current.load(currentAudio.url);
  }, [currentAudio]);

  // Update playback position
  useEffect(() => {
    if (wavesurferRef.current && isWaveformReady && currentAudio?.duration) {
      const progress = currentTime / currentAudio.duration;
      wavesurferRef.current.seekTo(progress);
    }
  }, [currentTime, isWaveformReady, currentAudio]);

  // Placeholder waveform for when no audio is loaded
  const generatePlaceholderWaveform = () => {
    const bars = [];
    for (let i = 0; i < 100; i++) {
      const height = Math.random() * 40 + 10;
      bars.push(
        <div
          key={i}
          className="bg-muted-foreground/30 transition-colors"
          style={{
            width: '2px',
            height: `${height}px`,
            marginRight: '1px'
          }}
        />
      );
    }
    return bars;
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Card className="p-3">
      <div 
        ref={waveformRef}
        className={`h-16 bg-muted/30 rounded ${!currentAudio ? 'hidden' : ''}`}
      />
      
      {!currentAudio && (
        <div className="flex items-center justify-center h-16 bg-muted/30 rounded">
          <div className="flex items-center h-full">
            {generatePlaceholderWaveform()}
          </div>
        </div>
      )}
      
      {/* Time markers */}
      <div className="flex justify-between text-xs text-muted-foreground mt-1">
        <span>{formatTime(currentTime)}</span>
        <span>{currentAudio?.duration ? formatTime(currentAudio.duration) : "0:00"}</span>
      </div>
    </Card>
  );
};