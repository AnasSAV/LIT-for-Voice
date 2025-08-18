import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, SkipBack, SkipForward, Volume2 } from "lucide-react";

interface AudioPlayerProps {
  isPlaying: boolean;
  onPlayPause: () => void;
  currentTime?: number;
  duration?: number;
  onSeek?: (time: number) => void;
  onVolumeChange?: (volume: number) => void;
}

export const AudioPlayer = ({ 
  isPlaying, 
  onPlayPause, 
  currentTime = 0, 
  duration = 0,
  onSeek,
  onVolumeChange 
}: AudioPlayerProps) => {
  const [volume, setVolume] = useState([70]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleVolumeChange = (newVolume: number[]) => {
    setVolume(newVolume);
    if (onVolumeChange) {
      onVolumeChange(newVolume[0] / 100);
    }
  };

  return (
    <div className="space-y-3">
      {/* Progress bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </div>
        <Slider
          value={[currentTime]}
          onValueChange={(value) => onSeek && onSeek(value[0])}
          max={duration || 100}
          step={0.1}
          className="w-full"
        />
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          <Button size="sm" variant="ghost" className="h-8 w-8 p-0">
            <SkipBack className="h-4 w-4" />
          </Button>
          
          <Button size="sm" onClick={onPlayPause} className="h-8 w-8 p-0">
            {isPlaying ? (
              <Pause className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4" />
            )}
          </Button>
          
          <Button size="sm" variant="ghost" className="h-8 w-8 p-0">
            <SkipForward className="h-4 w-4" />
          </Button>
        </div>

        {/* Volume */}
        <div className="flex items-center gap-2">
          <Volume2 className="h-4 w-4 text-muted-foreground" />
          <Slider
            value={volume}
            onValueChange={handleVolumeChange}
            max={100}
            step={1}
            className="w-16"
          />
        </div>
      </div>
    </div>
  );
};