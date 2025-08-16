import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, SkipBack, SkipForward, Volume2 } from "lucide-react";
import { useAudio } from "@/contexts/AudioContext";

export const AudioPlayer = () => {
  const { 
    currentAudio, 
    isPlaying, 
    currentTime, 
    duration, 
    volume, 
    playPause, 
    seekTo, 
    setVolume 
  } = useAudio();

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleSeek = (value: number[]) => {
    seekTo(value[0]);
  };

  const handleVolumeChange = (value: number[]) => {
    setVolume(value[0]);
  };

  const handleSkipBack = () => {
    const newTime = Math.max(0, currentTime - 10);
    seekTo(newTime);
  };

  const handleSkipForward = () => {
    const newTime = Math.min(duration, currentTime + 10);
    seekTo(newTime);
  };

  if (!currentAudio) {
    return (
      <div className="space-y-3 opacity-50">
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>0:00</span>
            <span>0:00</span>
          </div>
          <Slider
            value={[0]}
            max={100}
            step={1}
            className="w-full"
            disabled
          />
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1">
            <Button size="sm" variant="ghost" className="h-8 w-8 p-0" disabled>
              <SkipBack className="h-4 w-4" />
            </Button>
            
            <Button size="sm" className="h-8 w-8 p-0" disabled>
              <Play className="h-4 w-4" />
            </Button>
            
            <Button size="sm" variant="ghost" className="h-8 w-8 p-0" disabled>
              <SkipForward className="h-4 w-4" />
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <Volume2 className="h-4 w-4 text-muted-foreground" />
            <Slider
              value={[volume]}
              max={100}
              step={1}
              className="w-16"
              disabled
            />
          </div>
        </div>
        <div className="text-xs text-muted-foreground text-center">
          No audio selected
        </div>
      </div>
    );
  }

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
          onValueChange={handleSeek}
          max={duration}
          step={0.1}
          className="w-full"
        />
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          <Button 
            size="sm" 
            variant="ghost" 
            className="h-8 w-8 p-0"
            onClick={handleSkipBack}
          >
            <SkipBack className="h-4 w-4" />
          </Button>
          
          <Button 
            size="sm" 
            onClick={playPause} 
            className="h-8 w-8 p-0"
          >
            {isPlaying ? (
              <Pause className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4" />
            )}
          </Button>
          
          <Button 
            size="sm" 
            variant="ghost" 
            className="h-8 w-8 p-0"
            onClick={handleSkipForward}
          >
            <SkipForward className="h-4 w-4" />
          </Button>
        </div>

        {/* Volume */}
        <div className="flex items-center gap-2">
          <Volume2 className="h-4 w-4 text-muted-foreground" />
          <Slider
            value={[volume]}
            onValueChange={handleVolumeChange}
            max={100}
            step={1}
            className="w-16"
          />
        </div>
      </div>
      
      {/* Current file info */}
      <div className="text-xs text-muted-foreground text-center truncate">
        {currentAudio.name}
      </div>
    </div>
  );
};