import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, SkipBack, SkipForward, Volume2 } from "lucide-react";

interface AudioPlayerProps {
  isPlaying: boolean;
  onPlayPause: () => void;
}

export const AudioPlayer = ({ isPlaying, onPlayPause }: AudioPlayerProps) => {
  const [currentTime, setCurrentTime] = useState([0]);
  const [duration] = useState(100);
  const [volume, setVolume] = useState([70]);

  return (
    <div className="space-y-3">
      {/* Progress bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>0:00</span>
          <span>3:20</span>
        </div>
        <Slider
          value={currentTime}
          onValueChange={setCurrentTime}
          max={duration}
          step={1}
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
            onValueChange={setVolume}
            max={100}
            step={1}
            className="w-16"
          />
        </div>
      </div>
    </div>
  );
};