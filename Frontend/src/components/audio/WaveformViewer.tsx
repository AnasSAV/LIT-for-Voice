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
  const objectUrlRef = useRef<string | null>(null);
  const [reloadTick, setReloadTick] = useState(0);
  const onReadyRef = useRef<typeof onReady>(onReady);
  const onProgressRef = useRef<typeof onProgress>(onProgress);

  // Keep latest callbacks without retriggering init
  useEffect(() => { onReadyRef.current = onReady; }, [onReady]);
  useEffect(() => { onProgressRef.current = onProgress; }, [onProgress]);

  // Initialize WaveSurfer instance (run once)
  useEffect(() => {
    if (!waveformRef.current) return;
    
    console.log('Initializing WaveSurfer...');
    
    // Create WaveSurfer instance
    const wavesurfer = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#6366f1',
      progressColor: '#4338ca',
      cursorColor: '#818cf8',
      height: 80,
      normalize: true,
      barWidth: 2,
      barRadius: 1,
      cursorWidth: 1,
      hideScrollbar: true,
    });

    wavesurferRef.current = wavesurfer;

    // Set up event listeners
    wavesurfer.on('ready', () => {
      console.log('WaveSurfer ready, duration:', wavesurfer.getDuration());
      setIsLoading(false);
      setError(null);
      if (onReadyRef.current) {
        onReadyRef.current(wavesurfer);
      }
    });

    wavesurfer.on('audioprocess', (currentTime) => {
      if (onProgressRef.current) {
        onProgressRef.current(currentTime, wavesurfer.getDuration());
      }
    });

    // 'interaction' event isn't typed in wavesurfer.js types
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (wavesurfer as any).on('interaction', () => {
      if (onProgressRef.current) {
        const currentTime = wavesurfer.getCurrentTime();
        const duration = wavesurfer.getDuration();
        onProgressRef.current(currentTime, duration || 0);
      }
    });

    wavesurfer.on('error', (err) => {
      console.error('WaveSurfer error:', err);
      setError('Failed to load audio file. Please check if the file exists and is accessible.');
      setIsLoading(false);
    });

    wavesurfer.on('loading', (progress) => {
      console.log('Loading progress:', progress);
      if (progress < 100) {
        setIsLoading(true);
      }
    });

    // Cleanup function
    return () => {
      console.log('Cleaning up WaveSurfer...');
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
        wavesurferRef.current = null;
      }
      // Revoke any remaining object URL to avoid memory leaks
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, []);

  // Handle audio URL changes
  useEffect(() => {
    if (!wavesurferRef.current || !audioUrl) {
      console.log('No WaveSurfer instance or audioUrl:', { wavesurfer: !!wavesurferRef.current, audioUrl });
      return;
    }

    console.log('Fetching audio with credentials:', audioUrl);
    setIsLoading(true);
    setError(null);

    // Revoke any previous object URL before creating a new one
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }

    fetch(audioUrl, { method: 'GET', credentials: 'include' })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        objectUrlRef.current = url;
        try {
          wavesurferRef.current?.load(url);
        } catch (err) {
          console.error('WaveSurfer load error:', err);
          setError(`WaveSurfer failed to load: ${(err as Error).message}`);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        console.error('Failed to fetch audio:', err);
        setError(`Cannot access audio file: ${err.message}`);
        setIsLoading(false);
      });
  }, [audioUrl, reloadTick]);

  // Handle play/pause state
  useEffect(() => {
    if (!wavesurferRef.current) return;
    
    try {
      if (isPlaying) {
        wavesurferRef.current.play();
      } else {
        wavesurferRef.current.pause();
      }
    } catch (err) {
      console.error('Error controlling playback:', err);
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
        className="h-20 bg-muted/30 rounded relative min-h-[80px]"
      >
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center z-10 bg-background/80">
            <div className="text-xs text-muted-foreground flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
              Loading waveform...
            </div>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center z-10 bg-background/80">
            <div className="text-xs text-red-500 text-center px-2">
              <div className="font-medium">⚠️ Error loading audio</div>
              <div className="mt-1">{error}</div>
              {audioUrl && (
                <div className="mt-2">
                  <button 
                    onClick={() => {
                      setError(null);
                      setIsLoading(true);
                      setReloadTick((t) => t + 1);
                    }}
                    className="px-2 py-1 bg-primary text-primary-foreground rounded text-xs hover:bg-primary/80"
                  >
                    Retry
                  </button>
                </div>
              )}
              {audioUrl && (
                <div className="mt-2 text-xs text-muted-foreground break-all">
                  URL: {audioUrl}
                </div>
              )}
            </div>
          </div>
        )}
        {!audioUrl && !isLoading && !error && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="text-xs text-muted-foreground text-center">
              <div>🎵 No audio file selected</div>
              <div className="mt-1">Upload and select a file to view waveform</div>
            </div>
          </div>
        )}
      </div>
      
      {/* Time markers and controls */}
      <div className="flex justify-between text-xs text-muted-foreground mt-2">
        <span>0:00</span>
        <span className="text-center flex-1 italic">
          {audioUrl ? 'Click waveform to seek' : 'Waveform will appear here'}
        </span>
        <span>{wavesurferRef.current ? formatTime(wavesurferRef.current.getDuration() || 0) : '0:00'}</span>
      </div>
    
    </Card>
  );
};