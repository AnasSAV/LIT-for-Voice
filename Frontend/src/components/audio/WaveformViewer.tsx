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

  // Initialize WaveSurfer instance
  useEffect(() => {
    if (!waveformRef.current) return;
    
    console.log('Initializing WaveSurfer...');
    
    // Create WaveSurfer instance
    const wavesurfer = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#3b82f6', // Blue-500 for waveform
      progressColor: '#1d4ed8', // Blue-700 for progress
      cursorColor: '#60a5fa', // Blue-400 for cursor
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
      if (onReady) {
        onReady(wavesurfer);
      }
    });

    wavesurfer.on('audioprocess', (currentTime) => {
      if (onProgress) {
        onProgress(currentTime, wavesurfer.getDuration());
      }
    });

    wavesurfer.on('interaction' as any, () => {
      if (onProgress) {
        const currentTime = wavesurfer.getCurrentTime();
        const duration = wavesurfer.getDuration();
        onProgress(currentTime, duration || 0);
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
    };
  }, []); // Only run once when component mounts

  // Handle audio URL changes
  useEffect(() => {
    if (!wavesurferRef.current || !audioUrl) {
      console.log('No WaveSurfer instance or audioUrl:', { wavesurfer: !!wavesurferRef.current, audioUrl });
      return;
    }

    console.log('Loading audio URL:', audioUrl);
    setIsLoading(true);
    setError(null);
    
    // First, test if the URL is accessible
    fetch(audioUrl, { method: 'HEAD' })
      .then(response => {
        console.log('Audio URL HEAD response:', response.status, response.statusText);
        if (!response.ok) {
          throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        
        // If HEAD request succeeds, load with WaveSurfer
        try {
          wavesurferRef.current?.load(audioUrl);
        } catch (err) {
          console.error('WaveSurfer load error:', err);
          setError(`WaveSurfer failed to load: ${err.message}`);
          setIsLoading(false);
        }
      })
      .catch(err => {
        console.error('Audio URL accessibility test failed:', err);
        setError(`Cannot access audio file: ${err.message}`);
        setIsLoading(false);
        
        // Try with GET request as fallback
        fetch(audioUrl, { method: 'GET' })
          .then(response => {
            console.log('Audio URL GET fallback response:', response.status);
            if (response.ok) {
              setError('File accessible but WaveSurfer may have issues with CORS or format');
              // Try loading anyway
              try {
                wavesurferRef.current?.load(audioUrl);
              } catch (wsErr) {
                setError(`WaveSurfer error: ${wsErr.message}`);
              }
            }
          })
          .catch(() => {
            setError('Audio file completely inaccessible');
          });
      });
  }, [audioUrl]);

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
    <Card className="p-3 border-gray-200 bg-white shadow-sm">
      <div 
        ref={waveformRef}
        className="h-20 bg-white rounded relative min-h-[80px] border border-gray-200"
      >
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center z-10 bg-white/90">
            <div className="text-xs text-blue-600 flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
              Loading waveform...
            </div>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center z-10 bg-white/90">
            <div className="text-xs text-red-500 text-center px-2">
              <div className="font-medium">⚠️ Error loading audio</div>
              <div className="mt-1">{error}</div>
              {audioUrl && (
                <div className="mt-2">
                  <button 
                    onClick={() => {
                      setError(null);
                      setIsLoading(true);
                      if (wavesurferRef.current && audioUrl) {
                        wavesurferRef.current.load(audioUrl);
                      }
                    }}
                    className="px-2 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700"
                  >
                    Retry
                  </button>
                </div>
              )}
              {audioUrl && (
                <div className="mt-2 text-xs text-blue-600 break-all">
                  URL: {audioUrl}
                </div>
              )}
            </div>
          </div>
        )}
        {!audioUrl && !isLoading && !error && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="text-xs text-gray-500 text-center">
              <div>🎵 No audio file selected</div>
              <div className="mt-1">Upload and select a file to view waveform</div>
            </div>
          </div>
        )}
      </div>
      
      {/* Time markers and controls */}
      <div className="flex justify-between text-xs text-gray-600 mt-2">
        <span>0:00</span>
        <span className="text-center flex-1 italic">
          {audioUrl ? 'Click waveform to seek' : 'Waveform will appear here'}
        </span>
        <span>{wavesurferRef.current ? formatTime(wavesurferRef.current.getDuration() || 0) : '0:00'}</span>
      </div>
    
    </Card>
  );
};