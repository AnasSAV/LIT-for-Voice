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
  const currentBlobUrlRef = useRef<string | null>(null);

  // Initialize WaveSurfer instance
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
      // Clean up blob URLs to prevent memory leaks
      if (currentBlobUrlRef.current) {
        URL.revokeObjectURL(currentBlobUrlRef.current);
        currentBlobUrlRef.current = null;
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
    
    // First, test if the URL is accessible with proper credentials for cross-origin
    const fetchOptions: RequestInit = { 
      method: 'HEAD',
      credentials: 'include',  // Include credentials for CORS
      mode: 'cors',           // Explicit CORS mode
      headers: {
        'Accept': 'audio/*',
      }
    };
    
    fetch(audioUrl, fetchOptions)
      .then(response => {
        console.log('Audio URL HEAD response:', response.status, response.statusText);
        if (!response.ok) {
          throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        
        // If HEAD request succeeds, try to load with WaveSurfer
        try {
          // For cross-origin requests (like ngrok), try to preload audio data
          if (audioUrl.includes('ngrok') || audioUrl.includes('colab') || audioUrl.includes('https://')) {
            // First try to fetch audio data
            fetch(audioUrl, {
              credentials: 'include',
              mode: 'cors'
            }).then(audioResponse => {
              if (audioResponse.ok) {
                return audioResponse.blob();
              }
              throw new Error('Failed to fetch audio data');
            }).then(blob => {
              // Clean up previous blob URL if exists
              if (currentBlobUrlRef.current) {
                URL.revokeObjectURL(currentBlobUrlRef.current);
              }
              const audioBlob = URL.createObjectURL(blob);
              currentBlobUrlRef.current = audioBlob;
              wavesurferRef.current?.load(audioBlob);
            }).catch(blobErr => {
              console.warn('Blob loading failed, trying direct URL:', blobErr);
              // Fallback to direct URL loading
              wavesurferRef.current?.load(audioUrl);
            });
          } else {
            // Local files can be loaded directly
            wavesurferRef.current?.load(audioUrl);
          }
        } catch (err) {
          console.error('WaveSurfer load error:', err);
          setError(`WaveSurfer failed to load: ${err?.message || 'Unknown error'}`);
          setIsLoading(false);
        }
      })
      .catch(err => {
        console.error('Audio URL accessibility test failed:', err);
        setError(`Cannot access audio file: ${err.message}`);
        setIsLoading(false);
        
        // Try with GET request as fallback
        const getFallbackOptions: RequestInit = {
          method: 'GET',
          credentials: 'include',
          mode: 'cors',
          headers: {
            'Accept': 'audio/*',
          }
        };
        
        fetch(audioUrl, getFallbackOptions)
          .then(response => {
            console.log('Audio URL GET fallback response:', response.status);
            if (response.ok) {
              setError('File accessible, trying blob loading for cross-origin compatibility');
              // Try blob loading for cross-origin
              try {
                if (audioUrl.includes('ngrok') || audioUrl.includes('colab') || audioUrl.includes('https://')) {
                  response.blob().then(blob => {
                    // Clean up previous blob URL if exists
                    if (currentBlobUrlRef.current) {
                      URL.revokeObjectURL(currentBlobUrlRef.current);
                    }
                    const audioBlob = URL.createObjectURL(blob);
                    currentBlobUrlRef.current = audioBlob;
                    wavesurferRef.current?.load(audioBlob);
                  }).catch(blobErr => {
                    setError(`Blob loading failed: ${blobErr?.message || 'Unknown error'}`);
                  });
                } else {
                  wavesurferRef.current?.load(audioUrl);
                }
              } catch (wsErr: any) {
                setError(`WaveSurfer error: ${wsErr?.message || 'Unknown error'}`);
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
                      if (wavesurferRef.current && audioUrl) {
                        wavesurferRef.current.load(audioUrl);
                      }
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