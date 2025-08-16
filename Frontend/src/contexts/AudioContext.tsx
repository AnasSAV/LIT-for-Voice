import React, { createContext, useContext, useState, useRef, useCallback } from 'react';

export interface AudioFile {
  id: string;
  file: File;
  name: string;
  duration?: number;
  url: string;
  predictions?: {
    transcription?: string;
    emotion?: string;
    model?: string;
  }[];
}

interface AudioContextType {
  audioFiles: AudioFile[];
  currentAudio: AudioFile | null;
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  volume: number;
  
  // Actions
  addAudioFile: (file: File) => Promise<string>;
  selectAudio: (audioId: string) => void;
  removeAudio: (audioId: string) => void;
  playPause: () => void;
  seekTo: (time: number) => void;
  setVolume: (volume: number) => void;
  
  // Predictions
  runPrediction: (audioId: string, model: string) => Promise<void>;
  isLoadingPrediction: boolean;
}

const AudioContext = createContext<AudioContextType | undefined>(undefined);

export const useAudio = () => {
  const context = useContext(AudioContext);
  if (!context) {
    throw new Error('useAudio must be used within an AudioProvider');
  }
  return context;
};

interface AudioProviderProps {
  children: React.ReactNode;
}

export const AudioProvider: React.FC<AudioProviderProps> = ({ children }) => {
  const [audioFiles, setAudioFiles] = useState<AudioFile[]>([]);
  const [currentAudio, setCurrentAudio] = useState<AudioFile | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolumeState] = useState(70);
  const [isLoadingPrediction, setIsLoadingPrediction] = useState(false);
  
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const addAudioFile = useCallback(async (file: File): Promise<string> => {
    const audioId = `audio-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const url = URL.createObjectURL(file);
    
    // Create a temporary audio element to get duration
    const tempAudio = new Audio(url);
    
    return new Promise((resolve) => {
      tempAudio.onloadedmetadata = () => {
        const newAudioFile: AudioFile = {
          id: audioId,
          file,
          name: file.name,
          duration: tempAudio.duration,
          url,
          predictions: []
        };
        
        setAudioFiles(prev => [...prev, newAudioFile]);
        resolve(audioId);
      };
    });
  }, []);

  const selectAudio = useCallback((audioId: string) => {
    const audio = audioFiles.find(a => a.id === audioId);
    if (audio) {
      setCurrentAudio(audio);
      setIsPlaying(false);
      setCurrentTime(0);
      
      // Create new audio element
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      
      const newAudio = new Audio(audio.url);
      newAudio.volume = volume / 100;
      
      newAudio.onloadedmetadata = () => {
        setDuration(newAudio.duration);
      };
      
      newAudio.ontimeupdate = () => {
        setCurrentTime(newAudio.currentTime);
      };
      
      newAudio.onended = () => {
        setIsPlaying(false);
        setCurrentTime(0);
      };
      
      audioRef.current = newAudio;
    }
  }, [audioFiles, volume]);

  const removeAudio = useCallback((audioId: string) => {
    setAudioFiles(prev => {
      const updated = prev.filter(a => a.id !== audioId);
      // Clean up URL
      const audioToRemove = prev.find(a => a.id === audioId);
      if (audioToRemove) {
        URL.revokeObjectURL(audioToRemove.url);
      }
      return updated;
    });
    
    if (currentAudio?.id === audioId) {
      setCurrentAudio(null);
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    }
  }, [currentAudio]);

  const playPause = useCallback(() => {
    if (!audioRef.current) return;
    
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  }, [isPlaying]);

  const seekTo = useCallback((time: number) => {
    if (!audioRef.current) return;
    
    audioRef.current.currentTime = time;
    setCurrentTime(time);
  }, []);

  const setVolume = useCallback((newVolume: number) => {
    setVolumeState(newVolume);
    if (audioRef.current) {
      audioRef.current.volume = newVolume / 100;
    }
  }, []);

  const runPrediction = useCallback(async (audioId: string, model: string) => {
    const audio = audioFiles.find(a => a.id === audioId);
    if (!audio) return;

    setIsLoadingPrediction(true);
    
    try {
      const formData = new FormData();
      formData.append('file', audio.file);
      formData.append('model', model);

      const response = await fetch('http://localhost:8000/inferences/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      // Update the audio file with prediction results
      setAudioFiles(prev => prev.map(a => {
        if (a.id === audioId) {
          const updatedPredictions = [...(a.predictions || [])];
          const existingIndex = updatedPredictions.findIndex(p => p.model === model);
          
          const newPrediction = {
            model,
            transcription: result.text || result.prediction?.text,
            emotion: result.emotion || result.prediction?.emotion,
          };
          
          if (existingIndex >= 0) {
            updatedPredictions[existingIndex] = newPrediction;
          } else {
            updatedPredictions.push(newPrediction);
          }
          
          return {
            ...a,
            predictions: updatedPredictions
          };
        }
        return a;
      }));
      
    } catch (error) {
      console.error('Error running prediction:', error);
      throw error;
    } finally {
      setIsLoadingPrediction(false);
    }
  }, [audioFiles]);

  const value: AudioContextType = {
    audioFiles,
    currentAudio,
    isPlaying,
    currentTime,
    duration,
    volume,
    
    addAudioFile,
    selectAudio,
    removeAudio,
    playPause,
    seekTo,
    setVolume,
    
    runPrediction,
    isLoadingPrediction,
  };

  return (
    <AudioContext.Provider value={value}>
      {children}
    </AudioContext.Provider>
  );
};
