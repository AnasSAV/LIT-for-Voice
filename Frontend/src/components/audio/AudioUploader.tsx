import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Card, CardContent } from "@/components/ui/card";
import { Upload, FileAudio } from "lucide-react";
import { toast } from "sonner";
import { useAudio } from "@/contexts/AudioContext";

interface AudioUploaderProps {
  onUpload?: () => void;
}

export const AudioUploader = ({ onUpload }: AudioUploaderProps) => {
  const { addAudioFile } = useAudio();
  
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    for (const file of acceptedFiles) {
      if (file.type.startsWith('audio/')) {
        try {
          const audioId = await addAudioFile(file);
          toast.success(`Uploaded: ${file.name}`);
          onUpload?.();
        } catch (error) {
          toast.error(`Failed to upload: ${file.name}`);
          console.error('Upload error:', error);
        }
      } else {
        toast.error(`Invalid file type: ${file.name}`);
      }
    }
  }, [addAudioFile, onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    },
    multiple: true
  });

  return (
    <div
      {...getRootProps()}
      className={`
        fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center
        transition-opacity duration-200
        ${isDragActive ? 'opacity-100' : 'opacity-0 pointer-events-none'}
      `}
    >
      <input {...getInputProps()} />
      <Card className="w-96 border-2 border-dashed border-primary">
        <CardContent className="p-8 text-center">
          <div className="flex flex-col items-center gap-4">
            <div className="p-4 rounded-full bg-primary/10">
              {isDragActive ? (
                <FileAudio className="h-8 w-8 text-primary" />
              ) : (
                <Upload className="h-8 w-8 text-primary" />
              )}
            </div>
            <div>
              <h3 className="font-medium">
                {isDragActive ? "Drop files here" : "Upload Audio Files"}
              </h3>
              <p className="text-sm text-muted-foreground mt-1">
                Supports WAV, MP3, M4A, FLAC, OGG
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};