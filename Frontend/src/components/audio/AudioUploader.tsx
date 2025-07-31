import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Card, CardContent } from "@/components/ui/card";
import { Upload, FileAudio } from "lucide-react";
import { toast } from "sonner";

export const AudioUploader = () => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    acceptedFiles.forEach((file) => {
      if (file.type.startsWith('audio/')) {
        toast.success(`Uploaded: ${file.name}`);
        // TODO: Process audio file
      } else {
        toast.error(`Invalid file type: ${file.name}`);
      }
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.m4a', '.flac']
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
                Supports WAV, MP3, M4A, FLAC
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};