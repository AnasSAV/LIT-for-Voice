import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Card, CardContent } from "@/components/ui/card";
import { Upload, FileAudio } from "lucide-react";
import { toast } from "sonner";

interface AudioUploaderProps {
  onUploadSuccess?: (uploadResponse: any) => void;
}

export const AudioUploader = ({ onUploadSuccess }: AudioUploaderProps) => {
  const uploadFile = async (file: File) => {
    console.log('Starting upload for file:', file.name, 'Type:', file.type, 'Size:', file.size);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      console.log('Sending request to http://localhost:8000/upload...');
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      console.log('Response status:', response.status);

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Upload failed with error:', errorData);
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await response.json();
      console.log('Upload successful:', data);
      toast.success(`Uploaded: ${file.name}`);
      
      // Call the callback with upload response
      if (onUploadSuccess) {
        onUploadSuccess(data);
      }
      
      return data;
    } catch (error) {
      console.error('Upload error:', error);
      toast.error(`Failed to upload ${file.name}: ${error.message || 'Unknown error'}`);
      throw error;
    }
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    console.log('Files dropped:', acceptedFiles.length, 'files');
    acceptedFiles.forEach(async (file, index) => {
      console.log(`Processing file ${index + 1}:`, file.name, 'Type:', file.type);
      if (file.type.startsWith('audio/')) {
        try {
          await uploadFile(file);
        } catch (error) {
          // Error already handled in uploadFile
        }
      } else {
        console.warn('Invalid file type:', file.type);
        toast.error(`Invalid file type: ${file.name}`);
      }
    });
  }, [onUploadSuccess]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.m4a', '.flac']
    },
    multiple: true
  });

  return (
    <>
      {/* Upload drop zone overlay - only visible when dragging */}
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
                <FileAudio className="h-8 w-8 text-primary" />
              </div>
              <div>
                <h3 className="font-medium">Drop files here</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Supports WAV, MP3, M4A, FLAC
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  );
};