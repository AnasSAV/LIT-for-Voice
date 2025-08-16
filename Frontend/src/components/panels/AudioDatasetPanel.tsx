import { useState, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Upload, Search, Play, Pause } from "lucide-react";
import { AudioUploader } from "../audio/AudioUploader";
import { AudioDataTable } from "../audio/AudioDataTable";
import { toast } from "sonner";

interface ApiData {
  prediction: {
    text: string;
  };
}

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
}

interface AudioDatasetPanelProps {
  apiData: ApiData | null;
  uploadedFiles?: UploadedFile[];
  selectedFile?: UploadedFile | null;
  onFileSelect?: (file: UploadedFile) => void;
  onUploadSuccess?: (uploadResponse: UploadedFile) => void;
}

export const AudioDatasetPanel = ({ 
  apiData, 
  uploadedFiles, 
  selectedFile, 
  onFileSelect, 
  onUploadSuccess 
}: AudioDatasetPanelProps) => {
  const [selectedRow, setSelectedRow] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (file.type.startsWith('audio/')) {
          try {
            await uploadFile(file);
          } catch (error) {
            console.error('Upload error:', error);
          }
        } else {
          toast.error(`Invalid file type: ${file.name}. Only audio files are supported.`);
        }
      }
    }
    // Reset the input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await response.json();
      toast.success(`Uploaded: ${file.name}`);
      
      if (onUploadSuccess) {
        onUploadSuccess(data);
      }
      
      return data;
    } catch (error) {
      console.error('Upload error:', error);
      toast.error(`Failed to upload ${file.name}: ${error.message}`);
      throw error;
    }
  };

  return (
    <div className="h-full panel-background flex flex-col">
      <div className="panel-header p-3 border-b panel-border">
        <div className="flex items-center justify-between">
          <h3 className="font-medium text-sm">Audio Dataset</h3>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="text-xs">
              {uploadedFiles ? `${uploadedFiles.length} uploaded` : "0 files"}
            </Badge>
            <Button size="sm" variant="outline" className="h-7" onClick={handleUploadClick}>
              <Upload className="h-3 w-3 mr-1" />
              Upload
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              multiple
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>
        </div>
        
        {/* Search bar */}
        <div className="mt-2 relative">
          <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 h-3 w-3 text-muted-foreground" />
          <Input
            placeholder="Search audio files..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-7 h-8 text-xs"
          />
        </div>
      </div>
      
      <div className="flex-1 overflow-hidden">
        <Card className="h-full rounded-none border-0">
          <CardContent className="p-0 h-full">
            <AudioDataTable 
              selectedRow={selectedRow}
              onRowSelect={setSelectedRow}
              searchQuery={searchQuery}
              apiData={apiData}
              uploadedFiles={uploadedFiles}
              onFilePlay={(file) => {
                console.log('Playing file:', file.filename);
                // This will be connected to the DatapointEditor
                if (onFileSelect) {
                  onFileSelect(file);
                }
              }}
            />
          </CardContent>
        </Card>
      </div>
      
      {/* Upload overlay */}
      <AudioUploader onUploadSuccess={onUploadSuccess} />
    </div>
  );
};