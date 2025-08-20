import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { Toolbar } from "./Toolbar";
import { EmbeddingPanel } from "../panels/EmbeddingPanel";
import { AudioDatasetPanel } from "../panels/AudioDatasetPanel";
import { DatapointEditorPanel } from "../panels/DatapointEditorPanel";
import { PredictionPanel } from "../panels/PredictionPanel";
import React, { useState } from "react";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
  label?: string;
  prediction?: {
    text?: string;
    label?: string;
    confidence?: number;
  };
  dataset_id?: string | null;
  autoplay?: boolean;
  meta?: Record<string, string>;
}

export const MainLayout = () => {
  const [apiData, setApiData] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<UploadedFile | null>(null);

  const handleUploadSuccess = (uploadResponse: UploadedFile) => {
    setUploadedFiles(prev => [...prev, uploadResponse]);
    // Automatically select the first uploaded file
    if (!selectedFile) {
      setSelectedFile(uploadResponse);
    }
  };

  const [model, setModel] = useState("whisper-base");
  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Top Navigation Bar */}
      <Toolbar
        apiData={apiData}
        setApiData={setApiData}
        selectedFile={selectedFile}
        uploadedFiles={uploadedFiles}
        onFileSelect={setSelectedFile}
        model={model}        // current model value
        setModel={setModel} 
      />
      
      {/* Main Content Area */}
      <div className="flex-1 overflow-hidden">
        <PanelGroup direction="horizontal" className="h-full">
          {/* Left Panel: Embeddings & Scalar Plots */}
          <Panel defaultSize={25} minSize={20}>
            <EmbeddingPanel />
          </Panel>
          
          <PanelResizeHandle className="w-1 bg-border hover:bg-border/80 transition-colors" />
          
          {/* Center Panel: Audio Dataset Table */}
          <Panel defaultSize={50} minSize={30}>
            <PanelGroup direction="vertical">
              <Panel defaultSize={70} minSize={40}>
                <AudioDatasetPanel
                  apiData={apiData}
                  uploadedFiles={uploadedFiles}
                  selectedFile={selectedFile}
                  onFileSelect={setSelectedFile}
                  onUploadSuccess={handleUploadSuccess}
                  model={model}
                />
              </Panel>
              
              <PanelResizeHandle className="h-1 bg-border hover:bg-border/80 transition-colors" />
              
              {/* Bottom Panel: Predictions */}
              <Panel defaultSize={30} minSize={20}>
                <PredictionPanel />
              </Panel>
            </PanelGroup>
          </Panel>
          
          <PanelResizeHandle className="w-1 bg-border hover:bg-border/80 transition-colors" />
          
          {/* Right Panel: Audio Player & Label Editor */}
          <Panel defaultSize={25} minSize={20}>
            <DatapointEditorPanel selectedFile={selectedFile} model={model} />
          </Panel>
        </PanelGroup>
      </div>
    </div>
  );
};