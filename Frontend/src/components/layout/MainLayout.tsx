import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { Toolbar } from "./Toolbar";
import { EmbeddingPanel } from "../panels/EmbeddingPanel";
import { AudioDatasetPanel } from "../panels/AudioDatasetPanel";
import { DatapointEditorPanel } from "../panels/DatapointEditorPanel";
import { PredictionPanel } from "../panels/PredictionPanel";
import { EmbeddingProvider } from "../../contexts/EmbeddingContext";
import React, { useState } from "react";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

export const MainLayout = () => {
  const [apiData, setApiData] = useState<unknown>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<UploadedFile | null>(null);
  const [model, setModel] = useState("whisper-base");
  const [dataset, setDataset] = useState("common-voice");
  const [batchInferenceStatus, setBatchInferenceStatus] = useState<'idle' | 'running' | 'done'>('idle');
  const [availableFiles, setAvailableFiles] = useState<string[]>([]);
  const [selectedEmbeddingFile, setSelectedEmbeddingFile] = useState<string | null>(null);

  const handleUploadSuccess = (uploadResponse: UploadedFile) => {
    setUploadedFiles(prev => [...prev, uploadResponse]);
    // Automatically select the first uploaded file
    if (!selectedFile) {
      setSelectedFile(uploadResponse);
    }
  };

  const handleFileSelection = (file: UploadedFile) => {
    setSelectedFile(file);
    // Sync embedding selection with audio dataset selection
    setSelectedEmbeddingFile(file.filename);
  };

  const handleEmbeddingSelection = (filename: string) => {
    setSelectedEmbeddingFile(filename);
    
    // Try to find and select corresponding file in audio dataset
    // First check uploaded files
    const matchingUploadedFile = uploadedFiles.find(f => f.filename === filename);
    if (matchingUploadedFile) {
      setSelectedFile(matchingUploadedFile);
      return;
    }
    
    // For dataset files, create a file-like object for the UI
    // The AudioDatasetPanel should handle highlighting the corresponding row
    const fileLike: UploadedFile = {
      file_id: filename,
      filename: filename,
      file_path: filename,
      message: "Selected from embeddings"
    };
    setSelectedFile(fileLike);
  };

  const handleBatchInference = async (selectedModel: string, selectedDataset: string) => {
    if (selectedDataset === 'custom') return;
    
    setBatchInferenceStatus('running');
    console.log(`Starting batch inference for ${selectedModel} on ${selectedDataset} dataset`);
    
    try {
      // This will be implemented by AudioDatasetPanel to run inference on all files
      // For now, just set the status to indicate batch inference is requested
      setBatchInferenceStatus('done');
    } catch (error) {
      console.error('Batch inference failed:', error);
      setBatchInferenceStatus('idle');
    }
  };
  return (
    <EmbeddingProvider>
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
          dataset={dataset}
          setDataset={setDataset}
          onBatchInference={handleBatchInference}
        />
        
        {/* Main Content Area */}
        <div className="flex-1 overflow-hidden">
          <PanelGroup direction="horizontal" className="h-full">
            {/* Left Panel: Embeddings & Scalar Plots */}
            <Panel defaultSize={25} minSize={20}>
              <EmbeddingPanel 
                model={model}
                dataset={dataset}
                availableFiles={availableFiles}
                selectedFile={selectedEmbeddingFile}
                onFileSelect={handleEmbeddingSelection}
              />
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
                    onFileSelect={handleFileSelection}
                    onUploadSuccess={handleUploadSuccess}
                    model={model}
                    dataset={dataset}
                    batchInferenceStatus={batchInferenceStatus}
                    onBatchInferenceStart={() => setBatchInferenceStatus('running')}
                    onBatchInferenceComplete={() => setBatchInferenceStatus('done')}
                    onAvailableFilesChange={setAvailableFiles}
                  />
                </Panel>
                
                <PanelResizeHandle className="h-1 bg-border hover:bg-border/80 transition-colors" />
                
                {/* Bottom Panel: Predictions */}
                <Panel defaultSize={30} minSize={20}>
                  <PredictionPanel 
                    selectedFile={selectedFile}
                    selectedEmbeddingFile={selectedEmbeddingFile}
                    model={model}
                    dataset={dataset}
                  />
                </Panel>
              </PanelGroup>
            </Panel>
            
            <PanelResizeHandle className="w-1 bg-border hover:bg-border/80 transition-colors" />
            
            {/* Right Panel: Audio Player & Label Editor */}
            <Panel defaultSize={25} minSize={20}>
              <DatapointEditorPanel selectedFile={selectedFile} dataset={dataset} />
            </Panel>
          </PanelGroup>
        </div>
      </div>
    </EmbeddingProvider>
  );
};