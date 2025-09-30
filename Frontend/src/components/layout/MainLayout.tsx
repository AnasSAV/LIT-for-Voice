import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { Toolbar } from "./Toolbar";
import { EmbeddingPanel } from "../panels/EmbeddingPanel";
import { AudioDatasetPanel } from "../panels/AudioDatasetPanel";
import { DatapointEditorPanel } from "../panels/DatapointEditorPanel";
import { PredictionPanel } from "../panels/PredictionPanel";
import { EmbeddingProvider } from "../../contexts/EmbeddingContext";
import React, { useState, useEffect, useCallback } from "react";

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
  
  // Determine effective dataset based on uploaded files and custom datasets
  const effectiveDataset = (() => {
    // If a custom dataset is selected (starts with custom:), use it as-is
    if (dataset.startsWith('custom:')) {
      return dataset;
    }
    // Legacy behavior: if there are uploaded files and no custom dataset, show as "custom"
    if (uploadedFiles && uploadedFiles.length > 0) {
      return "custom";
    }
    // Otherwise use the selected dataset
    return dataset;
  })();
  const [batchInferenceStatus, setBatchInferenceStatus] = useState<'idle' | 'running' | 'done'>('idle');
  const [availableFiles, setAvailableFiles] = useState<string[]>([]);
  const [selectedEmbeddingFile, setSelectedEmbeddingFile] = useState<string | null>(null);
  const [perturbationResult, setPerturbationResult] = useState<any>(null);

  // Clear perturbation result when selected file changes
  useEffect(() => {
    setPerturbationResult(null);
  }, [selectedFile]);

  const handleUploadSuccess = (uploadResponse: UploadedFile) => {
    console.log("DEBUG: Upload success response:", uploadResponse);
    setUploadedFiles(prev => {
      const newFiles = [...prev, uploadResponse];
      console.log("DEBUG: Updated uploaded files:", newFiles);
      return newFiles;
    });
    // Always select the newly uploaded file
    setSelectedFile(uploadResponse);
  };

  const handleFileSelection = (file: UploadedFile) => {
    console.log("DEBUG: handleFileSelection called with:", file.filename, "message:", file.message, "file_path:", file.file_path);
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

  const handlePerturbationComplete = (result: any) => {
    setPerturbationResult(result);
  };

  const handlePredictionRefresh = (file: UploadedFile, prediction: string) => {
    console.log("DEBUG: handlePredictionRefresh called with:", file.filename, prediction);
    
    // For uploaded files, we'll handle prediction updates through the AudioDatasetPanel
    // This function is now mainly for perturbed files
    if (file.message === "Perturbed file") {
      // Add the perturbed file to uploaded files
      setUploadedFiles(prevFiles => {
        const existingFile = prevFiles.find(f => f.file_id === file.file_id);
        if (existingFile) {
          return prevFiles.map(f => 
            f.file_id === file.file_id 
              ? { ...f, prediction: prediction }
              : f
          );
        } else {
          return [...prevFiles, { ...file, prediction: prediction }];
        }
      });
      
      // Update predictionMap for perturbed file
      setPredictionMap(prev => {
        const updated = { ...prev, [file.filename]: prediction };
        console.log("DEBUG: Updated predictionMap for perturbed file:", updated);
        return updated;
      });
    }
    
    // Update selected file if it's the same file
    if (selectedFile && selectedFile.file_id === file.file_id) {
      setSelectedFile(prev => prev ? { ...prev, prediction: prediction } : null);
    }
  };

  const [predictionMap, setPredictionMap] = useState<Record<string, string>>({});

  const handlePredictionUpdate = (fileId: string, prediction: string) => {
    console.log("DEBUG: MainLayout handlePredictionUpdate called with:", fileId, prediction);
    setPredictionMap(prev => {
      const updated = { ...prev, [fileId]: prediction };
      console.log("DEBUG: MainLayout - Updated predictionMap:", updated);
      return updated;
    });
  };

  const handleBatchInferenceStart = useCallback(() => {
    setBatchInferenceStatus('running');
  }, []);

  const handleBatchInferenceComplete = useCallback(() => {
    setBatchInferenceStatus('done');
  }, []);

  // Clear predictions when model or dataset changes
  useEffect(() => {
    console.log('Model or dataset changed, clearing predictions:', model, dataset);
    setPredictionMap({});
    setBatchInferenceStatus('idle');
  }, [model, dataset]);

  const handleBatchInference = async (selectedModel: string, selectedDataset: string) => {
    // Don't run batch inference for custom datasets or legacy "custom" 
    if (selectedDataset === 'custom' || selectedDataset.startsWith('custom:')) return;
    
    // Clear predictions when dataset/model changes to avoid showing old predictions
    console.log('Clearing predictions for new dataset/model combination:', selectedModel, selectedDataset);
    setPredictionMap({});
    
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
            
            {/* Center Panel: Predictions */}
            <Panel defaultSize={50} minSize={30}>
              <PanelGroup direction="vertical">
                <Panel defaultSize={70} minSize={40}>
                  <PredictionPanel 
                    selectedFile={selectedFile}
                    selectedEmbeddingFile={selectedEmbeddingFile}
                    model={model}
                    dataset={effectiveDataset}
                    originalDataset={dataset}
                    onPerturbationComplete={handlePerturbationComplete}
                    onPredictionRefresh={handlePredictionRefresh}
                    onPredictionUpdate={handlePredictionUpdate}
                  />
                </Panel>
                
                <PanelResizeHandle className="h-1 bg-border hover:bg-border/80 transition-colors" />
                
                {/* Bottom Panel: Audio Dataset Table */}
                <Panel defaultSize={30} minSize={20}>
                  <AudioDatasetPanel
                    apiData={apiData}
                    uploadedFiles={uploadedFiles}
                    selectedFile={selectedFile}
                    onFileSelect={handleFileSelection}
                    onUploadSuccess={handleUploadSuccess}
                    model={model}
                    dataset={effectiveDataset}
                    originalDataset={dataset}
                    batchInferenceStatus={batchInferenceStatus}
                    onBatchInferenceStart={handleBatchInferenceStart}
                    onBatchInferenceComplete={handleBatchInferenceComplete}
                    onAvailableFilesChange={setAvailableFiles}
                    onPredictionUpdate={handlePredictionUpdate}
                    predictionMap={predictionMap}
                  />
                </Panel>
              </PanelGroup>
            </Panel>
            
            <PanelResizeHandle className="w-1 bg-border hover:bg-border/80 transition-colors" />
            
            {/* Right Panel: Audio Player & Label Editor */}
            <Panel defaultSize={25} minSize={20}>
              <DatapointEditorPanel 
                selectedFile={selectedFile} 
                dataset={effectiveDataset} 
                perturbationResult={perturbationResult}
                predictionMap={predictionMap}
              />
            </Panel>
          </PanelGroup>
        </div>
      </div>
    </EmbeddingProvider>
  );
};