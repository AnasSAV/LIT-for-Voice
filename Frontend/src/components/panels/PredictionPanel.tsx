import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SaliencyVisualization } from "../visualization/SaliencyVisualization";
import { PerturbationTools } from "../analysis/PerturbationTools";
import { ScalersVisualization } from "../visualization/ScalersVisualization";
import { useState } from "react";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

interface PerturbationResult {
  perturbed_file: string;
  filename: string;
  duration_ms: number;
  sample_rate: number;
  applied_perturbations: Array<{
    type: string;
    params: Record<string, any>;
    status: string;
    error?: string;
  }>;
  success: boolean;
  error?: string;
}

interface PredictionPanelProps {
  selectedFile?: UploadedFile | null;
  selectedEmbeddingFile?: string | null;
  model?: string;
  dataset?: string;
  originalDataset?: string;
  onPerturbationComplete?: (result: PerturbationResult) => void;
  onPredictionRefresh?: (file: UploadedFile, prediction: string) => void;
  onPredictionUpdate?: (fileId: string, prediction: string) => void;
}

export const PredictionPanel = ({ selectedFile, selectedEmbeddingFile, model, dataset, originalDataset, onPerturbationComplete, onPredictionRefresh, onPredictionUpdate }: PredictionPanelProps) => {
  // Handle perturbation completion
  const handlePerturbationComplete = async (result: PerturbationResult) => {
    if (!result.success) {
      console.error("Perturbation failed:", result.error);
      return;
    }

    // Notify parent component about perturbation completion
    if (onPerturbationComplete) {
      onPerturbationComplete(result);
    }
  };

  return (
    <div className="h-full panel-background border-t panel-border">
      <Tabs defaultValue="scalers" className="h-full">
        <div className="panel-header border-b panel-border px-3 py-2">
            <TabsList className="h-7 grid grid-cols-3 w-full">
            <TabsTrigger value="scalers" className="text-xs">Scalers</TabsTrigger>
            <TabsTrigger value="saliency" className="text-xs">Saliency</TabsTrigger>
            <TabsTrigger value="perturbation" className="text-xs">Perturbation</TabsTrigger>
          </TabsList>
        </div>
        
        <div className="h-[calc(100%-2.5rem)] overflow-auto">
          <TabsContent value="scalers" className="m-0 h-full">
            <div className="p-3 h-full">
              <ScalersVisualization 
                model={model}
                dataset={dataset}
              />
            </div>
          </TabsContent>
          
          <TabsContent value="saliency" className="m-0 h-full">
            <div className="p-3">
              <SaliencyVisualization 
                selectedFile={selectedFile || selectedEmbeddingFile} 
                model={model} 
                dataset={dataset} 
                originalDataset={originalDataset}
              />
            </div>
          </TabsContent>
          
          <TabsContent value="perturbation" className="m-0 h-full">
            <div className="p-3">
              <PerturbationTools 
                selectedFile={selectedFile} 
                onPerturbationComplete={handlePerturbationComplete}
                onPredictionRefresh={onPredictionRefresh}
                model={model}
                dataset={dataset}
                originalDataset={originalDataset}
              />
            </div>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
};