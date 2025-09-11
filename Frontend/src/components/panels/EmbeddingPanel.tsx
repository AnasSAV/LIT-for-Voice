import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { EmbeddingPlot } from "../visualization/EmbeddingPlot";
import { ScalarPlot } from "../visualization/ScalarPlot";
import { useEmbedding } from "../../contexts/EmbeddingContext";
import { RefreshCw, Eye, Box, Square } from "lucide-react";

interface EmbeddingPanelProps {
  model?: string;
  dataset?: string;
  availableFiles?: string[];
  selectedFile?: string | null;
  onFileSelect?: (filename: string) => void;
}

export const EmbeddingPanel = ({ model = "whisper-base", dataset = "common-voice", availableFiles = [], selectedFile, onFileSelect }: EmbeddingPanelProps) => {
  const [reductionMethod, setReductionMethod] = useState("pca");
  const [is3D, setIs3D] = useState(false);
  const { embeddingData, isLoading, error, fetchEmbeddings, clearEmbeddings } = useEmbedding();

  const handleFetchEmbeddings = () => {
    if (availableFiles.length > 0) {
      // Use entire dataset for better visualization
      const filesToProcess = availableFiles;
      const nComponents = is3D ? 3 : 2;
      fetchEmbeddings(model, dataset, filesToProcess, reductionMethod, nComponents);
    }
  };

  const handleReductionMethodChange = (method: string) => {
    setReductionMethod(method);
    if (embeddingData && availableFiles.length > 0) {
      // Re-fetch with new reduction method using entire dataset
      const filesToProcess = availableFiles;
      const nComponents = is3D ? 3 : 2;
      fetchEmbeddings(model, dataset, filesToProcess, method, nComponents);
    }
  };

  const handle3DToggle = (checked: boolean) => {
    setIs3D(checked);
    if (embeddingData && availableFiles.length > 0) {
      // Re-fetch with new dimensionality using entire dataset
      const filesToProcess = availableFiles;
      const nComponents = checked ? 3 : 2;
      fetchEmbeddings(model, dataset, filesToProcess, reductionMethod, nComponents);
    }
  };

  const handlePointSelect = (filename: string, coordinates: number[]) => {
    if (onFileSelect) {
      onFileSelect(filename);
    }
  };

  return (
    <div className="h-full bg-white border-r border-gray-200 flex flex-col">
      <div className="p-4 border-b border-gray-200 bg-gray-50">
        <h3 className="font-medium text-sm text-gray-800">Audio Embeddings</h3>
      </div>
      
      <div className="flex-1 p-4 bg-white flex flex-col min-h-0">
        <div className="flex flex-col h-full gap-4">
          {/* Controls Section */}
          <div className="flex-shrink-0 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {embeddingData && (
                  <Badge variant="default" className="text-[10px] bg-blue-100 text-blue-800 border-blue-200">
                    {embeddingData.model.toUpperCase()}
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Select value={reductionMethod} onValueChange={handleReductionMethodChange}>
                  <SelectTrigger className="w-20 h-6 text-xs border-gray-300 hover:border-gray-400 transition-colors">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="pca">PCA</SelectItem>
                    <SelectItem value="umap">UMAP</SelectItem>
                    <SelectItem value="tsne">t-SNE</SelectItem>
                  </SelectContent>
                </Select>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleFetchEmbeddings}
                  disabled={isLoading || availableFiles.length === 0}
                  className="h-6 w-6 p-0 border-gray-300 hover:border-blue-400 hover:bg-blue-50 transition-all duration-200"
                >
                  <RefreshCw className={`h-3 w-3 ${isLoading ? 'animate-spin text-blue-600' : 'text-gray-600'}`} />
                </Button>
              </div>
            </div>
            
            {/* 3D Toggle */}
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-200">
              <div className="flex items-center space-x-2">
                <Switch
                  id="3d-mode"
                  checked={is3D}
                  onCheckedChange={handle3DToggle}
                  disabled={isLoading}
                />
                <Label htmlFor="3d-mode" className="text-xs flex items-center gap-1 font-medium text-gray-700">
                  {is3D ? <Box className="h-3 w-3 text-blue-600" /> : <Square className="h-3 w-3 text-gray-500" />}
                  <span className={is3D ? "text-blue-600" : "text-gray-500"}>
                    {is3D ? '3D View' : '2D View'}
                  </span>
                </Label>
              </div>
            </div>

            {/* Status Messages */}
            {availableFiles.length === 0 && (
              <div className="text-xs text-gray-500 flex items-center gap-2 p-2 bg-yellow-50 rounded border border-yellow-200">
                <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                No files available for embedding extraction
              </div>
            )}
            {availableFiles.length > 0 && !embeddingData && !isLoading && (
              <div className="text-xs text-gray-600 flex items-center gap-2 p-2 bg-blue-50 rounded border border-blue-200">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                Click <RefreshCw className="inline h-3 w-3 mx-1" /> to extract embeddings from all {availableFiles.length} files
              </div>
            )}
            {isLoading && (
              <div className="text-xs text-blue-600 flex items-center gap-2 p-2 bg-blue-50 rounded border border-blue-200">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-ping"></div>
                Processing {availableFiles.length} files... This may take a few moments.
              </div>
            )}
            {error && (
              <div className="text-xs text-red-600 flex items-center gap-2 p-2 bg-red-50 rounded border border-red-200">
                <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                {error}
              </div>
            )}
          </div>

          {/* Embedding Plot */}
          <div className="flex-1 border border-gray-200 rounded-lg bg-white p-2 min-h-0 overflow-hidden">
            <EmbeddingPlot 
              selectedMethod={reductionMethod} 
              is3D={is3D}
              onPointSelect={handlePointSelect}
              selectedFile={selectedFile}
            />
          </div>

        </div>
      </div>
    </div>
  );
};