import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { EmbeddingPlot } from "../visualization/EmbeddingPlot";
import { ScalarPlot } from "../visualization/ScalarPlot";
import { useEmbedding } from "../../contexts/EmbeddingContext";
import { RefreshCw, Eye } from "lucide-react";

interface EmbeddingPanelProps {
  model?: string;
  dataset?: string;
  availableFiles?: string[];
}

export const EmbeddingPanel = ({ model = "whisper-base", dataset = "common-voice", availableFiles = [] }: EmbeddingPanelProps) => {
  const [reductionMethod, setReductionMethod] = useState("pca");
  const { embeddingData, isLoading, error, fetchEmbeddings, clearEmbeddings } = useEmbedding();

  const handleFetchEmbeddings = () => {
    if (availableFiles.length > 0) {
      // Limit to first 50 files to avoid overwhelming the system
      const filesToProcess = availableFiles.slice(0, 50);
      fetchEmbeddings(model, dataset, filesToProcess, reductionMethod, 2);
    }
  };

  const handleReductionMethodChange = (method: string) => {
    setReductionMethod(method);
    if (embeddingData && availableFiles.length > 0) {
      // Re-fetch with new reduction method
      const filesToProcess = availableFiles.slice(0, 50);
      fetchEmbeddings(model, dataset, filesToProcess, method, 2);
    }
  };
  return (
    <div className="h-full panel-background border-r panel-border flex flex-col">
      <div className="panel-header p-3 border-b panel-border">
        <h3 className="font-medium text-sm">Embeddings & Scalars</h3>
      </div>
      
      <div className="flex-1 p-3 overflow-auto">
        <Tabs defaultValue="embeddings" className="h-full">
          <TabsList className="grid w-full grid-cols-2 h-8">
            <TabsTrigger value="embeddings" className="text-xs">Embeddings</TabsTrigger>
            <TabsTrigger value="scalars" className="text-xs">Scalars</TabsTrigger>
          </TabsList>
          
          <TabsContent value="embeddings" className="mt-3 h-[calc(100%-2rem)]">
            <Card className="h-full flex flex-col">
              <CardHeader className="pb-2 flex-shrink-0">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CardTitle className="text-sm">Audio Embeddings</CardTitle>
                    {embeddingData && (
                      <Badge variant="outline" className="text-[10px]">
                        {embeddingData.model}
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <Select value={reductionMethod} onValueChange={handleReductionMethodChange}>
                      <SelectTrigger className="w-20 h-6 text-xs">
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
                      className="h-6 w-6 p-0"
                    >
                      <RefreshCw className={`h-3 w-3 ${isLoading ? 'animate-spin' : ''}`} />
                    </Button>
                  </div>
                </div>
                {availableFiles.length === 0 && (
                  <div className="text-xs text-muted-foreground">
                    No files available for embedding extraction
                  </div>
                )}
                {availableFiles.length > 0 && !embeddingData && !isLoading && (
                  <div className="text-xs text-muted-foreground">
                    Click <RefreshCw className="inline h-3 w-3 mx-1" /> to extract embeddings from {Math.min(availableFiles.length, 50)} files
                  </div>
                )}
              </CardHeader>
              <CardContent className="flex-1 min-h-0">
                <EmbeddingPlot selectedMethod={reductionMethod} />
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="scalars" className="mt-3 h-[calc(100%-2rem)]">
            <div className="space-y-3 h-full">
              <Card className="flex-1">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Confidence Distribution</CardTitle>
                </CardHeader>
                <CardContent className="h-24">
                  <ScalarPlot type="confidence" />
                </CardContent>
              </Card>
              
              <Card className="flex-1">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Attention Scores</CardTitle>
                </CardHeader>
                <CardContent className="h-24">
                  <ScalarPlot type="attention" />
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};