import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { EmbeddingPlot } from "../visualization/EmbeddingPlot";
import { ScalarPlot } from "../visualization/ScalarPlot";

export const EmbeddingPanel = () => {
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
            <Card className="h-full">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm">Audio Embeddings</CardTitle>
                  <Select defaultValue="umap">
                    <SelectTrigger className="w-20 h-6 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="umap">UMAP</SelectItem>
                      <SelectItem value="tsne">t-SNE</SelectItem>
                      <SelectItem value="pca">PCA</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardHeader>
              <CardContent className="h-[calc(100%-4rem)]">
                <EmbeddingPlot />
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