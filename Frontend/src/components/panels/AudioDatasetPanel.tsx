import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Upload, Search, Play, Pause } from "lucide-react";
import { AudioUploader } from "../audio/AudioUploader";
import { AudioDataTable } from "../audio/AudioDataTable";

interface ApiData {
  prediction: {
    text: string;
  };
}

interface AudioDatasetPanelProps {
  apiData: ApiData | null; // allow null when not loaded
}

export const AudioDatasetPanel = ({ apiData }: AudioDatasetPanelProps) => {
  const [selectedRow, setSelectedRow] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  return (
    <div className="h-full panel-background flex flex-col">
      <div className="panel-header p-3 border-b panel-border">
        <div className="flex items-center justify-between">
          <h3 className="font-medium text-sm">Audio Dataset</h3>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="text-xs">
              1,247 samples
            </Badge>
            <Button size="sm" variant="outline" className="h-7">
              <Upload className="h-3 w-3 mr-1" />
              Upload
            </Button>
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
            />
          </CardContent>
        </Card>
      </div>
      
      {/* Upload overlay */}
      <AudioUploader />
    </div>
  );
};