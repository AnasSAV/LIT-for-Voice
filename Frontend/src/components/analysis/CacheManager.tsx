import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Trash2, RefreshCw, Database, Clock, Download, RotateCcw } from "lucide-react";
import { toast } from "sonner";

interface CacheStats {
  redis_entries: number;
  redis_size_bytes: number;
  redis_size_mb: number;
  file_entries: number;
  file_size_bytes: number;
  file_size_mb: number;
  database_entries: number;
  database_files: number;
  database_size_bytes: number;
  database_size_mb: number;
  total_size_mb: number;
  cache_dir: string;
  database_path: string;
  redis_available: boolean;
  models: string[];
  datasets: string[];
}

interface CachedCombination {
  model: string;
  dataset: string;
  file_count: number;
  created_at: string;
  updated_at: string;
}

interface CacheManagerProps {
  onCacheCleared?: () => void;
}

export const CacheManager = ({ onCacheCleared }: CacheManagerProps) => {
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [cachedCombinations, setCachedCombinations] = useState<CachedCombination[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [isRebuilding, setIsRebuilding] = useState(false);

  const fetchCacheStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/database/stats');
      if (response.ok) {
        const result = await response.json();
        setCacheStats(result.stats);
      }
    } catch (error) {
      console.error('Failed to fetch cache stats:', error);
    }
  };

  const fetchCachedCombinations = async () => {
    try {
      const response = await fetch('http://localhost:8000/database/predictions');
      if (response.ok) {
        const data = await response.json();
        setCachedCombinations(data.predictions || []);
      }
    } catch (error) {
      console.error('Failed to fetch cached combinations:', error);
    }
  };

  const exportDatabase = async () => {
    setIsExporting(true);
    try {
      const response = await fetch('http://localhost:8000/database/export/download');
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `predictions_export_${new Date().toISOString().replace(/[:.]/g, '-')}.sql`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        toast.success('Database exported successfully!');
      } else {
        throw new Error('Failed to export database');
      }
    } catch (error) {
      console.error('Failed to export database:', error);
      toast.error('Failed to export database');
    } finally {
      setIsExporting(false);
    }
  };

  const rebuildCache = async () => {
    setIsRebuilding(true);
    try {
      const response = await fetch('http://localhost:8000/database/rebuild-cache', {
        method: 'POST'
      });
      
      if (response.ok) {
        const result = await response.json();
        toast.success(`Cache rebuilt! ${result.rebuilt_count} prediction sets restored`);
        
        // Refresh cache data
        await fetchCacheStats();
        await fetchCachedCombinations();
        
        if (onCacheCleared) {
          onCacheCleared();
        }
      } else {
        throw new Error('Failed to rebuild cache');
      }
    } catch (error) {
      console.error('Failed to rebuild cache:', error);
      toast.error('Failed to rebuild cache');
    } finally {
      setIsRebuilding(false);
    }
  };

  const clearCache = async (model?: string, dataset?: string) => {
    setIsLoading(true);
    try {
      let url = 'http://localhost:8000/database/predictions';
      const params = new URLSearchParams();
      if (model) params.append('model', model);
      if (dataset) params.append('dataset', dataset);
      if (params.toString()) url += `?${params.toString()}`;

      const response = await fetch(url, { method: 'DELETE' });
      if (response.ok) {
        const result = await response.json();
        toast.success(`Cache cleared! ${result.deleted_count} entries removed`);
        
        // Refresh cache data
        await fetchCacheStats();
        await fetchCachedCombinations();
        
        // Notify parent component
        if (onCacheCleared) {
          onCacheCleared();
        }
      } else {
        throw new Error('Failed to clear cache');
      }
    } catch (error) {
      console.error('Failed to clear cache:', error);
      toast.error('Failed to clear cache');
    } finally {
      setIsLoading(false);
    }
  };

  const refreshData = async () => {
    setIsLoading(true);
    try {
      await Promise.all([fetchCacheStats(), fetchCachedCombinations()]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return timestamp;
    }
  };

  useEffect(() => {
    refreshData();
  }, []);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm">Cache Statistics</CardTitle>
            <Button
              size="sm"
              variant="outline"
              onClick={refreshData}
              disabled={isLoading}
              className="h-7"
            >
              <RefreshCw className="h-3 w-3 mr-1" />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-2">
          {cacheStats ? (
            <>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <Database className="h-4 w-4" />
                    <span>Redis entries:</span>
                    <Badge variant="secondary">{cacheStats.redis_entries}</Badge>
                    {!cacheStats.redis_available && (
                      <Badge variant="destructive" className="text-xs">Offline</Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <Database className="h-4 w-4" />
                    <span>File backup:</span>
                    <Badge variant="outline">{cacheStats.file_entries}</Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    <Database className="h-4 w-4" />
                    <span>Database:</span>
                    <Badge variant="secondary">{cacheStats.database_entries}</Badge>
                    <span className="text-xs text-muted-foreground">
                      ({cacheStats.database_files} files)
                    </span>
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="text-sm">
                    <span>Total cache size:</span>
                    <Badge variant="outline" className="ml-2">
                      {cacheStats.total_size_mb} MB
                    </Badge>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Redis: {cacheStats.redis_size_mb} MB | Files: {cacheStats.file_size_mb} MB
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Database: {cacheStats.database_size_mb} MB
                  </div>
                </div>
              </div>
              
              <Separator />
              
              <div className="space-y-2">
                <div className="text-sm font-medium">Available Models & Datasets:</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="font-medium">Models:</span>
                    <div className="mt-1 space-y-1">
                      {cacheStats.models.length > 0 ? (
                        cacheStats.models.map((model, i) => (
                          <Badge key={i} variant="outline" className="text-xs mr-1">
                            {model}
                          </Badge>
                        ))
                      ) : (
                        <span className="text-muted-foreground">None</span>
                      )}
                    </div>
                  </div>
                  <div>
                    <span className="font-medium">Datasets:</span>
                    <div className="mt-1 space-y-1">
                      {cacheStats.datasets.length > 0 ? (
                        cacheStats.datasets.map((dataset, i) => (
                          <Badge key={i} variant="outline" className="text-xs mr-1">
                            {dataset}
                          </Badge>
                        ))
                      ) : (
                        <span className="text-muted-foreground">None</span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
              
              <Separator />
              
              <div className="flex gap-2 flex-wrap">
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={() => clearCache()}
                  disabled={isLoading}
                  className="h-7"
                >
                  <Trash2 className="h-3 w-3 mr-1" />
                  Clear All Cache
                </Button>
                
                <Button
                  size="sm"
                  variant="outline"
                  onClick={exportDatabase}
                  disabled={isExporting}
                  className="h-7"
                >
                  <Download className="h-3 w-3 mr-1" />
                  {isExporting ? 'Exporting...' : 'Export SQL'}
                </Button>
                
                <Button
                  size="sm"
                  variant="outline"
                  onClick={rebuildCache}
                  disabled={isRebuilding}
                  className="h-7"
                >
                  <RotateCcw className="h-3 w-3 mr-1" />
                  {isRebuilding ? 'Rebuilding...' : 'Rebuild Cache'}
                </Button>
              </div>
            </>
          ) : (
            <div className="text-center text-sm text-muted-foreground py-4">
              {isLoading ? "Loading cache stats..." : "No cache data available"}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Cached Predictions</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {cachedCombinations.length > 0 ? (
            cachedCombinations.map((combination, index) => (
              <div
                key={index}
                className="p-2 border rounded-md space-y-1"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge variant="default" className="text-xs">
                      {combination.model}
                    </Badge>
                    <Badge variant="secondary" className="text-xs">
                      {combination.dataset}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {combination.file_count} files
                    </span>
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => clearCache(combination.model, combination.dataset)}
                    disabled={isLoading}
                    className="h-6 w-6 p-0"
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </div>
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Clock className="h-3 w-3" />
                  <span>
                    Created: {formatTimestamp(combination.created_at)}
                    {combination.updated_at !== combination.created_at && (
                      <span className="ml-2">
                        Updated: {formatTimestamp(combination.updated_at)}
                      </span>
                    )}
                  </span>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center text-sm text-muted-foreground py-4">
              No cached predictions found
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
