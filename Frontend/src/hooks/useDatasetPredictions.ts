import { useQuery } from '@tanstack/react-query';
import type { QueryObserverResult } from '@tanstack/react-query';
import { DatasetFile } from '@/lib/api/datasets';
import { getBatchPredictions, type BatchPredictionResult, type PredictionResult } from '@/lib/api/predictions';

type UseDatasetPredictionsReturn = {
  predictions: BatchPredictionResult | undefined;
  getPrediction: (file: DatasetFile) => PredictionResult | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<QueryObserverResult<BatchPredictionResult, Error>>;
};

interface UseDatasetPredictionsProps {
  files: DatasetFile[];
  model: string;
  enabled?: boolean;
}

export function useDatasetPredictions({ 
  files, 
  model, 
  enabled = true 
}: UseDatasetPredictionsProps): UseDatasetPredictionsReturn {
  const fileHashes = files.map(file => file.h);
  const queryKey = ['predictions', model, ...fileHashes];
  
  const { 
    data: predictions, 
    isLoading, 
    error,
    refetch 
  } = useQuery({
    queryKey,
    queryFn: () => getBatchPredictions(model, fileHashes),
    enabled: enabled && files.length > 0,
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 30 * 60 * 1000, // 30 minutes (cacheTime was renamed to gcTime in newer versions)
  });

  const getPrediction = (file: DatasetFile): PredictionResult | null => {
    if (!predictions?.results) return null;
    return predictions.results[file.h] ?? null;
  };

  return {
    predictions,
    getPrediction,
    isLoading,
    error: error ?? null,
    refetch,
  };
}