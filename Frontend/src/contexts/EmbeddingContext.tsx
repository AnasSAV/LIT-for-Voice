import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { API_BASE } from '@/lib/api';

export interface EmbeddingPoint {
  filename: string;
  coordinates: number[];
  embedding?: number[];
  embedding_dim?: number;
}

interface EmbeddingData {
  model: string;
  dataset: string;
  reduction_method: string;
  n_components: number;
  embeddings: Array<{
    filename: string;
    embedding: number[];
    embedding_dim: number;
  }>;
  reduced_embeddings?: EmbeddingPoint[];
  total_files: number;
  original_dimension: number;
}

interface EmbeddingContextType {
  embeddingData: EmbeddingData | null;
  isLoading: boolean;
  error: string | null;
  fetchEmbeddings: (
    model: string,
    dataset: string,
    files: string[],
    reductionMethod?: string,
    nComponents?: number
  ) => Promise<void>;
  clearEmbeddings: () => void;
}

const EmbeddingContext = createContext<EmbeddingContextType | undefined>(undefined);

export const useEmbedding = () => {
  const context = useContext(EmbeddingContext);
  if (context === undefined) {
    throw new Error('useEmbedding must be used within an EmbeddingProvider');
  }
  return context;
};

export const EmbeddingProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [embeddingData, setEmbeddingData] = useState<EmbeddingData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchEmbeddings = useCallback(async (
    model: string,
    dataset: string,
    files: string[],
    reductionMethod: string = 'pca',
    nComponents: number = 3
  ) => {
    if (!files || files.length === 0) {
      setError('No files provided');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/inferences/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          model,
          dataset,
          files,
          reduction_method: reductionMethod,
          n_components: nComponents,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch embeddings: ${response.status}`);
      }

      const data = await response.json();
      setEmbeddingData(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch embeddings';
      setError(errorMessage);
      console.error('Error fetching embeddings:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearEmbeddings = useCallback(() => {
    setEmbeddingData(null);
    setError(null);
  }, []);

  return (
    <EmbeddingContext.Provider value={{
      embeddingData,
      isLoading,
      error,
      fetchEmbeddings,
      clearEmbeddings,
    }}>
      {children}
    </EmbeddingContext.Provider>
  );
};
