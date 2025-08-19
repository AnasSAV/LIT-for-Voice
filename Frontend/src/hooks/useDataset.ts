import { useState, useCallback, useEffect } from 'react';
import { DatasetFile, listDatasetFiles } from '@/lib/api/datasets';

type UseDatasetReturn = {
  files: DatasetFile[];
  selectedFile: DatasetFile | null;
  selectFile: (file: DatasetFile | null) => void;
  isLoading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
};

export function useDataset(): UseDatasetReturn {
  const [files, setFiles] = useState<DatasetFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<DatasetFile | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const loadFiles = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const { files: datasetFiles } = await listDatasetFiles(200, 0);
      setFiles(datasetFiles);
      
      // Select the first file by default if none is selected
      if (datasetFiles.length > 0 && !selectedFile) {
        setSelectedFile(datasetFiles[0]);
      }
    } catch (err) {
      console.error('Failed to load dataset files:', err);
      setError(err instanceof Error ? err : new Error('Failed to load dataset files'));
    } finally {
      setIsLoading(false);
    }
  }, [selectedFile]);

  useEffect(() => {
    // Initial load
    loadFiles();

    // Refresh when the active dataset changes (emitted by Toolbar)
    const handler = () => {
      loadFiles();
    };
    window.addEventListener('dataset-changed', handler);
    return () => {
      window.removeEventListener('dataset-changed', handler);
    };
  }, [loadFiles]);

  const selectFile = useCallback((file: DatasetFile | null) => {
    setSelectedFile(file);
  }, []);

  return {
    files,
    selectedFile,
    selectFile,
    isLoading,
    error,
    refresh: loadFiles,
  };
}
