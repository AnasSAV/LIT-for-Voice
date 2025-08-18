import { useState, useCallback } from 'react';
import { DatasetFile } from '@/lib/api/datasets';

type UseDatasetReturn = {
  files: DatasetFile[];
  selectedFile: DatasetFile | null;
  selectFile: (file: DatasetFile | null) => void;
  isLoading: boolean;
  error: Error | null;
};

export function useDataset(): UseDatasetReturn {
  const [files, setFiles] = useState<DatasetFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<DatasetFile | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  // TODO: Implement file loading from the API
  // useEffect(() => {
  //   const loadFiles = async () => {
  //     try {
  //       setIsLoading(true);
  //       const response = await fetch('/api/datasets/files');
  //       if (!response.ok) throw new Error('Failed to load files');
  //       const data = await response.json();
  //       setFiles(data.files);
  //     } catch (err) {
  //       setError(err instanceof Error ? err : new Error('Failed to load files'));
  //     } finally {
  //       setIsLoading(false);
  //     }
  //   };
  //   loadFiles();
  // }, []);

  const selectFile = useCallback((file: DatasetFile | null) => {
    setSelectedFile(file);
  }, []);

  return {
    files,
    selectedFile,
    selectFile,
    isLoading,
    error,
  };
}
