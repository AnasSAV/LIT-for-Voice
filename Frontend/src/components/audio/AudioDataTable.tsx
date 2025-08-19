import { useState, useEffect } from "react";

import {
  useReactTable,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  createColumnHelper,
  flexRender,
} from "@tanstack/react-table";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Play, Pause, ChevronLeft, ChevronRight } from "lucide-react";
import { listDatasetFiles, datasetFileUrl } from "@/lib/api/datasets";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
  prediction?:string;
}

interface AudioData {
  id: string;
  filename: string;
  relpath: string;
  predictedTranscript: string;
  predictedLabel: string;
  prediction?: any;
  groundTruthLabel: string;
  confidence: number;
  duration: number;
  file_path?: string;
  size?: number;
}

async function fetchRows(): Promise<AudioData[]> {
  const { files } = await listDatasetFiles(200, 0);
  return files.map((f) => ({
    id: f.id,
    filename: f.filename,
    relpath: f.relpath,
    // Placeholders until inference results are available
    predictedTranscript: "",
    predictedLabel: "",
    groundTruthLabel: "",
    confidence: 0,
    duration: 0,
  }));
}

const columnHelper = createColumnHelper<AudioData>();
interface ApiData {
  prediction: {
    text: string;
  };
}
interface AudioDataTableProps {
  selectedRow: string | null;
  onRowSelect: (id: string) => void;
  searchQuery: string;
  apiData: ApiData;
  uploadedFiles?: UploadedFile[];
  onFilePlay?: (file: UploadedFile) => void;
}

<<<<<<< HEAD
export const AudioDataTable = ({ selectedRow, onRowSelect, searchQuery }: AudioDataTableProps) => {
  const [data, setData] = useState<AudioData[]>([]);
  const [loading, setLoading] = useState(false);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [audioEl, setAudioEl] = useState<HTMLAudioElement | null>(null);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      setLoading(true);
      try {
        const rows = await fetchRows();
        if (mounted) setData(rows);
      } catch (e) {
        console.warn("Failed to load dataset files", e);
        if (mounted) setData([]);
      } finally {
        if (mounted) setLoading(false);
      }
    };

    load();

    const onChanged = () => load();
    window.addEventListener("dataset-changed", onChanged);
    return () => {
      mounted = false;
      window.removeEventListener("dataset-changed", onChanged);
      if (audioEl) {
        audioEl.pause();
      }
    };
  }, []);

  const togglePlay = (row: AudioData) => {
    try {
      // If clicking the currently playing row, toggle pause
      if (playingId === row.id && audioEl) {
        if (!audioEl.paused) {
          audioEl.pause();
          setPlayingId(null);
        } else {
          audioEl.play();
          setPlayingId(row.id);
        }
        return;
      }

      // Stop any previous audio
      if (audioEl) {
        audioEl.pause();
      }

      const src = datasetFileUrl(row.relpath);
      const el = new Audio(src);
      el.addEventListener("ended", () => setPlayingId(null));
      el.addEventListener("pause", () => {
        // If paused for a different row play, ignore
      });
      setAudioEl(el);
      setPlayingId(row.id);
      void el.play();
    } catch (e) {
      console.warn("Failed to play audio", e);
    }
  };

  const columns = [
    columnHelper.accessor("filename", {
      header: "Filename",
      cell: (info) => (
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 p-0"
            onClick={(e) => {
              e.stopPropagation();
              togglePlay(info.row.original as AudioData);
            }}
            aria-label={playingId === info.row.original.id ? "Pause" : "Play"}
          >
            {playingId === info.row.original.id ? (
              <Pause className="h-3 w-3" />
            ) : (
              <Play className="h-3 w-3" />
            )}
          </Button>
          <span className="font-mono text-xs">{info.getValue()}</span>
        </div>
      ),
=======
export const AudioDataTable = ({
  selectedRow,
  onRowSelect,
  searchQuery,
  apiData,
  uploadedFiles = [],
  onFilePlay
}: AudioDataTableProps) => {
  const [data, setData] = useState<AudioData[]>([]);
  const [loading, setLoading] = useState(false);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [audioEl, setAudioEl] = useState<HTMLAudioElement | null>(null);

  // Load data from API on component mount
  useEffect(() => {
    let mounted = true;
    const load = async () => {
      setLoading(true);
      try {
        const rows = await fetchRows();
        if (mounted) setData(rows);
      } catch (e) {
        console.warn("Failed to load dataset files", e);
        if (mounted) setData([]);
      } finally {
        if (mounted) setLoading(false);
      }
    };

    load();

    const onChanged = () => load();
    window.addEventListener("dataset-changed", onChanged);
    return () => {
      mounted = false;
      window.removeEventListener("dataset-changed", onChanged);
      if (audioEl) {
        audioEl.pause();
      }
    };
  }, []);

  // Combine API data with uploaded files
  const tableData = useMemo(() => {
    const uploadedData = uploadedFiles.map(file => ({
      id: file.file_id,
      filename: file.filename,
      relpath: file.file_path,
      predictedTranscript: file.prediction?.text || "",
      predictedLabel: file.prediction?.label || "",
      prediction: file.prediction,
      groundTruthLabel: "",
      confidence: file.prediction?.confidence || 0,
      duration: file.duration || 0,
      file_path: file.file_path,
      size: file.size
    }));
    
    // Merge with existing data, giving priority to uploaded files
    const existingIds = new Set(uploadedData.map(d => d.id));
    const mergedData = [...uploadedData];
    
    for (const item of data) {
      if (!existingIds.has(item.id)) {
        mergedData.push(item);
      }
    }
    
    return mergedData;
  }, [data, uploadedFiles]);

  const togglePlay = (row: AudioData) => {
    // If we have an onFilePlay handler, use that for uploaded files
    if (onFilePlay && row.file_path) {
      const file = uploadedFiles?.find(f => f.file_id === row.id);
      if (file) {
        onFilePlay(file);
        setPlayingId(playingId === row.id ? null : row.id);
        return;
      }
    }

    // Fallback to direct audio playback for non-uploaded files
    try {
      if (playingId === row.id && audioEl) {
        if (!audioEl.paused) {
          audioEl.pause();
          setPlayingId(null);
        } else {
          audioEl.play();
          setPlayingId(row.id);
        }
        return;
      }

      if (audioEl) {
        audioEl.pause();
      }

      const src = datasetFileUrl(row.relpath);
      const el = new Audio(src);
      el.addEventListener("ended", () => setPlayingId(null));
      setAudioEl(el);
      setPlayingId(row.id);
      void el.play();
    } catch (e) {
      console.warn("Failed to play audio", e);
    }
  };

  const columns = [
    columnHelper.accessor("filename", {
      header: "Filename",
      cell: (info) => {
        const row = info.row.original;
        const isPlaying = playingId === row.id;
        
        return (
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              className="h-6 w-6 p-0"
              onClick={(e) => {
                e.stopPropagation();
                togglePlay(row);
              }}
              aria-label={isPlaying ? "Pause" : "Play"}
            >
              {isPlaying ? (
                <Pause className="h-3 w-3" />
              ) : (
                <Play className="h-3 w-3" />
              )}
            </Button>
            <span className="font-mono text-xs">{info.getValue()}</span>
          </div>
        );
      },
    }),
    // columnHelper.accessor("predictedTranscript", {
    //   header: "Predicted Transcript",
    //   cell: (info) => (
    //     <span className="text-xs">{info.getValue()}</span>
    //   ),
    // }),
    columnHelper.accessor("prediction", {
      header: "Predicted Label",
      cell: (info) => (
        <Badge variant="outline" className="text-xs">
          {info.getValue()}
        </Badge>
      ),
    }),
    columnHelper.accessor("groundTruthLabel", {
      header: "Ground Truth",
      cell: (info) => (
        <Badge variant="secondary" className="text-xs">
          {info.getValue()}
        </Badge>
      ),
    }),
    columnHelper.accessor("confidence", {
      header: "Confidence",
      cell: (info) => (
        <span className="text-xs">{info.getValue()}%</span>
      ),
    }),
    columnHelper.accessor("duration", {
      header: "Duration",
      cell: (info) => {
        const value = info.getValue() as number; // get numeric value
        return <span className="text-xs">{value.toFixed(2)}s</span>;
      },
    }),
  ];

  const table = useReactTable({
    data: tableData,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    state: {
      globalFilter: searchQuery,
    },
    onGlobalFilterChange: () => {},
    initialState: {
      pagination: {
        pageSize: 20,
      },
    },
  });

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 overflow-auto">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead key={header.id} className="h-8 text-xs">
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={columns.length} className="h-24 text-center text-xs">
                  Loading...
                </TableCell>
              </TableRow>
            ) : table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  data-state={selectedRow === row.original.id ? "selected" : undefined}
                  className="cursor-pointer hover:bg-muted/50"
                  onClick={() => onRowSelect(row.original.id)}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id} className="py-2">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={columns.length} className="h-24 text-center text-xs">
                  No results.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
      
      {/* Pagination */}
      <div className="border-t panel-border p-2 flex items-center justify-between">
        <div className="text-xs text-muted-foreground">
          Page {table.getState().pagination.pageIndex + 1} of {table.getPageCount()}
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            className="h-6 w-6 p-0"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            <ChevronLeft className="h-3 w-3" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-6 w-6 p-0"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            <ChevronRight className="h-3 w-3" />
          </Button>
        </div>
      </div>
    </div>
  );
};