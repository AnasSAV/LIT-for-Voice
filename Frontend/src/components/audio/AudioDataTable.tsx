import { useState, useEffect, useMemo, useCallback } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  createColumnHelper,
  flexRender,
  ColumnDef,
} from "@tanstack/react-table";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Play, Pause, ChevronLeft, ChevronRight } from "lucide-react";
import { listDatasetFiles, datasetFileUrl } from "@/lib/api/datasets";
import { cn } from "@/lib/utils";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
  prediction?: {
    text?: string;
    label?: string;
    confidence?: number;
  };
}

interface AudioData {
  id: string;
  filename: string;
  relpath: string;
  predictedTranscript: string;
  predictedLabel: string;
  prediction?: {
    text?: string;
    label?: string;
    confidence?: number;
  };
  groundTruthLabel: string;
  confidence: number;
  duration: number;
  file_path?: string;
  size?: number;
}

interface ApiData {
  prediction: {
    text: string;
    label?: string;
    confidence?: number;
  };
}

interface AudioDataTableProps {
  selectedRow: string | null;
  onRowSelect: (id: string) => void;
  searchQuery: string;
  apiData: ApiData | null;
  uploadedFiles?: UploadedFile[];
  onFilePlay?: (file: UploadedFile) => void;
}

async function fetchRows(): Promise<AudioData[]> {
  const { files } = await listDatasetFiles(200, 0);
  return files.map((f) => ({
    id: f.id,
    filename: f.filename,
    relpath: f.relpath,
    predictedTranscript: "",
    predictedLabel: "",
    groundTruthLabel: "",
    confidence: 0,
    duration: 0,
  }));
}

export const AudioDataTable = ({
  selectedRow,
  onRowSelect,
  searchQuery,
  apiData,
  uploadedFiles = [],
  onFilePlay,
}: AudioDataTableProps) => {
  const [data, setData] = useState<AudioData[]>([]);
  const [loading, setLoading] = useState(false);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [audioEl, setAudioEl] = useState<HTMLAudioElement | null>(null);
  const columnHelper = createColumnHelper<AudioData>();

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
  }, [audioEl]);

  // Toggle audio playback for a row
  const togglePlay = useCallback((row: AudioData) => {
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
      setAudioEl(el);
      setPlayingId(row.id);
      void el.play();
    } catch (e) {
      console.warn("Failed to play audio", e);
    }
  }, [playingId, audioEl]);

  // Combine API data with uploaded files
  const tableData = useMemo(() => {
    const uploadedData = (uploadedFiles || []).map((file) => ({
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
      size: file.size,
    }));

    return [...data, ...uploadedData];
  }, [data, uploadedFiles]);

  // Define table columns
  const columns = useMemo<ColumnDef<AudioData>[]>(
    () => [
      {
        accessorKey: "filename",
        header: "Filename",
        cell: (info) => (
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              className="h-6 w-6 p-0"
              onClick={(e) => {
                e.stopPropagation();
                togglePlay(info.row.original);
              }}
              aria-label={playingId === info.row.original.id ? "Pause" : "Play"}
            >
              {playingId === info.row.original.id ? (
                <Pause className="h-3 w-3" />
              ) : (
                <Play className="h-3 w-3" />
              )}
            </Button>
            <span className="font-mono text-xs">{info.getValue() as string}</span>
          </div>
        ),
      },
      {
        accessorKey: "predictedTranscript",
        header: "Predicted Transcript",
        cell: (info) => (
          <span className="text-xs">{info.getValue() as string}</span>
        ),
      },
      {
        accessorKey: "predictedLabel",
        header: "Predicted Label",
        cell: (info) => (
          <Badge variant="outline" className="text-xs">
            {info.getValue() as string || "N/A"}
          </Badge>
        ),
      },
      {
        accessorKey: "groundTruthLabel",
        header: "Ground Truth",
        cell: (info) => (
          <Badge variant="outline" className="text-xs">
            {info.getValue() as string || "N/A"}
          </Badge>
        ),
      },
      {
        accessorKey: "confidence",
        header: "Confidence",
        cell: (info) => {
          const value = info.getValue() as number;
          return <span className="text-xs">{(value * 100).toFixed(1)}%</span>;
        },
      },
      {
        accessorKey: "duration",
        header: "Duration",
        cell: (info) => {
          const value = info.getValue() as number;
          return <span className="text-xs">{value.toFixed(2)}s</span>;
        },
      },
    ],
    [playingId]
  );

  // Initialize table
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
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <TableHead key={header.id}>
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
          {table.getRowModel().rows?.length ? (
            table.getRowModel().rows.map((row) => (
              <TableRow
                key={row.id}
                data-state={selectedRow === row.original.id ? "selected" : undefined}
                onClick={() => onRowSelect(row.original.id)}
                className={cn(
                  "cursor-pointer hover:bg-muted/50",
                  selectedRow === row.original.id && "bg-muted"
                )}
              >
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))
          ) : (
            <TableRow>
              <TableCell colSpan={columns.length} className="h-24 text-center">
                {loading ? "Loading..." : "No results."}
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
      <div className="flex items-center justify-end space-x-2 p-4">
        <div className="flex-1 text-sm text-muted-foreground">
          {table.getFilteredSelectedRowModel().rows.length} of{" "}
          {table.getFilteredRowModel().rows.length} row(s) selected.
        </div>
        <div className="space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};
