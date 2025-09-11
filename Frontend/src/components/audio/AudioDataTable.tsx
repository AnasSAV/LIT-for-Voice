import {
  useReactTable,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  flexRender,
  ColumnDef,
} from "@tanstack/react-table";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Play, ChevronLeft, ChevronRight } from "lucide-react";
import { useMemo, useCallback, useEffect } from "react";

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
  prediction?: string;
  groundTruthLabel: string;
  confidence: number;
  duration: number;
  file_path?: string;
  size?: number;
}

interface AudioDataTableProps {
  selectedRow: string | null;
  onRowSelect: (id: string) => void;
  searchQuery: string;
  apiData?: unknown;
  model: string;
  dataset: string; // "custom" | "common-voice" | "ravdess"
  datasetMetadata?: Record<string, string | number>[];
  uploadedFiles?: UploadedFile[];
  onFilePlay?: (file: UploadedFile) => void;
  predictionMap?: Record<string, string>;
  inferenceStatus?: Record<string, 'idle' | 'loading' | 'done' | 'error'>;
  onVisibleRowIdsChange?: (rowIds: string[]) => void;
}

export const AudioDataTable = ({ selectedRow, onRowSelect, searchQuery, apiData, model, dataset, datasetMetadata, uploadedFiles, onFilePlay, predictionMap, inferenceStatus, onVisibleRowIdsChange }: AudioDataTableProps) => {
  // Branch: dataset mode vs custom uploads
  const isDatasetMode = dataset !== "custom" && (datasetMetadata?.length || 0) > 0;

  // Custom uploads data and columns
  const customTableData: AudioData[] = useMemo(() => (
    uploadedFiles?.map(file => ({
      id: file.file_id,
      filename: file.filename,
      prediction: file.prediction,
      groundTruthLabel: "",
      confidence: 0,
      duration: file.duration || 0,
      file_path: file.file_path,
      size: file.size
    })) || []
  ), [uploadedFiles]);

  const customColumns: ColumnDef<unknown, unknown>[] = useMemo(() => [
    {
      id: "filename",
      header: "Filename",
      cell: ({ row }) => {
        const data = row.original as AudioData;
        const file = uploadedFiles?.find(f => f.file_id === data.id);
        return (
          <div className="flex items-center gap-2">
            <Button
              type="button"
              size="sm"
              variant="ghost"
              className="h-6 w-6 p-0"
              onClick={(e) => {
                e.stopPropagation();
                e.preventDefault();
                if (file && onFilePlay) onFilePlay(file);
              }}
            >
              <Play className="h-3 w-3" />
            </Button>
            <span className="font-mono text-xs">{data.filename}</span>
          </div>
        );
      },
    },
    {
      id: "prediction",
      header: model.startsWith("whisper") ? "Predicted Transcript" : "Predicted Label",
      cell: ({ row }) => {
        const data = row.original as AudioData;
        const rowId = row.id as string;
        const status = inferenceStatus?.[rowId];
        if (status === 'loading') {
          return "";
        }
        const pred = predictionMap?.[rowId] ?? data.prediction ?? "";
        if (!pred) return "";
        return <Badge variant="outline" className="text-xs">{pred}</Badge>;
      },
    },
    {
      id: "groundTruthLabel",
      header: "Ground Truth",
      cell: ({ row }) => {
        const data = row.original as AudioData;
        return <span className="text-xs">{data.groundTruthLabel}</span>;
      },
    },
    {
      id: "confidence",
      header: "Confidence",
      cell: ({ row }) => {
        const data = row.original as AudioData;
        // Don't display confidence if it's 0
        if (data.confidence === 0) return null;
        return <span className="text-xs">{data.confidence}</span>;
      },
    },
    {
      id: "duration",
      header: "Duration",
      cell: ({ row }) => {
        const data = row.original as AudioData;
        return <span className="text-xs">{data.duration.toFixed(2)}s</span>;
      },
    },
  ], [model, uploadedFiles, onFilePlay, predictionMap, inferenceStatus]);

  // Dataset metadata data and columns
  type DatasetRow = Record<string, string | number | null | undefined>;
  const datasetRows: DatasetRow[] = useMemo(() => datasetMetadata ?? [], [datasetMetadata]);

  const getFrom = useCallback((row: DatasetRow, keys: string[], fallback = ""): string => {
    for (const k of keys) {
      const v = row[k];
      if (v !== undefined && v !== null && String(v).length > 0) return String(v);
    }
    return fallback;
  }, []);

  const getDatasetRowId = useCallback((row: DatasetRow, fallback: string): string => {
    const v = row["id"] ?? row["path"] ?? row["filepath"] ?? row["file"] ?? row["filename"];
    return v !== undefined && v !== null && String(v).length > 0 ? String(v) : fallback;
  }, []);

  const datasetColumnsCommonVoice: ColumnDef<unknown, unknown>[] = useMemo(() => [
    {
      id: "filename",
      header: "Filename",
      cell: ({ row }) => {
        const data = row.original as DatasetRow;
        const path = getFrom(data, ["path", "filepath", "file", "filename"], "");
        const filename = path.split("/").pop() || path;
        return <span className="font-mono text-xs">{filename}</span>;
      },
    },
    {
      id: "prediction",
      header: model.startsWith("whisper") ? "Predicted Transcript" : "Predicted Label",
      cell: ({ row }) => {
        const rowId = row.id as string;
        const status = inferenceStatus?.[rowId];
        if (status !== 'done') {
          return <span className="text-xs"></span>;
        }
        const pred = predictionMap?.[rowId] ?? "";
        return <span className="text-xs">{pred}</span>;
      },
    },
    {
      id: "ground_truth",
      header: "Ground Truth",
      cell: ({ row }) => {
        const data = row.original as DatasetRow;
        return <span className="text-xs">{getFrom(data, ["sentence", "transcript", "text"], "")}</span>;
      },
    },
    {
      id: "duration",
      header: "Duration",
      cell: ({ row }) => {
        const data = row.original as DatasetRow;
        const d = Number(getFrom(data, ["duration"], "0"));
        return <span className="text-xs">{isNaN(d) ? "" : `${d.toFixed(2)}s`}</span>;
      },
    },
  ], [getFrom, model, predictionMap, inferenceStatus]);

  const datasetColumnsRavdess: ColumnDef<unknown, unknown>[] = useMemo(() => [
    {
      id: "filename",
      header: "Filename",
      cell: ({ row }) => {
        const data = row.original as DatasetRow;
        const path = getFrom(data, ["path", "filepath", "file", "filename"], "");
        const filename = path.split("/").pop() || path;
        return <span className="font-mono text-xs">{filename}</span>;
      },
    },
    {
      id: "prediction",
      header: model.startsWith("whisper") ? "Predicted Transcript" : "Predicted Label",
      cell: ({ row }) => {
        const rowId = row.id as string;
        const status = inferenceStatus?.[rowId];
        if (status !== 'done') {
          return <span className="text-xs"></span>;
        }
        const pred = predictionMap?.[rowId] ?? "";
        return <span className="text-xs">{pred}</span>;
      },
    },
    {
      id: "ground_truth",
      header: "Ground Truth",
      cell: ({ row }) => {
        const data = row.original as DatasetRow;
        return <span className="text-xs">{getFrom(data, ["emotion", "label"], "")}</span>;
      },
    },
    {
      id: "duration",
      header: "Duration",
      cell: ({ row }) => {
        const data = row.original as DatasetRow;
        // Ravdess dataset doesn't have duration in metadata, so we'll show "N/A"
        const d = Number(getFrom(data, ["duration", "length"], "0"));
        if (d > 0) {
          return <span className="text-xs">{d.toFixed(2)}s</span>;
        }
        return <span className="text-xs text-muted-foreground">N/A</span>;
      },
    },
  ], [getFrom, model, predictionMap, inferenceStatus]);

  // Build table config based on mode
  const data: unknown[] = useMemo(() => (isDatasetMode ? datasetRows : customTableData), [isDatasetMode, datasetRows, customTableData]);
  const columns: ColumnDef<unknown, unknown>[] = useMemo(
    () => (isDatasetMode ? (dataset === "ravdess" ? datasetColumnsRavdess : datasetColumnsCommonVoice) : customColumns),
    [isDatasetMode, dataset, datasetColumnsRavdess, datasetColumnsCommonVoice, customColumns]
  );

  const getRowId = useCallback((row: unknown, index?: number) => {
    if (isDatasetMode) {
      return getDatasetRowId(row as DatasetRow, String(index ?? ""));
    }
    return (row as AudioData).id;
  }, [isDatasetMode, getDatasetRowId]);

  const table = useReactTable<unknown>({
    data,
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
    getRowId,
  });

  // Notify parent of currently visible row ids (for sequential per-page inference)
  useEffect(() => {
    if (!onVisibleRowIdsChange) return;
    const rows = table.getRowModel().rows;
    const ids = rows.map(r => String(r.id));
    onVisibleRowIdsChange(ids);
  }, [onVisibleRowIdsChange, searchQuery, dataset, model, isDatasetMode]);

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
            {table.getRowModel().rows?.length ? (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  data-state={(() => {
                    const rowId: string = isDatasetMode
                      ? getDatasetRowId(row.original as DatasetRow, String(row.id))
                      : (row.original as AudioData).id;
                    return selectedRow === rowId ? "selected" : undefined;
                  })()}
                  className="cursor-pointer hover:bg-muted/50 data-[state=selected]:bg-blue-50 data-[state=selected]:border-blue-200 data-[state=selected]:shadow-sm"
                  onClick={() => {
                    const rowId: string = isDatasetMode
                      ? getDatasetRowId(row.original as DatasetRow, String(row.id))
                      : (row.original as AudioData).id;
                    onRowSelect(rowId);
                  }}
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
                <TableCell colSpan={Array.isArray(columns) ? columns.length : 0} className="h-24 text-center text-xs">
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