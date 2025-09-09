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
}

export const AudioDataTable = ({ selectedRow, onRowSelect, searchQuery, apiData, model, dataset, datasetMetadata, uploadedFiles, onFilePlay }: AudioDataTableProps) => {
  // Branch: dataset mode vs custom uploads
  const isDatasetMode = dataset !== "custom" && (datasetMetadata?.length || 0) > 0;

  // Custom uploads data and columns
  const customTableData: AudioData[] = (uploadedFiles?.map(file => ({
    id: file.file_id,
    filename: file.filename,
    prediction: file.prediction,
    groundTruthLabel: "",
    confidence: 0.85,
    duration: file.duration || 0,
    file_path: file.file_path,
    size: file.size
  })) || []);

  const customColumns: ColumnDef<unknown, unknown>[] = [
    {
      id: "filename",
      header: "Filename",
      cell: ({ row }) => {
        const data = row.original as AudioData;
        const file = uploadedFiles?.find(f => f.file_id === data.id);
        return (
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              className="h-6 w-6 p-0"
              onClick={() => file && onFilePlay && onFilePlay(file)}
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
        return (
          <Badge variant="outline" className="text-xs">
            {data.prediction}
          </Badge>
        );
      },
    },
    {
      id: "groundTruthLabel",
      header: "Ground Truth",
      cell: ({ row }) => {
        const data = row.original as AudioData;
        return (
          <Badge variant="secondary" className="text-xs">
            {data.groundTruthLabel}
          </Badge>
        );
      },
    },
    {
      id: "confidence",
      header: "Confidence",
      cell: ({ row }) => {
        const data = row.original as AudioData;
        return <span className="text-xs">{data.confidence}%</span>;
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
  ];

  // Dataset metadata data and columns
  type DatasetRow = Record<string, string | number | null | undefined>;
  const datasetRows: DatasetRow[] = (datasetMetadata || []).map((r) => r);

  const getFrom = (row: DatasetRow, keys: string[], fallback = ""): string => {
    for (const k of keys) {
      const v = row[k];
      if (v !== undefined && v !== null && String(v).length > 0) return String(v);
    }
    return fallback;
  };

  const getDatasetRowId = (row: DatasetRow, fallback: string): string => {
    const v = row["id"] ?? row["path"] ?? row["filepath"] ?? row["file"] ?? row["filename"];
    return v !== undefined && v !== null && String(v).length > 0 ? String(v) : fallback;
  };

  const datasetColumnsCommonVoice: ColumnDef<unknown, unknown>[] = [
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
      id: "sentence",
      header: "Transcript",
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
  ];

  const datasetColumnsRavdess: ColumnDef<unknown, unknown>[] = [
    {
      id: "filename",
      header: "Filename",
      cell: ({ row }) => {
        const data = row.original as DatasetRow;
        const filename = getFrom(data, ["path", "filepath", "file", "filename"], "");
        const displayName = filename.split("/").pop() || filename;
        return (
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              className="h-6 w-6 p-0"
              onClick={async (e) => {
                e.preventDefault();
                e.stopPropagation();
                try {
                  // Run prediction on this dataset file
                  const response = await fetch(`http://localhost:8000/predictions/single?model=${model}&dataset=${dataset}&filename=${displayName}`);
                  if (response.ok) {
                    const result = await response.json();
                    console.log('Prediction result:', result);
                    // You could show this in a modal or update the UI
                    alert(`Prediction: ${result.emotion_prediction}\nTranscript: ${result.transcript}\nGround Truth: ${result.ground_truth_emotion}`);
                  } else {
                    throw new Error(`HTTP ${response.status}`);
                  }
                } catch (error) {
                  console.error('Prediction failed:', error);
                  alert(`Prediction failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
                }
              }}
            >
              <Play className="h-3 w-3" />
            </Button>
            <span className="font-mono text-xs">{displayName}</span>
          </div>
        );
      },
    },
    {
      id: "emotion",
      header: "Ground Truth Emotion",
      cell: ({ row }) => {
        const data = row.original as DatasetRow;
        return <Badge variant="secondary" className="text-xs">{getFrom(data, ["emotion", "label"], "")}</Badge>;
      },
    },
    {
      id: "transcript",
      header: "Ground Truth Transcript",
      cell: ({ row }) => {
        const data = row.original as DatasetRow;
        return <span className="text-xs">{getFrom(data, ["statement", "sentence"], "")}</span>;
      },
    },
    {
      id: "intensity",
      header: "Intensity",
      cell: ({ row }) => {
        const data = row.original as DatasetRow;
        return <Badge variant="outline" className="text-xs">{getFrom(data, ["intensity"], "")}</Badge>;
      },
    },
    {
      id: "actor",
      header: "Actor",
      cell: ({ row }) => {
        const data = row.original as DatasetRow;
        return <span className="text-xs">{getFrom(data, ["actor"], "")}</span>;
      },
    },
    {
      id: "gender",
      header: "Gender",
      cell: ({ row }) => {
        const data = row.original as DatasetRow;
        return <Badge variant="outline" className="text-xs">{getFrom(data, ["gender"], "")}</Badge>;
      },
    },
  ];

  // Build table config based on mode
  const data: unknown[] = isDatasetMode ? datasetRows : customTableData;
  const columns: ColumnDef<unknown, unknown>[] = isDatasetMode ? (dataset === "ravdess" ? datasetColumnsRavdess : datasetColumnsCommonVoice) : customColumns;

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
                  className="cursor-pointer hover:bg-muted/50"
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