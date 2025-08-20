import { useState, useEffect, useMemo, useCallback, useRef } from "react";
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
import { Play, ChevronLeft, ChevronRight } from "lucide-react";
import { listDatasetFiles } from "@/lib/api/datasets";
import { cn } from "@/lib/utils";
import { runInference, type InferenceResponse } from "@/lib/api/inferences";
import { toast } from "sonner";

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
    probs?: Record<string, number>;
  };
  label?: string;
  dataset_id?: string | null;
  autoplay?: boolean;
  meta?: Record<string, string>;
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
    probs?: Record<string, number>;
  };
  groundTruthLabel: string;
  confidence: number;
  duration: number;
  sample_rate?: number;
  file_path?: string;
  size?: number;
  // Optional metadata from backend (RAVDESS: emotion, intensity, statement, repetition, actor, gender)
  meta?: Record<string, string>;
  // Dataset-only helpers
  hash?: string;
  dataset_id?: string | null;
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
  model?: string | null;
}

async function fetchRowsWithActive(): Promise<{ rows: AudioData[]; active: string | null }> {
  const { files, active } = await listDatasetFiles(200, 0);
  const rows = files.map((f) => ({
    id: f.id,
    filename: f.filename,
    relpath: f.relpath,
    predictedTranscript: "",
    predictedLabel: "",
    groundTruthLabel: f.label || f.meta?.emotion || "",
    confidence: 0,
    duration: f.duration || 0,
    sample_rate: f.sample_rate,
    size: f.size,
    meta: f.meta,
    hash: f.h,
    dataset_id: active,
  }));
  return { rows, active };
}

export const AudioDataTable = ({
  selectedRow,
  onRowSelect,
  searchQuery,
  apiData,
  uploadedFiles = [],
  onFilePlay,
  model,
}: AudioDataTableProps) => {
  const [data, setData] = useState<AudioData[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeDatasetId, setActiveDatasetId] = useState<string | null>(null);
  const [uiDataset, setUiDataset] = useState<string | null>(null); // 'ravdess' | 'common-voice' | 'custom'
  const columnHelper = createColumnHelper<AudioData>();
  const [loadingById, setLoadingById] = useState<Record<string, boolean>>({});
  const loadingByIdRef = useRef<Record<string, boolean>>({});
  // Track previously active dataset to decide when to reset pagination
  const prevActiveDatasetRef = useRef<string | null>(null);
  useEffect(() => {
    loadingByIdRef.current = loadingById;
  }, [loadingById]);
  // Controlled pagination so we can react to page changes
  const [pagination, setPagination] = useState({ pageIndex: 0, pageSize: 20 });

  // Load data from API on component mount and when dataset changes
  useEffect(() => {
    let mounted = true;
    const load = async () => {
      setLoading(true);
      try {
        // If UI dataset is custom, do not render any server dataset
        if (uiDataset === 'custom') {
          if (mounted) {
            setData([]);
            setActiveDatasetId(null);
          }
          return;
        }
        const { rows, active } = await fetchRowsWithActive();
        if (mounted) {
          // Gate RAVDESS visibility: only show when model is wav2vec2 (subset only)
          const isRavdessActive = active === "ravdess_subset";
          if (isRavdessActive && model !== "wav2vec2") {
            setData([]);
          } else {
            setData(rows);
          }
          // Detect dataset change to avoid unwanted page resets during rerenders
          const datasetChanged = prevActiveDatasetRef.current !== active;
          setActiveDatasetId(active);
          if (datasetChanged) {
            // Remember new dataset id
            prevActiveDatasetRef.current = active;
            // Only reset to first page when dataset actually changes
            setPagination((prev) => ({ ...prev, pageIndex: 0 }));
          }
        }
      } catch (e) {
        console.warn("Failed to load dataset files", e);
        if (mounted) setData([]);
      } finally {
        if (mounted) setLoading(false);
      }
    };

    load();

    const onChanged = (ev: Event) => {
      // Update UI dataset hint from toolbar if provided
      const detail = (ev as CustomEvent).detail as { uiDataset?: string } | undefined;
      if (detail && typeof detail.uiDataset === 'string') {
        setUiDataset(detail.uiDataset);
      }
      load();
    };
    window.addEventListener("dataset-changed", onChanged as EventListener);
    
    return () => {
      mounted = false;
      window.removeEventListener("dataset-changed", onChanged as EventListener);
    };
  }, [model, uiDataset]);

  // Trigger prediction for a specific row (sequential per click, limited via API helper)
  const handlePredict = useCallback(async (row: AudioData) => {
    if (!(model === "wav2vec2" || model === "whisper-base" || model === "whisper-large")) return;
    const id = row.id;
    if (loadingByIdRef.current[id]) return;
    // Build params: prefer dataset caching (ds_id + h)
    const params: { ds_id?: string | null; h?: string; file_path?: string } = {};
    if (row.hash && activeDatasetId) {
      params.ds_id = activeDatasetId;
      params.h = row.hash;
    } else if (row.file_path) {
      params.file_path = row.file_path;
    } else {
      return;
    }

    setLoadingById((prev) => ({ ...prev, [id]: true }));
    try {
      if (model === "wav2vec2") {
        const res = await runInference("wav2vec2", params) as InferenceResponse;
        const label = res?.label ? String(res.label) : "";
        const confidence = typeof res?.confidence === "number" ? Number(res.confidence) : 0;
        setData((prev) => prev.map((r) => (
          r.id === id
            ? {
                ...r,
                predictedLabel: label,
                confidence,
                prediction: { ...(r.prediction || {}), label, confidence, probs: res?.probs },
              }
            : r
        )));
      } else if (model === "whisper-base") {
        const res = await runInference("whisper-base", params) as unknown;
        const text = typeof res === "string" ? res : ((res as InferenceResponse)?.text ? String((res as InferenceResponse).text) : "");
        setData((prev) => prev.map((r) => (
          r.id === id
            ? {
                ...r,
                predictedTranscript: text,
                prediction: { ...(r.prediction || {}), text },
              }
            : r
        )));
      } else if (model === "whisper-large") {
        const res = await runInference("whisper-large", params) as unknown;
        const text = typeof res === "string" ? res : ((res as InferenceResponse)?.text ? String((res as InferenceResponse).text) : "");
        setData((prev) => prev.map((r) => (
          r.id === id
            ? {
                ...r,
                predictedTranscript: text,
                prediction: { ...(r.prediction || {}), text },
              }
            : r
        )));
      }
    } catch (e: unknown) {
      console.warn("Prediction failed", e);
      let message = "Prediction failed";
      if (e instanceof Error) {
        message = e.message;
      } else if (typeof e === "string") {
        message = e;
      } else if (e && typeof e === "object" && "message" in e) {
        const maybe = (e as { message?: unknown }).message;
        if (typeof maybe === "string") message = maybe;
      }
      toast.error(message);
    } finally {
      setLoadingById((prev) => ({ ...prev, [id]: false }));
    }
  }, [model, activeDatasetId]);

  

  // Combine API data with uploaded files
  const tableData = useMemo(() => {
    const uploadedData: AudioData[] = (uploadedFiles || []).map((file) => ({
      id: file.file_id,
      filename: file.filename,
      relpath: file.file_path,
      predictedTranscript: file.prediction?.text || "",
      predictedLabel: file.prediction?.label || "",
      prediction: file.prediction,
      groundTruthLabel: "",
      confidence: file.prediction?.confidence || 0,
      duration: file.duration || 0,
      sample_rate: file.sample_rate,
      file_path: file.file_path,
      size: file.size,
      meta: file.meta,
      hash: undefined,
      dataset_id: null,
    }));

    return [...data, ...uploadedData];
  }, [data, uploadedFiles]);

  // Define table columns (hide Predicted Label for Whisper models)
  const columns = useMemo<ColumnDef<AudioData>[]>(() => {
    const cols: ColumnDef<AudioData>[] = [
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
                // Ensure selection and right panel update when playing via button
                const orig = info.row.original as AudioData;
                onRowSelect(orig.id);
                if (onFilePlay) {
                  onFilePlay({
                    file_id: orig.id,
                    filename: orig.filename,
                    file_path: orig.relpath,
                    message: "dataset",
                    size: orig.size,
                    duration: orig.duration,
                    sample_rate: orig.sample_rate,
                    prediction: orig.prediction,
                    label: orig.groundTruthLabel,
                    dataset_id: activeDatasetId,
                    autoplay: true,
                    meta: orig.meta,
                  });
                }
                // Trigger prediction for current model
                void handlePredict(orig);
              }}
              aria-label={"Play"}
            >
              <Play className="h-3 w-3" />
            </Button>
            <span className="font-mono text-xs">{info.getValue() as string}</span>
          </div>
        ),
      },
    ];

    const isWhisper = (model || "").toLowerCase().includes("whisper");
    const isWav2Vec2 = model === "wav2vec2";

    // For Whisper models: show Predicted Transcript column
    if (isWhisper) {
      cols.push({
        accessorKey: "predictedTranscript",
        header: "Predicted Transcript",
        cell: (info) => {
          const text = (info.getValue() as string) || "N/A";
          return (
            <p className="inline-block max-w-full whitespace-pre-wrap break-words rounded-2xl bg-muted px-3 py-1.5 text-xs leading-snug text-foreground shadow-sm">
              {text}
            </p>
          );
        },
      });
    }

    if (!isWhisper) {
      cols.push({
        accessorKey: "predictedLabel",
        header: "Predicted Label",
        cell: (info) => (
          <Badge variant="default" className="text-xs">
            {loadingById[info.row.original.id] ? "…" : (info.getValue() as string || "N/A")}
          </Badge>
        ),
      });
    }

    // Ground Truth always before static metadata
    cols.push({
      accessorKey: "groundTruthLabel",
      header: "Ground Truth",
      cell: (info) => {
        const orig = info.row.original;
        const gt = isWhisper
          ? ((orig.meta?.text as string) || (orig.meta?.statement as string) || "N/A")
          : ((info.getValue() as string) || "N/A");
        if (isWhisper) {
          return (
            <Badge variant="outline" className="text-xs rounded-full px-3 py-1 normal-case font-normal">
              {gt}
            </Badge>
          );
        }
        return (
          <Badge variant="secondary" className="text-xs">
            {gt}
          </Badge>
        );
      },
    });

    // For wav2vec2: show static metadata columns: Intensity, Gender, Actor (placed between Ground Truth and Confidence)
    if (isWav2Vec2) {
      cols.push(
        {
          id: "meta_intensity",
          header: "Intensity",
          cell: (info) => (
            <Badge variant="outline" className="text-xs">
              {(info.row.original.meta?.intensity as string) || "N/A"}
            </Badge>
          ),
        },
        {
          id: "meta_gender",
          header: "Gender",
          cell: (info) => (
            <Badge variant="outline" className="text-xs">
              {(info.row.original.meta?.gender as string) || "N/A"}
            </Badge>
          ),
        },
        {
          id: "meta_actor",
          header: "Actor",
          cell: (info) => (
            <Badge variant="outline" className="text-xs">
              {(info.row.original.meta?.actor as string) || "N/A"}
            </Badge>
          ),
        },
      );
    }

    // Confidence and Duration last
    cols.push(
      {
        accessorKey: "confidence",
        header: "Confidence",
        cell: (info) => {
          const value = info.getValue() as number;
          const isLoadingRow = !!loadingById[info.row.original.id];
          return <span className="text-xs">{isLoadingRow ? "…" : (value * 100).toFixed(1) + "%"}</span>;
        },
      },
      {
        accessorKey: "duration",
        header: "Duration",
        cell: (info) => {
          const value = info.getValue() as number;
          return <span className="text-xs">{value.toFixed(2)}s</span>;
        },
      }
    );

    return cols;
  }, [onRowSelect, onFilePlay, activeDatasetId, loadingById, handlePredict, model]);

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
      pagination,
    },
    onGlobalFilterChange: () => {},
    onPaginationChange: setPagination,
    // Do not auto-reset to page 1 when data updates (e.g., inference results)
    autoResetPageIndex: false,
  });

  // Preload predictions for the currently visible page rows (wav2vec2 and whisper-base)
  const preloadVisiblePage = useCallback(() => {
    if (!(model === "wav2vec2" || model === "whisper-base" || model === "whisper-large")) return;
    const rows = table.getRowModel().rows;
    for (const r of rows) {
      const orig = r.original as AudioData;
      if (!orig) continue;
      if (loadingByIdRef.current[orig.id]) continue;
      // Skip if we already have a prediction for the active model
      if (model === "wav2vec2") {
        if (orig.predictedLabel && orig.predictedLabel.trim().length > 0) continue;
      } else if (model === "whisper-base" || model === "whisper-large") {
        if (orig.predictedTranscript && orig.predictedTranscript.trim().length > 0) continue;
      }
      void handlePredict(orig);
    }
  }, [model, handlePredict, table]);

  // Run preload when page, size, search, dataset, or model changes
  useEffect(() => {
    preloadVisiblePage();
  }, [preloadVisiblePage, pagination.pageIndex, pagination.pageSize, searchQuery, activeDatasetId, tableData]);

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <TableHead key={header.id} className="py-2">
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
                onClick={() => {
                  const orig = row.original as AudioData;
                  onRowSelect(orig.id);
                  if (onFilePlay) {
                    onFilePlay({
                      file_id: orig.id,
                      filename: orig.filename,
                      file_path: orig.relpath,
                      message: "dataset",
                      size: orig.size,
                      duration: orig.duration,
                      sample_rate: orig.sample_rate,
                      prediction: orig.prediction,
                      // carry cached GT label for the editor panel
                      // (AudioDataTable derives this from backend label/meta)
                      label: orig.groundTruthLabel,
                      dataset_id: activeDatasetId,
                      meta: orig.meta,
                    });
                  }
                  // Also trigger prediction when selecting a row
                  void handlePredict(orig);
                }}
                className={cn(
                  "cursor-pointer hover:bg-muted/50 h-9",
                  selectedRow === row.original.id && "bg-muted"
                )}
              >
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id} className="py-1">
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
          Page {table.getState().pagination.pageIndex + 1} of {table.getPageCount()}
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
