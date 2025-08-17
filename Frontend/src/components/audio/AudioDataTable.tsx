import { useState } from "react";
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
import { Play, ChevronLeft, ChevronRight } from "lucide-react";

interface UploadedFile {
  file_id: string;
  filename: string;
  file_path: string;
  message: string;
  size?: number;
  duration?: number;
  sample_rate?: number;
}

interface AudioData {
  id: string;
  filename: string;
  predictedTranscript: string;
  predictedLabel: string;
  groundTruthLabel: string;
  confidence: number;
  duration: number;
  file_path?: string;
  size?: number;
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

export const AudioDataTable = ({ selectedRow, onRowSelect, searchQuery, apiData, uploadedFiles, onFilePlay }: AudioDataTableProps) => {
  // Convert uploaded files to table data format
  const tableData: AudioData[] = uploadedFiles?.map(file => ({
    id: file.file_id,
    filename: file.filename,
    predictedTranscript: "", // Will be populated from API predictions
    predictedLabel: apiData?.prediction?.text || "",
    groundTruthLabel: "", // Can be manually set later
    confidence: 0.85, // Default confidence, will be updated from predictions
    duration: file.duration || 0,
    file_path: file.file_path,
    size: file.size
  })) || [];
  const columns = [
    columnHelper.accessor("filename", {
      header: "Filename",
      cell: (info) => {
        const row = info.row.original;
        const file = uploadedFiles?.find(f => f.file_id === row.id);
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
    columnHelper.accessor("predictedLabel", {
      header: "Predicted Label",
      cell: (info) => (
        <Badge variant="outline" className="text-xs">
          {apiData?.prediction?.text ?? ""}
        </Badge>
      ),
    }),
    columnHelper.accessor("groundTruthLabel", {
      header: "Ground Truth",
      cell: (info) => (
        <Badge variant="secondary" className="text-xs">
          {apiData?.prediction?.text ?? ""}
        </Badge>
      ),
    }),
    columnHelper.accessor("confidence", {
      header: "Confidence",
      cell: (info) => (
        <span className="text-xs">{apiData ? (info.getValue() * 100).toFixed(0) : ""}%</span>
      ),
    }),
    columnHelper.accessor("duration", {
      header: "Duration",
      cell: (info) => (
        <span className="text-xs">{apiData ? info.getValue() : ""}s</span>
      ),
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
            {table.getRowModel().rows?.length ? (
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