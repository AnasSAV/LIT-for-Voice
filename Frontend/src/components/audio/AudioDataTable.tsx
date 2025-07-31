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

interface AudioData {
  id: string;
  filename: string;
  predictedTranscript: string;
  predictedLabel: string;
  groundTruthLabel: string;
  confidence: number;
  duration: number;
}

const mockData: AudioData[] = [
  {
    id: "1",
    filename: "audio_sample_001.wav",
    predictedTranscript: "The quick brown fox jumps over the lazy dog",
    predictedLabel: "neutral",
    groundTruthLabel: "neutral",
    confidence: 0.87,
    duration: 3.2
  },
  {
    id: "2", 
    filename: "audio_sample_002.wav",
    predictedTranscript: "Hello world this is a test",
    predictedLabel: "happy",
    groundTruthLabel: "happy",
    confidence: 0.92,
    duration: 2.8
  },
  // Add more mock data...
];

const columnHelper = createColumnHelper<AudioData>();

interface AudioDataTableProps {
  selectedRow: string | null;
  onRowSelect: (id: string) => void;
  searchQuery: string;
}

export const AudioDataTable = ({ selectedRow, onRowSelect, searchQuery }: AudioDataTableProps) => {
  const columns = [
    columnHelper.accessor("filename", {
      header: "Filename",
      cell: (info) => (
        <div className="flex items-center gap-2">
          <Button size="sm" variant="ghost" className="h-6 w-6 p-0">
            <Play className="h-3 w-3" />
          </Button>
          <span className="font-mono text-xs">{info.getValue()}</span>
        </div>
      ),
    }),
    columnHelper.accessor("predictedTranscript", {
      header: "Predicted Transcript",
      cell: (info) => (
        <span className="text-xs">{info.getValue()}</span>
      ),
    }),
    columnHelper.accessor("predictedLabel", {
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
        <span className="text-xs">{(info.getValue() * 100).toFixed(0)}%</span>
      ),
    }),
    columnHelper.accessor("duration", {
      header: "Duration",
      cell: (info) => (
        <span className="text-xs">{info.getValue()}s</span>
      ),
    }),
  ];

  const table = useReactTable({
    data: mockData,
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