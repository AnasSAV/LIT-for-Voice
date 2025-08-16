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
import { Play, ChevronLeft, ChevronRight, Trash2 } from "lucide-react";
import { useAudio, AudioFile } from "@/contexts/AudioContext";

const columnHelper = createColumnHelper<AudioFile>();

interface AudioDataTableProps {
  selectedRow: string | null;
  onRowSelect: (id: string) => void;
  searchQuery: string;
}

export const AudioDataTable = ({ selectedRow, onRowSelect, searchQuery }: AudioDataTableProps) => {
  const { audioFiles, selectAudio, removeAudio, currentAudio } = useAudio();

  const formatDuration = (duration: number | undefined) => {
    if (!duration) return "0:00";
    const mins = Math.floor(duration / 60);
    const secs = Math.floor(duration % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handlePlayClick = (audioFile: AudioFile, e: React.MouseEvent) => {
    e.stopPropagation();
    selectAudio(audioFile.id);
  };

  const handleDeleteClick = (audioFile: AudioFile, e: React.MouseEvent) => {
    e.stopPropagation();
    removeAudio(audioFile.id);
  };

  const columns = [
    columnHelper.accessor("name", {
      header: "Filename",
      cell: (info) => {
        const audioFile = info.row.original;
        const isSelected = currentAudio?.id === audioFile.id;
        return (
          <div className="flex items-center gap-2">
            <Button 
              size="sm" 
              variant={isSelected ? "default" : "ghost"} 
              className="h-6 w-6 p-0"
              onClick={(e) => handlePlayClick(audioFile, e)}
            >
              <Play className="h-3 w-3" />
            </Button>
            <span className="font-mono text-xs truncate max-w-[200px]" title={info.getValue()}>
              {info.getValue()}
            </span>
          </div>
        );
      },
    }),
    columnHelper.accessor("predictions", {
      header: "Predictions",
      cell: (info) => {
        const predictions = info.getValue() || [];
        return (
          <div className="flex flex-wrap gap-1">
            {predictions.map((pred, index) => (
              <Badge key={index} variant="outline" className="text-xs">
                {pred.model}: {pred.transcription || pred.emotion || "Processing..."}
              </Badge>
            ))}
            {predictions.length === 0 && (
              <span className="text-xs text-muted-foreground">No predictions</span>
            )}
          </div>
        );
      },
    }),
    columnHelper.accessor("duration", {
      header: "Duration",
      cell: (info) => (
        <span className="text-xs">{formatDuration(info.getValue())}</span>
      ),
    }),
    columnHelper.display({
      id: "actions",
      header: "Actions",
      cell: (info) => {
        const audioFile = info.row.original;
        return (
          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 p-0 text-destructive hover:text-destructive"
            onClick={(e) => handleDeleteClick(audioFile, e)}
          >
            <Trash2 className="h-3 w-3" />
          </Button>
        );
      },
    }),
  ];

  const filteredData = audioFiles.filter(file => 
    file.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const table = useReactTable({
    data: filteredData,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
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
                  onClick={() => {
                    onRowSelect(row.original.id);
                    selectAudio(row.original.id);
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
                <TableCell colSpan={columns.length} className="h-24 text-center text-xs">
                  {audioFiles.length === 0 ? "No audio files uploaded." : "No results match your search."}
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
      
      {/* Pagination */}
      <div className="border-t panel-border p-2 flex items-center justify-between">
        <div className="text-xs text-muted-foreground">
          {audioFiles.length} file(s) • Page {table.getState().pagination.pageIndex + 1} of {table.getPageCount()}
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