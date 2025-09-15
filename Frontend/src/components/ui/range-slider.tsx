"use client"

import * as React from "react"
import * as SliderPrimitive from "@radix-ui/react-slider"
import { cn } from "@/lib/utils"

interface RangeSliderProps extends Omit<React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>, 'value' | 'onValueChange'> {
  value: [number, number]
  onValueChange: (value: [number, number]) => void
  formatLabel?: (value: number) => string
  showLabels?: boolean
}

const RangeSlider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  RangeSliderProps
>(({ className, value, onValueChange, formatLabel, showLabels = true, ...props }, ref) => {
  return (
    <div className="space-y-3">
      <SliderPrimitive.Root
        ref={ref}
        className={cn(
          "relative flex w-full touch-none select-none items-center",
          className
        )}
        value={value}
        onValueChange={onValueChange}
        {...props}
      >
        <SliderPrimitive.Track className="relative h-2 w-full grow overflow-hidden rounded-full bg-gray-200">
          <SliderPrimitive.Range className="absolute h-full bg-blue-600" />
        </SliderPrimitive.Track>
        
        {/* First thumb (start) */}
        <SliderPrimitive.Thumb className="block h-5 w-5 rounded-full border-2 border-blue-600 bg-white ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 hover:scale-110 shadow-md" />
        
        {/* Second thumb (end) */}
        <SliderPrimitive.Thumb className="block h-5 w-5 rounded-full border-2 border-blue-600 bg-white ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 hover:scale-110 shadow-md" />
      </SliderPrimitive.Root>
      
      {showLabels && (
        <div className="flex justify-between text-xs text-gray-600">
          <span>{formatLabel ? formatLabel(value[0]) : `${value[0]}%`}</span>
          <span>Range: {formatLabel ? formatLabel(value[1] - value[0]) : `${value[1] - value[0]}%`}</span>
          <span>{formatLabel ? formatLabel(value[1]) : `${value[1]}%`}</span>
        </div>
      )}
    </div>
  )
})

RangeSlider.displayName = "RangeSlider"

export { RangeSlider }