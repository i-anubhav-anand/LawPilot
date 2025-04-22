"use client"

import * as React from "react"
import { Loader2 } from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Progress } from "@/components/ui/progress"

interface ProcessingDialogProps {
  open: boolean
  title: string
  description: string
  progress?: number // Optional progress value (0-100)
  status?: string // Optional status message
  showCancel?: boolean // Whether to show a cancel button
  onCancel?: () => void // Function to call when cancel is clicked
}

export function ProcessingDialog({
  open,
  title,
  description,
  progress,
  status,
  showCancel = false,
  onCancel
}: ProcessingDialogProps) {
  return (
    <Dialog open={open}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>
        <div className="flex flex-col items-center justify-center py-4 space-y-4">
          {progress !== undefined ? (
            <div className="w-full space-y-2">
              <Progress value={progress} className="h-2" />
              <p className="text-xs text-center text-gray-500">{progress.toFixed(0)}%</p>
            </div>
          ) : (
            <div className="flex items-center justify-center">
              <Loader2 className="w-8 h-8 text-purple-500 animate-spin" />
            </div>
          )}
          
          {status && (
            <div className="text-sm text-center text-gray-600 max-w-sm">
              {status}
            </div>
          )}
          
          {showCancel && onCancel && (
            <button
              onClick={onCancel}
              className="px-4 py-2 mt-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
            >
              Cancel
            </button>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
} 