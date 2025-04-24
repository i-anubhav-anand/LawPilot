"use client"

import { useState, useEffect } from 'react'
import { DISABLE_PROCESSING_DIALOGS, shouldShowProcessingDialog } from '@/lib/processing-settings'

/**
 * This hook is designed to completely disable processing dialogs
 * It always returns state indicating dialogs should be closed/hidden
 */
export function useProcessingDialog() {
  // Always set to false to disable all dialogs
  const [isOpen, setIsOpen] = useState(false)
  const [progress, setProgress] = useState(0)
  
  // Override any attempts to open the dialog
  const open = () => {
    // Do nothing - dialog should stay closed
    console.log('Processing dialog open prevented by user preference')
  }
  
  const close = () => {
    setIsOpen(false)
  }
  
  const updateProgress = (value: number) => {
    // Still update progress for tracking purposes
    setProgress(value)
  }
  
  // Force any dialogs closed on mount
  useEffect(() => {
    // Make sure dialog stays closed
    if (isOpen) {
      setIsOpen(false)
    }
  }, [isOpen])
  
  return {
    isOpen: false, // Always return false to ensure dialogs stay hidden
    progress,
    open,
    close,
    updateProgress
  }
} 