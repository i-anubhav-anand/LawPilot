/**
 * Global settings for document processing UI
 */

// Global flag to disable all document processing popup dialogs
export const DISABLE_PROCESSING_DIALOGS = true;

// Use inline progress indicators instead of modal dialogs
export const USE_INLINE_PROGRESS = true;

// Maximum number of concurrent document uploads
export const MAX_CONCURRENT_UPLOADS = 5;

/**
 * Helper to determine if any processing dialogs should be shown
 * This will always return false to ensure no dialogs appear
 */
export function shouldShowProcessingDialog(): boolean {
  return false; // Always disabled as per user request
}

/**
 * Track file upload progress without showing dialogs
 */
export function trackUploadProgress(
  fileId: string, 
  progress: number,
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error'
): void {
  // Log to console but don't display any dialogs
  console.log(`File ${fileId} progress: ${progress}%, status: ${status}`);
} 