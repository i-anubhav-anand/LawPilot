"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Paperclip, ImageIcon, Send, File, FileText, X, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { LegalDisclaimer } from "./legal-disclaimer"
import { uploadDocument, pollDocumentStatus, type DocumentResponse } from "@/lib/api"
import { toast } from "@/components/ui/use-toast"
import { ProcessingDialog } from "./processing-dialog"

interface ChatInputProps {
  onSendMessage: (message: string, attachments: File[]) => void
  isLoading?: boolean
  disabled?: boolean
  sessionId?: string
  caseFileId?: string
}

export function ChatInput({ 
  onSendMessage, 
  isLoading = false, 
  disabled = false,
  sessionId,
  caseFileId 
}: ChatInputProps) {
  const [input, setInput] = useState("")
  const [charCount, setCharCount] = useState(0)
  const [showAttachMenu, setShowAttachMenu] = useState(false)
  const [showImageMenu, setShowImageMenu] = useState(false)
  const [attachments, setAttachments] = useState<File[]>([])
  const [uploadingFiles, setUploadingFiles] = useState<{name: string, progress?: number}[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [processingFiles, setProcessingFiles] = useState<DocumentResponse[]>([])
  const [showProcessingDialog, setShowProcessingDialog] = useState(false)
  const [processingStatus, setProcessingStatus] = useState("")

  const fileInputRef = useRef<HTMLInputElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  
  // Helper function to get overall progress
  const getOverallProgress = (): number => {
    if (uploadingFiles.length === 0) return 0;
    
    // If we have progress values, calculate average
    const progressValues = uploadingFiles
      .filter(f => f.progress !== undefined)
      .map(f => f.progress || 0); // Use 0 as fallback even though we've filtered undefined
      
    if (progressValues.length > 0) {
      return progressValues.reduce((a, b) => a + b, 0) / progressValues.length;
    }
    
    // Otherwise calculate based on processing status
    const processed = processingFiles.filter(f => f.status === "processed").length;
    const totalFiles = processingFiles.length || 1; // Ensure we don't divide by zero
    return (processed / totalFiles) * 100;
  }

  // Add this file text extraction function
  async function extractTextFromFile(file: File): Promise<string> {
    try {
      if (file.type.includes('text')) {
        return await file.text();
      }
      // For other file types, we'll just rely on the backend extraction
      return "";
    } catch (error) {
      console.error("Error extracting text:", error);
      return "";
    }
  }

  // Modify the handleFileUpload function
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files)
      
      // If we're uploading directly to the backend
      if (sessionId) {
        // Show the processing dialog
        setShowProcessingDialog(true)
        setIsUploading(true)
        setUploadingFiles(newFiles.map(f => ({ name: f.name })))
        
        try {
          let processedTextMessage = "";
          
          // First, try to extract text from text files only
          for (let i = 0; i < newFiles.length; i++) {
            const file = newFiles[i];
            if (file.type.includes('text')) {
              setProcessingStatus(`Extracting text from ${file.name}...`);
              
              // Extract text from file
              const extractedText = await extractTextFromFile(file);
              if (extractedText && extractedText.length > 0) {
                processedTextMessage += `\n\n--- Content from ${file.name} ---\n${extractedText}`;
              }
            }
          }
          
          // Silently process the extracted text in the background if applicable
          if (processedTextMessage.length > 0 && input.trim().length > 0) {
            // No UI change here - this happens silently
            setProcessingStatus(`Processing extracted text...`);
            
            // Create a custom event with the extracted text
            if (window && sessionId) {
              const event = new CustomEvent('extracted-text-ready', {
                detail: {
                  text: processedTextMessage,
                  documentName: "Extracted Text",
                  message: input,
                  sessionId: sessionId,
                  caseFileId: caseFileId
                }
              } as CustomEventInit);
              
              window.dispatchEvent(event);
            }
          }
          
          // Upload files one by one for better tracking
          const uploadedFiles: DocumentResponse[] = [];
          
          for (let i = 0; i < newFiles.length; i++) {
            const file = newFiles[i];
            setProcessingStatus(`Uploading ${file.name} (${i+1}/${newFiles.length})...`);
            
            // Upload the file
            const response = await uploadDocument(file, sessionId, caseFileId);
            uploadedFiles.push(response);
            
            // Update progress
            setUploadingFiles(prev => {
              const updated = [...prev];
              const fileIndex = updated.findIndex(f => f.name === file.name);
              if (fileIndex !== -1) {
                updated[fileIndex].progress = 50; // We're halfway done with this file
              }
              return updated;
            });
            
            // Start polling for status updates
            setProcessingStatus(`Processing ${file.name}...`);
            try {
              await pollDocumentStatus(
                response.document_id,
                (status) => {
                  // Update processing files
                  setProcessingFiles(prev => {
                    // Replace the file if it exists, otherwise add it
                    const fileIndex = prev.findIndex(f => f.document_id === status.document_id);
                    if (fileIndex !== -1) {
                      const updated = [...prev];
                      updated[fileIndex] = status;
                      return updated;
                    } else {
                      return [...prev, status];
                    }
                  });
                  
                  // Update progress
                  setUploadingFiles(prev => {
                    const updated = [...prev];
                    const fileIndex = updated.findIndex(f => f.name === file.name);
                    if (fileIndex !== -1) {
                      const progress = status.status === "processed" ? 100 : 
                                      status.status === "failed" ? 0 : 75;
                      updated[fileIndex].progress = progress;
                    }
                    return updated;
                  });
                  
                  setProcessingStatus(`Processing ${file.name}: ${status.status}`);
                }
              );
            } catch (e) {
              console.warn(`Status polling failed for ${file.name}:`, e);
              // Continue with other files even if polling fails
            }
          }
          
          // All files processed, update UI
          setProcessingStatus("All files processed successfully");
          setTimeout(() => {
            setShowProcessingDialog(false);
          }, 1500);
          
          // Add a message about the uploads
          const fileNames = newFiles.map(f => f.name).join(", ");
          const uploadMsg = `I've uploaded the following document(s): ${fileNames}`;
          onSendMessage(uploadMsg, []);
          
          toast({
            title: "Documents uploaded",
            description: `${newFiles.length} document(s) uploaded successfully and are being processed.`,
          });
        } catch (error) {
          console.error("Error uploading documents:", error);
          setProcessingStatus("Upload failed. Please try again.");
          
          setTimeout(() => {
            setShowProcessingDialog(false);
          }, 2000);
          
          toast({
            title: "Upload failed",
            description: error instanceof Error ? error.message : "Failed to upload documents",
            variant: "destructive"
          });
        } finally {
          setTimeout(() => {
            setIsUploading(false);
            setUploadingFiles([]);
            setProcessingFiles([]);
          }, 2000);
        }
      } else {
        // Just add to local attachments if no session ID
        setAttachments((prev) => [...prev, ...newFiles]);
      }

      // Close menus
      setShowAttachMenu(false);
      setShowImageMenu(false);
      
      // Reset file input to allow the same file to be selected again
      if (e.target) {
        e.target.value = "";
      }
    }
  };

  const removeAttachment = (index: number) => {
    setAttachments((prev) => prev.filter((_, i) => i !== index));
  };

  // Update the handleSubmit function
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (input.trim() || attachments.length > 0) {
      // Create a message that includes info about attachments if they exist
      let finalMessage = input;
      if (attachments.length > 0) {
        const fileNames = attachments.map(file => file.name).join(", ");
        if (input.trim()) {
          finalMessage = `${input}\n\n[Attached files: ${fileNames}]`;
        } else {
          finalMessage = `I've uploaded the following document(s): ${fileNames}`;
        }
      }
      
      onSendMessage(finalMessage, attachments);
      setInput("");
      setCharCount(0);
      setAttachments([]);
    }
  };

  // Cancel processing
  const handleCancelProcessing = () => {
    setShowProcessingDialog(false);
    setIsUploading(false);
    setUploadingFiles([]);
    setProcessingFiles([]);
  };

  // Auto-grow textarea
  useEffect(() => {
    const textarea = document.querySelector('textarea');
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = textarea.scrollHeight + 'px';
    }
  }, [input]);

  return (
    <div className="relative bg-white">
      <LegalDisclaimer />

      {/* Processing Dialog */}
      <ProcessingDialog
        open={showProcessingDialog}
        title="Processing Documents"
        description="Your documents are being uploaded and processed..."
        progress={getOverallProgress()}
        status={processingStatus}
        showCancel={true}
        onCancel={handleCancelProcessing}
      />

      <div className="rounded-lg border border-gray-200 bg-white overflow-hidden shadow-sm">
        {/* Attachment preview */}
        {attachments.length > 0 && (
          <div className="px-5 pt-3">
            <div className="flex flex-wrap gap-2">
              {attachments.map((file, index) => (
                <div key={index} className="flex items-center bg-gray-50 rounded-md px-2 py-1 text-sm">
                  {file.type.startsWith("image/") ? (
                    <ImageIcon size={14} className="text-purple-500 mr-1" />
                  ) : (
                    <FileText size={14} className="text-purple-500 mr-1" />
                  )}
                  <span className="truncate max-w-[120px]">{file.name}</span>
                  <button onClick={() => removeAttachment(index)} className="ml-1 text-gray-400 hover:text-gray-600">
                    <X size={14} />
                  </button>
                </div>
              ))}
            </div>
            
            <div className="flex items-center justify-between mt-2 pb-1">
              <div className="text-xs text-purple-600 font-medium flex items-center">
                <Paperclip size={12} className="mr-1" />
                {attachments.length > 1 ? `${attachments.length} files attached` : '1 file attached'}
              </div>
            </div>
          </div>
        )}
        
        {/* File upload progress */}
        {isUploading && uploadingFiles.length > 0 && (
          <div className="px-5 pt-3">
            <div className="bg-purple-50 border border-purple-100 rounded-md p-2">
              <div className="flex items-center text-sm text-purple-600">
                <Loader2 size={14} className="mr-2 animate-spin" />
                <span>Uploading {uploadingFiles.length} file(s)...</span>
              </div>
              <div className="mt-1 text-xs text-purple-500">
                {uploadingFiles.slice(0, 3).map((file, i) => (
                  <div key={i} className="truncate max-w-[300px]">{file.name}</div>
                ))}
                {uploadingFiles.length > 3 && (
                  <div>and {uploadingFiles.length - 3} more...</div>
                )}
              </div>
            </div>
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <textarea
            placeholder="Ask whatever you want...."
            className="w-full px-5 pt-4 pb-12 resize-none focus:outline-none min-h-[80px] max-h-[200px] font-normal text-base"
            value={input}
            onChange={(e) => {
              const value = e.target.value;
              setInput(value);
              setCharCount(value.length);
            }}
            maxLength={1000}
            disabled={disabled || isUploading}
            style={{ overflow: 'hidden' }}
          />

          <div className="absolute bottom-0 left-0 right-0 px-4 py-2.5 flex items-center justify-between border-t border-gray-100">
            <div className="flex items-center gap-3">
              {/* File attachment button with indicator */}
              <div className="relative">
                <button
                  type="button"
                  className={`p-1.5 rounded-md hover:bg-gray-100 transition-colors ${
                    disabled || isUploading ? "opacity-50 cursor-not-allowed" : ""
                  } ${attachments.length > 0 ? "bg-purple-100 text-purple-600" : ""}`}
                  onClick={() => {
                    if (!disabled && !isUploading) {
                      setShowAttachMenu(!showAttachMenu)
                      setShowImageMenu(false)
                    }
                  }}
                  disabled={disabled || isUploading}
                  aria-label="Attach file"
                  title="Attach file"
                >
                  <Paperclip className="h-5 w-5" />
                  {attachments.length > 0 && (
                    <span className="absolute -top-2 -right-2 bg-purple-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                      {attachments.length}
                    </span>
                  )}
                </button>

                {/* Hidden file input */}
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  className="hidden" 
                  onChange={handleFileUpload} 
                  accept=".pdf,.txt,.docx,.doc"
                  multiple 
                />

                {/* Attachment dropdown menu */}
                {showAttachMenu && (
                  <div className="absolute bottom-full left-0 mb-1 bg-white border border-gray-200 rounded-md shadow-md py-1 min-w-[160px] z-10">
                    <button
                      type="button"
                      className="w-full text-left px-3 py-2 text-sm hover:bg-gray-50 flex items-center"
                      onClick={() => {
                        fileInputRef.current?.click()
                      }}
                    >
                      <FileText size={16} className="mr-2 text-purple-500" />
                      Upload Document
                    </button>
                  </div>
                )}
              </div>

              {/* Image upload button with indicator */}
              <div className="relative">
                <button
                  type="button"
                  className={`p-1.5 rounded-md hover:bg-gray-100 transition-colors ${
                    disabled || isUploading ? "opacity-50 cursor-not-allowed" : ""
                  } ${attachments.some(file => file.type.startsWith("image/")) ? "bg-purple-100 text-purple-600" : ""}`}
                  onClick={() => {
                    if (!disabled && !isUploading) {
                      setShowImageMenu(!showImageMenu)
                      setShowAttachMenu(false)
                    }
                  }}
                  disabled={disabled || isUploading}
                  aria-label="Attach image"
                  title="Attach image"
                >
                  <ImageIcon className="h-5 w-5" />
                </button>

                {/* Hidden image input */}
                <input
                  type="file"
                  ref={imageInputRef}
                  className="hidden"
                  accept="image/*"
                  onChange={handleFileUpload}
                  multiple
                />

                {/* Image dropdown menu */}
                {showImageMenu && (
                  <div className="absolute bottom-full left-0 mb-1 bg-white border border-gray-200 rounded-md shadow-md py-1 min-w-[160px] z-10">
                    <button
                      type="button"
                      className="w-full text-left px-3 py-2 text-sm hover:bg-gray-50 flex items-center"
                      onClick={() => {
                        imageInputRef.current?.click()
                      }}
                    >
                      <ImageIcon size={16} className="mr-2 text-purple-500" />
                      Upload image
                    </button>
                  </div>
                )}
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500">{charCount}/1000</span>
                <div className="relative">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="text-xs border-gray-200 font-normal h-7 px-2.5"
                    disabled={disabled || isUploading}
                  >
                    AI Web
                    <svg
                      className="ml-1 h-3 w-3 text-gray-500"
                      fill="none"
                      height="24"
                      stroke="currentColor"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      viewBox="0 0 24 24"
                      width="24"
                    >
                      <path d="m6 9 6 6 6-6"></path>
                    </svg>
                  </Button>
                </div>
              </div>

              <Button
                type="submit"
                size="icon"
                className="rounded-md bg-purple-500 hover:bg-purple-600 text-white h-7 w-7"
                disabled={(!input.trim() && attachments.length === 0) || isLoading || disabled || isUploading}
              >
                {isLoading || isUploading ? (
                  <div className="h-3.5 w-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <Send className="h-3.5 w-3.5" />
                )}
              </Button>
            </div>
          </div>
        </form>
      </div>
    </div>
  )
}
