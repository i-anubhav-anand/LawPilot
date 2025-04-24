"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Paperclip, ImageIcon, Send, File, FileText, X, Loader2, CheckCircle, AlertCircle, Info } from "lucide-react"
import { Button } from "@/components/ui/button"
import { LegalDisclaimer } from "./legal-disclaimer"
import { uploadDocument, pollDocumentStatus, type DocumentResponse } from "@/lib/api"
import { toast } from "@/components/ui/use-toast"
import { CircularProgress } from "@/components/ui/circular-progress"
import { DISABLE_PROCESSING_DIALOGS, trackUploadProgress } from "@/lib/processing-settings"
import { v4 as uuidv4 } from "uuid"

// Add an interface for FileWithStatus to track upload progress and processing status in UI
interface FileWithStatus {
  file: File;
  id?: string;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  error?: string;
}

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
  const [attachments, setAttachments] = useState<FileWithStatus[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  
  const fileInputRef = useRef<HTMLInputElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  
  // Helper function to get overall progress
  const getOverallProgress = (): number => {
    if (attachments.length === 0) return 0;
    
    // Calculate average progress of all files
    const progressValues = attachments.map(f => f.progress ?? 0);
    
    if (progressValues.length > 0) {
      return progressValues.reduce((a, b) => a + b, 0) / progressValues.length;
    }
    
    return 0;
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

  // Modified to track file attachment status better
  const handleFileUpload = async (files: FileList) => {
    if (files.length === 0) return;
    
    console.log(`📋 Preparing ${files.length} files for upload`);
    
    const newAttachments: FileWithStatus[] = [];
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      console.log(`📄 Adding file: ${file.name} (${file.type}, ${file.size} bytes)`);
      
      // Check file size - add a reasonable limit (e.g., 50MB)
      const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
      if (file.size > MAX_FILE_SIZE) {
        toast({
          title: "File too large",
          description: `"${file.name}" exceeds the 50MB size limit.`,
          variant: "destructive",
          duration: 5000,
        });
        continue;
      }
      
      // Check file extension for supported types
      const fileExtension = file.name.split('.').pop()?.toLowerCase();
      const supportedExtensions = ['pdf', 'txt', 'doc', 'docx', 'jpg', 'jpeg', 'png'];
      
      if (!fileExtension || !supportedExtensions.includes(fileExtension)) {
        toast({
          title: "Unsupported file type",
          description: `"${file.name}" is not a supported file type. Please upload PDF, text, Word documents, or images.`,
          variant: "destructive",
          duration: 5000,
        });
        continue;
      }
      
      const fileWithStatus: FileWithStatus = {
        file,
        id: uuidv4(),
        status: "pending",
        progress: 0
      };
      
      newAttachments.push(fileWithStatus);
    }
    
    if (newAttachments.length === 0) {
      return; // No valid attachments to add
    }
    
    setAttachments(prev => [...prev, ...newAttachments]);
    
    // Just mark files as ready for upload
    for (const attachment of newAttachments) {
      try {
        console.log(`✅ File ready for upload: ${attachment.file.name}`);
        
        // Display a toast notification for better user feedback
        toast({
          title: "File ready",
          description: `"${attachment.file.name}" is ready to send. Add a message and click send to process the file.`,
          duration: 4000,
        });
        
        setAttachments(prev => 
          prev.map(att => 
            att.id === attachment.id 
              ? { ...att, status: "pending", progress: 100 } 
              : att
          )
        );
      } catch (error) {
        console.error(`❌ Error preparing file ${attachment.file.name}:`, error);
        
        let errorMessage = "Failed to prepare file";
        if (error instanceof Error) {
          errorMessage = error.message;
        }
        
        setAttachments(prev => 
          prev.map(att => 
            att.id === attachment.id 
              ? { ...att, status: "error", error: errorMessage } 
              : att
          )
        );
        
        toast({
          title: "File Error",
          description: errorMessage,
          variant: "destructive",
          duration: 5000,
        });
      }
    }
  };

  const removeAttachment = (index: number) => {
    setAttachments((prev) => prev.filter((_, i) => i !== index));
  };

  // Update the handleSubmit function to process attachments when sending
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Enhanced validation for files without a message
    if (!input.trim()) {
      if (attachments.length > 0) {
        // Highlight textarea with error styling
        const textarea = document.querySelector('textarea');
        if (textarea) {
          textarea.classList.add('border', 'border-red-300', 'bg-red-50');
          
          // Remove error styling after 2 seconds
          setTimeout(() => {
            textarea.classList.remove('border', 'border-red-300', 'bg-red-50');
          }, 2000);
        }
        
        toast({
          title: "Message required",
          description: "Please add a message to send with your file attachment.",
          variant: "destructive"
        });
      }
      return;
    }
    
    setIsProcessing(true);
    
    try {
      // Create a message that includes info about attachments if they exist
      let finalMessage = input;
      
      // Mark files as being processed
      if (attachments.length > 0) {
        console.log(`Sending message with ${attachments.length} attachments`);
        
        setAttachments(prev => 
          prev.map(att => ({ 
            ...att, 
            status: "uploading", 
            progress: 50 
          }))
        );
      }
      
      // Send all files through the onSendMessage handler
      const filesToSend = attachments.map(a => a.file);
      
      // The actual API call happens in the parent component via onSendMessage
      onSendMessage(finalMessage, filesToSend);
      
      // Reset state
      setInput("");
      setCharCount(0);
      
      // Mark attachments as completed before clearing
      setAttachments(prev => 
        prev.map(att => ({ 
          ...att, 
          status: "completed", 
          progress: 100 
        }))
      );
      
      // Clear attachments with a short delay to show completion state
      setTimeout(() => {
        setAttachments([]);
      }, 500);
      
    } catch (error) {
      console.error("Error sending message:", error);
      
      // Update attachment statuses to error
      if (attachments.length > 0) {
        setAttachments(prev => 
          prev.map(att => ({ 
            ...att, 
            status: "error", 
            error: "Failed to send message with attachment" 
          }))
        );
      }
      
      toast({
        title: "Error",
        description: "Failed to send message. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsProcessing(false);
    }
  };

  // Auto-grow textarea
  useEffect(() => {
    const textarea = document.querySelector('textarea');
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = textarea.scrollHeight + 'px';
    }
  }, [input]);

  // Determine if send button should be disabled
  const sendButtonDisabled = !input.trim() || isLoading || disabled || isProcessing;

  // Update the file attachment section to use CircularProgress for better status indication
  const renderAttachments = () => {
    if (attachments.length === 0) return null;

    return (
      <div className="flex flex-col space-y-2 mt-2">
        {attachments.map((attachment, index) => (
          <div
            key={attachment.id || index}
            className="flex items-center justify-between p-2 rounded-md bg-gray-50 border border-gray-200 text-sm"
          >
            <div className="flex items-center space-x-2 overflow-hidden">
              {/* File icon based on type */}
              <div className="flex-shrink-0">
                {attachment.file.type.includes("image") ? (
                  <ImageIcon className="h-4 w-4 text-gray-500" />
                ) : (
                  <File className="h-4 w-4 text-gray-500" />
                )}
              </div>
              
              {/* File name and status */}
              <div className="flex flex-col min-w-0">
                <span className="truncate font-medium" title={attachment.file.name}>
                  {attachment.file.name}
                </span>
                <span className="text-xs text-gray-500">
                  {formatFileSize(attachment.file.size)}
                </span>
              </div>
            </div>
            
            {/* Status indicators */}
            <div className="flex items-center space-x-2">
              {attachment.status === 'error' && (
                <span className="text-xs text-red-500">{attachment.error || 'Error'}</span>
              )}
              
              {attachment.status === 'pending' && (
                <span className="text-xs text-gray-500">Ready</span>
              )}
              
              {attachment.status === 'uploading' && (
                <div className="flex items-center space-x-1">
                  <CircularProgress value={attachment.progress} size={16} />
                  <span className="text-xs text-purple-500">Uploading</span>
                </div>
              )}
              
              {attachment.status === 'processing' && (
                <div className="flex items-center space-x-1">
                  <CircularProgress indeterminate size={16} />
                  <span className="text-xs text-blue-500">Processing</span>
                </div>
              )}
              
              {attachment.status === 'completed' && (
                <CheckCircle className="h-4 w-4 text-green-500" />
              )}
              
              {/* Remove button */}
              <button
                type="button"
                onClick={() => removeAttachment(index)}
                className="text-gray-400 hover:text-red-500"
                disabled={isProcessing || isLoading}
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>
        ))}
      </div>
    );
  };

  // Helper function to format file size
  const formatFileSize = (size: number): string => {
    if (size < 1024) {
      return `${size} B`;
    } else if (size < 1024 * 1024) {
      return `${(size / 1024).toFixed(1)} KB`;
    } else {
      return `${(size / (1024 * 1024)).toFixed(1)} MB`;
    }
  };

  return (
    <div className="relative bg-white">
      <LegalDisclaimer />

      <div className="rounded-lg border border-gray-200 bg-white overflow-hidden shadow-sm">
        {/* Display file attachments with processing status */}
        {attachments.length > 0 && (
          <div className="px-4 pt-3 pb-1">
            {/* Instructions banner for users */}
            <div className="mb-2 p-2 bg-blue-50 border border-blue-100 rounded-md text-sm text-blue-700 flex items-center">
              <Info className="h-4 w-4 mr-2 flex-shrink-0" />
              <span>Add a message below and click send to process your {attachments.length > 1 ? 'files' : 'file'}</span>
            </div>
            {renderAttachments()}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <textarea
            placeholder={attachments.length > 0 ? "Type a message to send with your attachment..." : "Ask whatever you want...."}
            className="w-full px-5 pt-4 pb-12 resize-none focus:outline-none min-h-[80px] max-h-[200px] font-normal text-base"
            value={input}
            onChange={(e) => {
              const value = e.target.value;
              setInput(value);
              setCharCount(value.length);
            }}
            maxLength={1000}
            disabled={disabled || isProcessing}
            style={{ overflow: 'hidden' }}
          />

          <div className="absolute bottom-0 left-0 right-0 px-4 py-2.5 flex items-center justify-between border-t border-gray-100">
            <div className="flex items-center gap-3">
              {/* File attachment button with indicator */}
              <div className="relative">
                <button
                  type="button"
                  className={`p-1.5 rounded-md hover:bg-gray-100 transition-colors ${
                    disabled || isProcessing ? "opacity-50 cursor-not-allowed" : ""
                  } ${attachments.length > 0 ? "bg-purple-100 text-purple-600" : ""}`}
                  onClick={() => {
                    if (!disabled && !isProcessing) {
                      setShowAttachMenu(!showAttachMenu)
                      setShowImageMenu(false)
                    }
                  }}
                  disabled={disabled || isProcessing}
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
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) {
                      console.log(`File selected: ${e.target.files[0].name}, size: ${e.target.files[0].size}, type: ${e.target.files[0].type}`);
                      handleFileUpload(e.target.files);
                      // Close menu after selection
                      setShowAttachMenu(false);
                      // Focus on textarea for user to add a message
                      document.querySelector('textarea')?.focus();
                    }
                  }}
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
                    disabled || isProcessing ? "opacity-50 cursor-not-allowed" : ""
                  } ${attachments.some(a => a.file.type.startsWith("image/")) ? "bg-purple-100 text-purple-600" : ""}`}
                  onClick={() => {
                    if (!disabled && !isProcessing) {
                      setShowImageMenu(!showImageMenu)
                      setShowAttachMenu(false)
                    }
                  }}
                  disabled={disabled || isProcessing}
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
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) {
                      console.log(`Image selected: ${e.target.files[0].name}, size: ${e.target.files[0].size}, type: ${e.target.files[0].type}`);
                      handleFileUpload(e.target.files);
                      // Close menu after selection
                      setShowImageMenu(false);
                      // Focus on textarea for user to add a message
                      document.querySelector('textarea')?.focus();
                    }
                  }}
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
              <div className="text-xs text-gray-400">
                {charCount}/1000
              </div>
            </div>
            
            <Button 
              type="submit" 
              size="sm" 
              disabled={sendButtonDisabled}
              className={`rounded-full px-3 py-1.5 ${
                attachments.length > 0 && input.trim() 
                  ? 'bg-purple-600 hover:bg-purple-700 text-white border-purple-600' 
                  : ''
              }`}
              title={!input.trim() && attachments.length > 0 ? "Please enter a message to send with your files" : ""}
            >
              {isLoading || isProcessing ? (
                <div className="h-4 w-4 border-2 border-current border-t-transparent rounded-full animate-spin mr-1" />
              ) : (
                <Send className={`h-4 w-4 ${attachments.length > 0 && input.trim() ? 'mr-1' : ''}`} />
              )}
              {attachments.length > 0 && input.trim() && <span>Send file{attachments.length > 1 ? 's' : ''}</span>}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
