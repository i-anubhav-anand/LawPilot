"use client"

import type React from "react"
import { useState, useEffect, useRef } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { useToast } from "@/components/ui/use-toast"
import { ChatInput } from "@/components/chat-input"
import { PromptSuggestions } from "@/components/prompt-suggestions"
import { FlipWords } from "@/components/ui/flip-words"
import { 
  sendChatMessage, 
  sendChatMessageWithFile, 
  sendChatMessageWithImage,
  sendChatMessageWithDocumentText,
  createChatSession,
  getChatSessions,
  getChatHistory,
  type ChatResponse,
  type Source,
  checkApiHealth,
  updateCaseSummaryFromChat
} from "@/lib/api"
import { Message } from "@/types/chat"
import { ChatMessage } from "@/components/chat-message"
// Import the ErrorMessage component
import { ErrorMessage } from "./error-message"
import { DocumentIssuesFab } from '@/components/document-issues-fab'

// Define the custom event type
declare global {
  interface WindowEventMap {
    'extracted-text-ready': CustomEvent<{
      text: string;
      documentName: string;
      message: string;
      sessionId: string;
      caseFileId?: string;
    }>;
  }
}

interface ChatContainerProps {
  sessionId?: string
  caseFileId?: string
}

export function ChatContainer({ sessionId: initialSessionId, caseFileId }: ChatContainerProps) {
  const [sessionId, setSessionId] = useState<string | undefined>(initialSessionId)
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [showWelcome, setShowWelcome] = useState(true)
  const [apiConnected, setApiConnected] = useState<boolean | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [activeDocumentId, setActiveDocumentId] = useState<string | null>(null)

  // Complete phrases for the FlipWords component
  const legalPhrases = [
    "Need help with case analysis today?",
    "Need a quick legal review today?",
    "Looking to analyze a case today?",
    "Want to start your legal intake today?",
  ]

  // Check API connectivity on component mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        await checkApiHealth()
        setApiConnected(true)
      } catch (error) {
        console.error("API health check failed:", error)
        setApiConnected(false)
      }
    }
    
    checkConnection()
    const intervalId = setInterval(checkConnection, 30000) // Check every 30 seconds
    
    return () => clearInterval(intervalId)
  }, [])

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Add a handler for the extracted text event
  useEffect(() => {
    const handleExtractedTextEvent = async (event: CustomEvent<{
      text: string;
      documentName: string;
      message: string;
      sessionId: string;
      caseFileId?: string;
    }>) => {
      const { text, documentName, message, sessionId, caseFileId } = event.detail;
      
      if (!sessionId || !text || text.trim().length === 0) return;
      
      try {
        // Process the extracted text with the API
        // Note: We don't show this message in the UI since it would duplicate the user's message
        console.log("Sending document text to API:", text.substring(0, 50) + "...");
        
        const response = await sendChatMessageWithDocumentText(
          message,
          text,
          documentName,
          sessionId,
          caseFileId
        );
        
        // No need to update UI as this is processed silently
        console.log("Document text processed successfully");
      } catch (error) {
        console.error("Error processing document text:", error);
        // While we don't update the UI, let's log details about the error
        if (error instanceof Error) {
          console.error("Error details:", error.message);
        }
        // No UI update needed as this is a background operation
      }
    };
    
    // Add the event listener
    window.addEventListener('extracted-text-ready', handleExtractedTextEvent);
    
    // Remove the event listener on component unmount
    return () => {
      window.removeEventListener('extracted-text-ready', handleExtractedTextEvent);
    };
  }, []);

  const handleSendMessage = async (message: string, attachments: File[]) => {
    // Don't send if there's no message and no attachments
    if (!message && attachments.length === 0) return;
    
    // Clear any previous API errors
    setApiError(null);
    
    // Add debugging for attachments
    if (attachments.length > 0) {
      console.log(`⬆️ handleSendMessage called with ${attachments.length} attachments:`, 
        attachments.map(file => `${file.name} (${file.type}, ${file.size} bytes)`));
    }

    // Create a user message with an attachment indicator if files are present
    let userContent = message;
    let fileAttachment = undefined;
    
    if (attachments.length > 0) {
      const file = attachments[0]; // For now we only handle the first file
      const fileNames = attachments.map(file => file.name).join(", ");
      userContent = `${message}\n\n[Attaching: ${fileNames}]`;
      
      // Create file attachment info
      fileAttachment = {
        name: file.name,
        type: file.type,
        isProcessing: true,
        documentId: 'pending' // Temporary ID that will be updated
      };
    }

    const userMessage: Message = {
      role: "user",
      content: userContent,
      timestamp: new Date().toISOString(),
      fileAttachment
    };

    // Update state with user message
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setShowWelcome(false);

    try {
      let response: ChatResponse;
      let retryCount = 0;
      const MAX_RETRIES = 2;
      
      const attemptSendMessage = async (): Promise<ChatResponse> => {
        try {
          // Handle message with file attachments
          if (attachments.length > 0) {
            // For now we'll just use the first file
            // In a more complete implementation, we might want to handle multiple files
            const file = attachments[0];
            console.log(`⚙️ Processing attachment: ${file.name} (${file.type}, ${file.size} bytes)`);
            
            // Update file status to uploading
            setMessages((prev) => prev.map(msg => {
              if (msg.role === 'user' && msg.fileAttachment && msg.fileAttachment.documentId === 'pending') {
                return {
                  ...msg,
                  fileAttachment: {
                    ...msg.fileAttachment,
                    isProcessing: true
                  }
                };
              }
              return msg;
            }));
            
            // Check if this is an image file
            const isImage = file.type.startsWith('image/');
            
            if (isImage) {
              console.log(`🖼️ Sending image through vision API: ${file.name}`);
              // Use the vision API for image files
              return await sendChatMessageWithImage(
                message,
                file,
                sessionId || "",
                caseFileId
              );
            } else {
              console.log(`📄 Sending file through regular file API: ${file.name}`);
              // Use regular file upload for non-image files
              return await sendChatMessageWithFile(
                message,
                file,
                sessionId || "",
                caseFileId
              );
            }
          } else {
            console.log(`💬 Sending text-only message: "${message.length > 50 ? message.substring(0, 50) + '...' : message}"`);
            // Handle text-only message
            return await sendChatMessage(
              message,
              sessionId || "",
              caseFileId
            );
          }
        } catch (error: any) {
          console.error("Error sending message:", error);
          
          // If sending failed, update the file attachment status
          if (attachments.length > 0) {
            setMessages((prev) => prev.map(msg => {
              if (msg.role === 'user' && msg.fileAttachment && msg.fileAttachment.documentId === 'pending') {
                return {
                  ...msg,
                  fileAttachment: {
                    ...msg.fileAttachment,
                    isProcessing: false,
                    error: error.message || "Upload failed"
                  }
                };
              }
              return msg;
            }));
          }
          
          // Check if error is retriable
          const isRetriable = 
            error.message.includes("timeout") || 
            error.message.includes("network") ||
            error.message.includes("failed to fetch") ||
            error.message.includes("aborted");
          
          if (isRetriable && retryCount < MAX_RETRIES) {
            retryCount++;
            console.warn(`Retrying message send (${retryCount}/${MAX_RETRIES})...`);
            
            // Add exponential backoff
            const backoffMs = 1000 * Math.pow(2, retryCount);
            await new Promise(resolve => setTimeout(resolve, backoffMs));
            
            return attemptSendMessage();
          }
          
          throw error;
        }
      };
      
      // Attempt to send the message
      response = await attemptSendMessage();
      
      // If the message was sent successfully, update session ID if needed
      if (!sessionId && response.session_id) {
        console.log(`🔄 Setting session ID: ${response.session_id}`);
        setSessionId(response.session_id);
      }
      
      // Check if we got a document ID back from the server for a file upload
      if (attachments.length > 0 && response.uploaded_document_id) {
        console.log(`📄 Got document ID from server: ${response.uploaded_document_id}`);
        
        // Find the user message we just added and update its file attachment with the actual document ID
        setMessages(prev => prev.map(msg => {
          if (msg.role === 'user' && msg.fileAttachment && msg.fileAttachment.documentId === 'pending') {
            return {
              ...msg,
              fileAttachment: {
                ...msg.fileAttachment,
                documentId: response.uploaded_document_id,
                isProcessing: false
              }
            };
          }
          return msg;
        }));
        setActiveDocumentId(response.uploaded_document_id);
      }
      
      // Add the assistant's response
      const assistantMessage: Message = {
        role: "assistant",
        content: response.message,
        timestamp: new Date().toISOString(),
        sources: response.sources,
        nextQuestions: response.next_questions
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      
      // If we have a case file ID, update the case summary
      if (caseFileId) {
        try {
          // Convert messages to the format expected by the API
          const chatHistoryForAPI = [...messages, assistantMessage].map(msg => ({
            role: msg.role,
            content: msg.content
          }));
          
          // Update the case summary in the background
          updateCaseSummaryFromChat(caseFileId, chatHistoryForAPI)
            .then(() => console.log('Case summary updated successfully'))
            .catch(error => console.error('Failed to update case summary:', error));
        } catch (error) {
          console.error('Error preparing chat history for summary update:', error);
        }
      }
      
    } catch (error: any) {
      console.error("Error sending message:", error);
      
      // Update file attachment status if there was an error
      if (attachments.length > 0) {
        setMessages((prev) => prev.map(msg => {
          if (msg.role === 'user' && msg.fileAttachment && msg.fileAttachment.documentId === 'pending') {
            return {
              ...msg,
              fileAttachment: {
                ...msg.fileAttachment,
                isProcessing: false,
                error: error.message || "Upload failed"
              }
            };
          }
          return msg;
        }));
      }
      
      let errorMessage = "An error occurred while sending your message.";
      
      // Handle specific error cases
      if (error.message.includes("timeout")) {
        errorMessage = "The request timed out. The document may be too large or the server is busy.";
      } else if (error.message.includes("Failed to fetch") || error.message.includes("network")) {
        errorMessage = "Unable to connect to the API server. Please check your internet connection.";
      } else if (error.message.includes("processing")) {
        errorMessage = "Error processing the document. The file may be corrupted or in an unsupported format.";
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      // Add an error message from the assistant
      const errorAssistantMessage: Message = {
        role: "assistant",
        content: `I'm sorry, I encountered an error: ${errorMessage}`,
        timestamp: new Date().toISOString(),
        isError: true
      };
      
      setMessages(prev => [...prev, errorAssistantMessage]);
      setApiError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    handleSendMessage(suggestion, [])
  }

  const handleFollowUpQuestion = (question: string) => {
    handleSendMessage(question, [])
  }

  // Add the error message to the JSX, right before the ChatInput component
  return (
    <div className="flex flex-col h-full overflow-hidden relative">
      {activeDocumentId && (
        <DocumentIssuesFab documentId={activeDocumentId} />
      )}
      
      <div className="flex-1 overflow-y-auto p-4 space-y-6" ref={messagesEndRef}>
        {showWelcome && messages.length === 0 ? (
          <div className="max-w-2xl mx-auto">
            <h1 className="text-3xl font-semibold tracking-tight text-gray-900">
              Hi there, <span className="text-purple-500">Client</span>
            </h1>
            <h2 className="text-2xl font-medium tracking-tight text-gray-800">
              <FlipWords words={legalPhrases} className="text-gray-800 font-medium !px-0" duration={3500} />
            </h2>
            <p className="text-gray-500 mt-2 mb-6 text-base font-normal">
              Use one of the most common prompts below or ask your own to begin
            </p>

            <div className="mb-8">
              <PromptSuggestions onSuggestionClick={handleSuggestionClick} />
            </div>
          </div>
        ) : (
          <div className="max-w-2xl mx-auto">
            {messages.map((message, index) => (
              <ChatMessage
                key={index}
                role={message.role}
                content={message.content}
                timestamp={message.timestamp}
                sources={message.sources}
                nextQuestions={index === messages.length - 1 && message.role === "assistant" ? message.nextQuestions : undefined}
                onFollowUpClick={handleFollowUpQuestion}
                fileAttachment={message.fileAttachment}
              />
            ))}
            {isLoading && (
              <div className="flex justify-start mb-6">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center flex-shrink-0 mt-1">
                    <span className="text-purple-600 text-sm font-medium">AI</span>
                  </div>
                  <div className="bg-gray-100 rounded-lg px-4 py-3 flex items-center">
                    <div className="flex space-x-1">
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0ms" }}
                      ></div>
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "150ms" }}
                      ></div>
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "300ms" }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* API Connection Status */}
      {apiConnected === false && (
        <div className="p-2 bg-red-100 text-red-800 text-center text-sm">
          Unable to connect to API server. Check if the server is running.
        </div>
      )}

      <ChatInput 
        onSendMessage={handleSendMessage} 
        isLoading={isLoading} 
        sessionId={sessionId}
        caseFileId={caseFileId}
        disabled={apiConnected === false}
      />
    </div>
  )
}
