"use client"

import { useEffect, useRef, useState } from "react"
import { ChatMessage } from "./chat-message"
import { ChatInput } from "./chat-input"
import { PromptSuggestions } from "./prompt-suggestions"
import { FlipWords } from "./ui/flip-words"
import { 
  sendChatMessage, 
  sendChatMessageWithFile, 
  sendChatMessageWithDocumentText, 
  uploadDocument,
  type Source, 
  checkApiHealth,
  ChatResponse,
  sendChatMessageWithImage
} from "@/lib/api"
import { Message } from "@/types/chat"
// Import the ErrorMessage component
import { ErrorMessage } from "./error-message"

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
  const [messages, setMessages] = useState<
    {
      role: "user" | "assistant"
      content: string
      timestamp?: string
      sources?: Source[]
      nextQuestions?: string[]
    }[]
  >([])
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string | undefined>(initialSessionId)
  const [showWelcome, setShowWelcome] = useState(true)
  const [apiConnected, setApiConnected] = useState<boolean | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  // Add a state for API connection errors
  const [apiError, setApiError] = useState<string | null>(null)

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
        setApiError("Unable to connect to the Legal AI Assistant API. Please ensure the API server is running.")
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

    // Create a user message
    const userMessage: Message = {
      role: "user",
      content: message,
      timestamp: new Date().toISOString(),
    };

    // Update state with user message
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      let response: ChatResponse;

      // Handle message with file attachments
      if (attachments.length > 0) {
        // For now we'll just use the first file
        // In a more complete implementation, we might want to handle multiple files
        const file = attachments[0];
        
        // Check if this is an image file
        const isImage = file.type.startsWith('image/');
        
        if (isImage) {
          // Use the vision API for image files
          response = await sendChatMessageWithImage(
            message,
            file,
            sessionId || "",
            caseFileId
          );
        } else {
          // Use regular file upload for non-image files
          response = await sendChatMessageWithFile(
            message,
            file,
            sessionId || "",
            caseFileId
          );
        }
      } else {
        // Handle text-only message
        response = await sendChatMessage(
          message,
          sessionId || "",
          caseFileId
        );
      }

      // Update session ID if needed
      if (!sessionId && response.session_id) {
        setSessionId(response.session_id);
      }

      // Add assistant response
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.message,
          timestamp: new Date().toISOString(),
          sources: response.sources || [],
          nextQuestions: response.next_questions || [],
        },
      ]);
    } catch (error) {
      console.error("Error sending message:", error);
      
      // Add error message
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: error instanceof Error 
            ? error.message 
            : "An error occurred while processing your message.",
          timestamp: new Date().toISOString(),
          isError: true,
        },
      ]);
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
    <div className="flex flex-col h-full bg-white">
      {/* API Connection Status */}
      {apiConnected === false && (
        <div className="bg-red-50 border-b border-red-200 py-2 px-4">
          <div className="flex items-center justify-center gap-2 text-red-600">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" 
                 stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M1 1l22 22"></path>
              <path d="M16.72 11.06A10.94 10.94 0 0 1 19 12.55"></path>
              <path d="M5 12.55a10.94 10.94 0 0 1 5.17-2.39"></path>
              <path d="M10.71 5.05A16 16 0 0 1 22.58 9"></path>
              <path d="M1.42 9a15.91 15.91 0 0 1 4.7-2.88"></path>
              <path d="M8.53 16.11a6 6 0 0 1 6.95 0"></path>
              <path d="M12 20h.01"></path>
            </svg>
            <span className="text-sm">
              Unable to connect to API server. Make sure it's running at http://localhost:8000
            </span>
          </div>
        </div>
      )}

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4 bg-white text-gray-900">
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
                nextQuestions={message.nextQuestions}
                onFollowUpClick={handleFollowUpQuestion}
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

      {/* Input area */}
      <div className="p-4 border-t border-gray-200">
        <div className="max-w-2xl mx-auto">
          {apiError && (
            <ErrorMessage
              message={apiError}
              suggestion="Make sure the API server is running and accessible at http://localhost:8000"
            />
          )}
          <ChatInput 
            onSendMessage={handleSendMessage} 
            isLoading={isLoading} 
            sessionId={sessionId}
            caseFileId={caseFileId}
            disabled={apiConnected === false}
          />
        </div>
      </div>
    </div>
  )
}
