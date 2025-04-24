"use client"

import { useState, useEffect, useRef } from "react"
import { Sidebar } from "@/components/sidebar"
import { PromptSuggestions } from "@/components/prompt-suggestions"
import { ChatInput } from "@/components/chat-input"
import { ChatMessage } from "./chat-message"
import { FlipWords } from "./ui/flip-words"
import { AlertTriangle, WifiOff, FileText, X } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { 
  sendChatMessage, 
  getChatHistory, 
  getSummaryFromSession,
  type Source, 
  type ChatMessage as ChatMessageType 
} from "@/lib/api"
import { DocumentIssuesFab } from "@/components/document-issues-fab"
import { Button } from "@/components/ui/button"

export function LegalChatInterface() {
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
  const [isLoadingHistory, setIsLoadingHistory] = useState(false)
  const [apiError, setApiError] = useState<string | null>(null)
  const [showWelcome, setShowWelcome] = useState(true)
  const [activeSessionId, setActiveSessionId] = useState<string | undefined>(undefined)
  const [activeCaseFileId, setActiveCaseFileId] = useState<string | undefined>(undefined)
  const [apiConnected, setApiConnected] = useState<boolean>(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  
  // Case summary state
  const [showSummaryPanel, setShowSummaryPanel] = useState(false)
  const [summaryContent, setSummaryContent] = useState<string>("")
  const [isSummaryLoading, setIsSummaryLoading] = useState(false)

  // Complete phrases for the FlipWords component
  const legalPhrases = [
    "Need help with case analysis today?",
    "Need a quick legal review today?",
    "Looking to analyze a case today?",
    "Want to start your legal intake today?",
  ]

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Load chat history when session changes
  useEffect(() => {
    if (activeSessionId) {
      loadChatHistory(activeSessionId)
    }
  }, [activeSessionId])

  const loadChatHistory = async (sessionId: string) => {
    setIsLoadingHistory(true)
    setApiError(null)

    try {
      const history = await getChatHistory(sessionId)

      // Convert API chat history to our message format
      const formattedMessages = history.map((msg: ChatMessageType) => ({
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp,
      }))

      setMessages(formattedMessages)
      setShowWelcome(false)
      setApiConnected(true)
    } catch (error) {
      console.error("Error loading chat history:", error)

      if (error instanceof Error) {
        setApiError(`Failed to load chat history: ${error.message}`)

        // Check if it's a connection error
        if (
          error.message.includes("Failed to fetch") ||
          error.message.includes("Unable to connect") ||
          error.message.includes("timed out")
        ) {
          setApiConnected(false)
        }
      } else {
        setApiError("Failed to load chat history")
      }

      // Keep showing welcome screen if history fails to load
      setShowWelcome(true)
    } finally {
      setIsLoadingHistory(false)
    }
  }

  const handleSessionSelect = (sessionId: string) => {
    setActiveSessionId(sessionId)
    // Chat history will be loaded by the useEffect
  }

  const handleNewChat = () => {
    setActiveSessionId(undefined)
    setActiveCaseFileId(undefined)
    setMessages([])
    setShowWelcome(true)
    setApiError(null)
  }

  const handleSendMessage = async (message: string, attachments: File[]) => {
    if (!message.trim() && attachments.length === 0) return

    // Add user message to chat
    const userMessage = {
      role: "user" as const,
      content: message,
      timestamp: new Date().toISOString(),
    }

    setMessages((prev) => [...prev, userMessage])
    setIsLoading(true)
    setShowWelcome(false)
    setApiError(null)

    try {
      // Send message to API
      const response = await sendChatMessage(message, activeSessionId, activeCaseFileId)

      // Update session ID if it's a new conversation
      if (response.session_id && !activeSessionId) {
        setActiveSessionId(response.session_id)
      }

      // Add assistant response to chat
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.message,
          timestamp: new Date().toISOString(),
          sources: response.sources,
          nextQuestions: response.next_questions,
        },
      ])

      // If we get here, the API is connected
      setApiConnected(true)
    } catch (error) {
      console.error("Error sending message:", error)

      if (error instanceof Error) {
        setApiError(error.message)

        // Check if it's a connection error
        if (
          error.message.includes("Failed to fetch") ||
          error.message.includes("Unable to connect") ||
          error.message.includes("timed out")
        ) {
          setApiConnected(false)
        }
      } else {
        setApiError("An unexpected error occurred. Please try again.")
      }

      // Add error message to chat
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I'm sorry, there was an error processing your request. Please try again.",
          timestamp: new Date().toISOString(),
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const handleFollowUpQuestion = (question: string) => {
    handleSendMessage(question, [])
  }

  // Add case file selection handler
  const handleCaseFileSelect = (caseFileId: string) => {
    setActiveCaseFileId(caseFileId)
    // Reset any previous summary when changing case files
    setSummaryContent("")
    setShowSummaryPanel(false)
  }

  // Generate case summary from chat history
  const generateCaseSummary = async () => {
    if (!activeSessionId || messages.length === 0) {
      // Show error or notification that we need messages
      setApiError("Please send some messages before generating a summary")
      return
    }

    setIsSummaryLoading(true)
    setApiError(null)

    try {
      // Get the summary directly from the session
      const response = await getSummaryFromSession(activeSessionId)
      setSummaryContent(response.summary)
      
      // Show the summary panel
      setShowSummaryPanel(true)
    } catch (error) {
      console.error("Error generating case summary:", error)
      
      if (error instanceof Error) {
        setApiError(`Failed to generate summary: ${error.message}`)
      } else {
        setApiError("An unexpected error occurred generating the summary")
      }
    } finally {
      setIsSummaryLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-screen w-full bg-white">
      <div className="flex flex-1 overflow-hidden">
        <Sidebar 
          onSessionSelect={handleSessionSelect} 
          onNewChat={handleNewChat} 
          onCaseFileSelect={handleCaseFileSelect} 
          activeSessionId={activeSessionId}
          activeCaseFileId={activeCaseFileId}
        />
        <div className="flex-1 flex flex-col h-full overflow-hidden bg-white relative">
          {/* API Connection Status */}
          {!apiConnected && (
            <div className="bg-red-50 border-b border-red-200 py-2 px-4">
              <div className="flex items-center justify-center gap-2 text-red-600">
                <WifiOff size={16} />
                <span className="text-sm">
                  Unable to connect to API server. Make sure it's running at http://localhost:8000
                </span>
              </div>
            </div>
          )}

          <div className="flex-1 overflow-y-auto bg-white">
            <div className="max-w-2xl w-full px-4 py-4 md:px-8 mx-auto">
              {showWelcome ? (
                <>
                  <h1 className="text-3xl font-semibold tracking-tight text-gray-900">
                    Hi there, <span className="text-purple-500">Client</span>
                  </h1>
                  <h2 className="text-2xl font-medium tracking-tight text-gray-800">
                    <FlipWords words={legalPhrases} className="text-gray-800 font-medium !px-0" duration={3500} />
                  </h2>
                  <p className="text-gray-500 mt-2 mb-6 text-base font-normal">
                    Use one of the most common prompts below or ask your own to begin
                  </p>

                  <PromptSuggestions onSuggestionClick={(suggestion) => handleSendMessage(suggestion, [])} />
                </>
              ) : (
                <div className="mb-6">
                  {isLoadingHistory ? (
                    <div className="flex justify-center py-10">
                      <div className="inline-block h-8 w-8 border-4 border-gray-200 border-t-purple-500 rounded-full animate-spin"></div>
                    </div>
                  ) : (
                    <>
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
                    </>
                  )}
                </div>
              )}

              {apiError && (
                <Alert className="bg-red-50 border border-red-200 mb-6">
                  <AlertTriangle className="h-4 w-4 text-red-500" />
                  <AlertDescription className="text-red-600 text-sm font-normal">{apiError}</AlertDescription>
                </Alert>
              )}
            </div>
          </div>

          <div className="p-4 border-t border-gray-200 relative">
            <div className="max-w-2xl mx-auto">
              {/* Case Summary Button - show whenever we have messages */}
              {activeSessionId && messages.length > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  className="absolute right-4 top-[-40px] bg-white border-gray-200 flex items-center gap-1 shadow-sm"
                  onClick={generateCaseSummary}
                  disabled={isSummaryLoading}
                >
                  {isSummaryLoading ? (
                    <div className="h-4 w-4 border-2 border-gray-200 border-t-purple-500 rounded-full animate-spin" />
                  ) : (
                    <FileText size={16} className="text-purple-500" />
                  )}
                  <span className="text-xs font-medium">
                    {isSummaryLoading ? "Generating..." : "Summarize Chat"}
                  </span>
                </Button>
              )}
              
              <ChatInput
                onSendMessage={handleSendMessage}
                isLoading={isLoading}
                sessionId={activeSessionId}
                caseFileId={activeCaseFileId}
                disabled={!apiConnected}
              />
            </div>
          </div>
          
          {/* Case Summary Side Panel */}
          {showSummaryPanel && (
            <div className="absolute top-4 right-4 w-96 max-h-[70vh] bg-white rounded-lg border border-gray-200 shadow-lg flex flex-col z-20 overflow-hidden">
              <div className="p-3 border-b border-gray-200 flex justify-between items-center bg-purple-50">
                <div className="flex items-center gap-2">
                  <FileText size={16} className="text-purple-500" />
                  <h3 className="font-medium text-sm">Case Summary</h3>
                </div>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  className="h-7 w-7 p-0 hover:bg-purple-100" 
                  onClick={() => setShowSummaryPanel(false)}
                >
                  <X size={14} />
                </Button>
              </div>
              <div className="p-4 overflow-y-auto max-h-[calc(70vh-3rem)]">
                {isSummaryLoading ? (
                  <div className="flex flex-col items-center justify-center py-10 space-y-4">
                    <div className="h-8 w-8 border-4 border-gray-200 border-t-purple-500 rounded-full animate-spin" />
                    <p className="text-gray-500 text-center">
                      Generating summary from your conversation...<br/>
                      <span className="text-xs">This may take a few minutes for longer conversations.</span>
                    </p>
                  </div>
                ) : (
                  <div className="prose prose-sm max-w-none">
                    {summaryContent ? (
                      <div dangerouslySetInnerHTML={{ __html: summaryContent }} />
                    ) : (
                      <p className="text-gray-500 italic">No summary available yet</p>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Add Document Issues FAB */}
      <DocumentIssuesFab />
    </div>
  )
}
