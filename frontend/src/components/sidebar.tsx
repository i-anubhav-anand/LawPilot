"use client"

import { Plus, ChevronRight, MessageSquare, Scale, RefreshCw, AlertCircle } from "lucide-react"
import { useEffect, useState } from "react"
import { getChatSessions, type ChatSession } from "@/lib/api"

interface SidebarProps {
  onSessionSelect?: (sessionId: string) => void
  onNewChat?: () => void
  activeSessionId?: string
}

export function Sidebar({ onSessionSelect, onNewChat, activeSessionId }: SidebarProps) {
  const [activeItem, setActiveItem] = useState<string | null>(activeSessionId || "new-chat")
  const [expanded, setExpanded] = useState(false)
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [apiError, setApiError] = useState<boolean>(false)
  const [retryCount, setRetryCount] = useState(0)

  // Fetch chat sessions
  const fetchSessions = async () => {
    setIsLoading(true)
    setApiError(false)

    try {
      const sessions = await getChatSessions()
      setChatSessions(sessions)
      // If we get here, the API is working
      setApiError(false)
    } catch (error) {
      console.error("Error fetching chat sessions:", error)
      setApiError(true)
      setChatSessions([])
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (expanded) {
      fetchSessions()
    }
  }, [expanded, retryCount])

  // Update active item when activeSessionId changes
  useEffect(() => {
    if (activeSessionId) {
      setActiveItem(activeSessionId)
    }
  }, [activeSessionId])

  const handleNewChatClick = () => {
    setActiveItem("new-chat")
    if (onNewChat) {
      onNewChat()
    }
  }

  const handleSessionClick = (sessionId: string) => {
    setActiveItem(sessionId)
    if (onSessionSelect) {
      onSessionSelect(sessionId)
    }
  }

  const handleRetry = () => {
    setRetryCount((prev) => prev + 1)
  }

  // Format session ID for display
  const formatSessionId = (sessionId: string) => {
    // If it's a UUID, take the first 8 characters
    if (sessionId.includes("-")) {
      return sessionId.split("-")[0]
    }

    // If it's a timestamp-based ID like "session_1745316533.302385"
    if (sessionId.startsWith("session_")) {
      const parts = sessionId.split("_")
      if (parts.length > 1) {
        // Try to format as a date if it's a timestamp
        try {
          const timestamp = Number.parseFloat(parts[1])
          if (!isNaN(timestamp)) {
            const date = new Date(timestamp * 1000)
            return date.toLocaleDateString()
          }
        } catch (e) {
          // If parsing fails, just return the raw ID
        }
      }
    }

    // Default fallback
    return sessionId
  }

  return (
    <div
      className={`border-r border-gray-200 bg-white flex flex-col h-full sticky top-0 transition-all duration-300 ease-in-out ${
        expanded ? "w-64" : "w-[72px]"
      }`}
    >
      {/* Logo and expand toggle */}
      <div className="flex items-center justify-between p-4">
        <div className={`flex items-center ${expanded ? "" : "justify-center w-full"}`}>
          <div className="w-9 h-9 bg-black text-white rounded-md flex items-center justify-center">
            <Scale size={18} />
          </div>
          {expanded && <span className="ml-3 font-medium">Legal Advisor</span>}
        </div>
        {expanded && (
          <button onClick={() => setExpanded(false)} className="text-gray-500 hover:text-gray-700">
            <ChevronRight size={18} />
          </button>
        )}
        {!expanded && (
          <button
            onClick={() => setExpanded(true)}
            className="absolute right-0 top-4 w-4 h-8 bg-gray-100 rounded-l-md flex items-center justify-center hover:bg-gray-200 transition-colors"
          >
            <ChevronRight size={14} />
          </button>
        )}
      </div>

      <div className="flex flex-col flex-1 overflow-hidden mt-4">
        {/* New Chat Button */}
        <div className="relative">
          <label
            htmlFor="new-chat"
            className={`relative mx-3 h-10 p-2 ease-in-out duration-300 flex items-center justify-center rounded-lg cursor-pointer ${
              activeItem === "new-chat" ? "shadow-md border border-purple-200 bg-purple-50/50" : "hover:bg-gray-100"
            } ${expanded ? "justify-start px-3" : ""}`}
            onClick={handleNewChatClick}
          >
            <input
              className="hidden"
              type="radio"
              name="sidebar-nav"
              id="new-chat"
              checked={activeItem === "new-chat"}
              onChange={() => {}}
            />
            <Plus
              size={18}
              className={`transition-all duration-300 ${
                activeItem === "new-chat" ? "text-purple-500" : "text-gray-700 hover:text-purple-500"
              }`}
            />
            {expanded && <span className="ml-3 text-sm">New Chat</span>}
          </label>
          {!expanded && (
            <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 bg-gray-800 text-white text-xs py-1 px-2 rounded opacity-0 group-hover:opacity-100 pointer-events-none whitespace-nowrap transition-opacity duration-200 z-50 invisible group-hover:visible">
              New Chat
            </div>
          )}
        </div>

        {/* Divider */}
        <div className="mx-3 my-4 border-t border-gray-200"></div>

        {/* Chat History Section */}
        <div className={`px-3 ${expanded ? "" : "text-center"} relative group`}>
          <div className="flex items-center justify-between">
            <h3 className={`text-xs font-medium text-gray-500 mb-2 ${expanded ? "" : "sr-only"}`}>Recent Cases</h3>
            {expanded && apiError && (
              <div className="flex items-center text-xs text-red-500 mb-2">
                <AlertCircle size={12} className="mr-1" />
                <span>API Error</span>
              </div>
            )}
          </div>

          {/* Chat history list - only visible when expanded */}
          {expanded ? (
            <div className="space-y-1 overflow-y-auto max-h-[calc(100vh-200px)]">
              {isLoading ? (
                <div className="text-center py-4">
                  <div className="inline-block h-4 w-4 border-2 border-gray-300 border-t-purple-500 rounded-full animate-spin"></div>
                </div>
              ) : apiError ? (
                <div className="text-center py-4">
                  <div className="bg-red-50 border border-red-200 rounded-md p-3 mb-2">
                    <p className="text-sm text-red-600 mb-2">Unable to connect to API server</p>
                    <p className="text-xs text-gray-600 mb-3">
                      Make sure the API server is running at http://localhost:8000
                    </p>
                    <button
                      onClick={handleRetry}
                      className="text-xs bg-red-100 hover:bg-red-200 text-red-700 py-1 px-3 rounded-md transition-colors"
                    >
                      Retry Connection
                    </button>
                  </div>
                </div>
              ) : chatSessions.length > 0 ? (
                chatSessions.map((session) => (
                  <button
                    key={session.session_id}
                    className={`w-full text-left p-2 rounded-lg ${
                      activeItem === session.session_id ? "bg-purple-50 border border-purple-200" : "hover:bg-gray-100"
                    }`}
                    onClick={() => handleSessionClick(session.session_id)}
                  >
                    <div className="flex items-start">
                      <MessageSquare size={16} className="text-gray-500 mt-0.5 flex-shrink-0" />
                      <div className="ml-2 overflow-hidden">
                        <p className="text-sm font-medium truncate">Session {formatSessionId(session.session_id)}</p>
                        <p className="text-xs text-gray-500">{session.message_count} messages</p>
                      </div>
                    </div>
                  </button>
                ))
              ) : (
                <div className="text-center py-4 text-sm text-gray-500">No chat history</div>
              )}

              <button
                className="w-full mt-2 flex items-center justify-center gap-1 text-xs text-gray-500 hover:text-purple-500 py-2"
                onClick={handleRetry}
              >
                <RefreshCw size={12} />
                <span>Refresh</span>
              </button>
            </div>
          ) : (
            // Chat history icon when collapsed
            <div className="relative group">
              <label
                htmlFor="chat-history"
                className={`relative w-full h-10 p-2 ease-in-out duration-300 flex items-center justify-center rounded-lg cursor-pointer ${
                  activeItem === "chat-history"
                    ? "shadow-md border border-purple-200 bg-purple-50/50"
                    : "hover:bg-gray-100"
                }`}
                onClick={() => {
                  setActiveItem("chat-history")
                  setExpanded(true)
                }}
              >
                <input
                  className="hidden"
                  type="radio"
                  name="sidebar-nav"
                  id="chat-history"
                  checked={activeItem === "chat-history"}
                  onChange={() => {}}
                />
                <MessageSquare
                  size={18}
                  className={`transition-all duration-300 ${
                    activeItem === "chat-history" ? "text-purple-500" : "text-gray-700 hover:text-purple-500"
                  }`}
                />
                {apiError && <span className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full"></span>}
              </label>
              <div className="absolute left-full ml-2 top-1/2 -translate-y-1/2 bg-gray-800 text-white text-xs py-1 px-2 rounded opacity-0 group-hover:opacity-100 pointer-events-none whitespace-nowrap transition-opacity duration-200 z-50 invisible group-hover:visible">
                View Cases & History
              </div>
            </div>
          )}
        </div>
      </div>

      {/* User Profile */}
      <div className="mt-auto border-t border-gray-200 p-4 flex items-center">
        <div className="w-7 h-7 rounded-full bg-gray-300 overflow-hidden hover:ring-2 hover:ring-purple-300 transition-all duration-300">
          <img src="/confident-professional.png" alt="User avatar" className="w-full h-full object-cover" />
        </div>
        {expanded && (
          <div className="ml-3">
            <p className="text-sm font-medium">Client Name</p>
            <p className="text-xs text-gray-500">client@example.com</p>
          </div>
        )}
      </div>
    </div>
  )
}
