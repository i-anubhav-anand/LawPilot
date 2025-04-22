"use client"

import type { Source } from "@/lib/api"
import ReactMarkdown from 'react-markdown'

interface ChatMessageProps {
  role: "user" | "assistant"
  content: string
  timestamp?: string
  sources?: Source[]
  nextQuestions?: string[]
  onFollowUpClick?: (question: string) => void
}

export function ChatMessage({ role, content, timestamp, sources, nextQuestions, onFollowUpClick }: ChatMessageProps) {
  const formattedTime = timestamp
    ? new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : ""

  return (
    <div className={`flex flex-col ${role === "user" ? "items-end" : "items-start"} mb-6`}>
      <div className="flex items-start gap-3 max-w-[85%]">
        {role === "assistant" && (
          <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center flex-shrink-0 mt-1">
            <span className="text-purple-600 text-sm font-medium">AI</span>
          </div>
        )}

        <div
          className={`rounded-lg px-4 py-3 ${
            role === "user" ? "bg-purple-500 text-white" : "bg-gray-100 text-gray-800"
          }`}
        >
          {role === "assistant" ? (
            <div className="prose prose-sm max-w-none 
              prose-h1:text-2xl prose-h1:font-bold prose-h1:mb-3 prose-h1:mt-4
              prose-h2:text-xl prose-h2:font-semibold prose-h2:mb-2 prose-h2:mt-3
              prose-h3:text-lg prose-h3:font-medium prose-h3:mb-2 prose-h3:mt-3
              prose-p:mb-2 
              prose-a:text-blue-600 prose-a:hover:underline 
              prose-strong:font-bold 
              prose-em:italic
              prose-ol:list-decimal prose-ol:pl-5 prose-ol:my-2 prose-ol:space-y-1
              prose-ul:list-disc prose-ul:pl-5 prose-ul:my-2 prose-ul:space-y-1
              prose-li:mb-1
              prose-blockquote:border-l-4 prose-blockquote:border-gray-300 prose-blockquote:pl-4 prose-blockquote:italic prose-blockquote:my-3
              prose-code:bg-gray-200 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-sm
              prose-pre:bg-gray-800 prose-pre:text-gray-100 prose-pre:p-3 prose-pre:rounded-md prose-pre:overflow-x-auto
              ">
              <ReactMarkdown>
                {content}
              </ReactMarkdown>
            </div>
          ) : (
            <div className="whitespace-pre-wrap">{content}</div>
          )}
        </div>

        {role === "user" && (
          <div className="w-8 h-8 rounded-full bg-gray-300 overflow-hidden flex-shrink-0 mt-1">
            <img src="/confident-professional.png" alt="User" className="w-full h-full object-cover" />
          </div>
        )}
      </div>

      {/* Sources section */}
      {sources && sources.length > 0 && (
        <div className="mt-3 ml-11 max-w-[85%]">
          <div className="text-sm font-medium text-gray-700 mb-2">Sources:</div>
          <div className="space-y-2">
            {sources.map((source, index) => (
              <div key={index} className="bg-white border border-gray-200 rounded-md p-3 text-sm">
                <div className="font-medium">{source.title}</div>
                <div className="text-gray-600 mt-1">{source.content}</div>
                <div className="text-xs text-gray-500 mt-1">Citation: {source.citation}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Next questions section - now with click functionality */}
      {nextQuestions && nextQuestions.length > 0 && (
        <div className="mt-3 ml-11 max-w-[85%]">
          <div className="text-sm font-medium text-gray-700 mb-2">Follow-up questions:</div>
          <div className="flex flex-wrap gap-2">
            {nextQuestions.map((question, index) => (
              <button
                key={index}
                className="bg-white border border-gray-200 hover:border-purple-300 hover:bg-purple-50 rounded-full px-3 py-1.5 text-sm transition-colors"
                onClick={() => onFollowUpClick && onFollowUpClick(question)}
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
