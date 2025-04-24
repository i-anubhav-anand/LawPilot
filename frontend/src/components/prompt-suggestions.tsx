"use client"

import { FileQuestion, FileText, Gavel, Scale, RefreshCw } from "lucide-react"

const suggestions = [
  {
    title: "Explain my legal rights in a tenant dispute",
    icon: <Scale className="h-5 w-5 text-gray-500" />,
  },
  {
    title: "Draft a cease and desist letter template",
    icon: <FileText className="h-5 w-5 text-gray-500" />,
  },
  {
    title: "Summarize contract terms and highlight risks",
    icon: <FileQuestion className="h-5 w-5 text-gray-500" />,
  },
  {
    title: "Explain the legal process for small claims court",
    icon: <Gavel className="h-5 w-5 text-gray-500" />,
  },
]

interface PromptSuggestionsProps {
  onSuggestionClick?: (suggestion: string) => void
}

export function PromptSuggestions({ onSuggestionClick }: PromptSuggestionsProps) {
  return (
    <div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            className="py-5 px-4 border border-gray-200 rounded-lg hover:border-purple-200 hover:bg-purple-50/30 transition-colors text-left flex items-start gap-4"
            onClick={() => onSuggestionClick?.(suggestion.title)}
          >
            <div className="mt-0.5">{suggestion.icon}</div>
            <span className="text-sm text-gray-700 font-normal">{suggestion.title}</span>
          </button>
        ))}
      </div>

      <button className="mt-4 flex items-center gap-2 text-sm text-gray-500 hover:text-purple-500 transition-colors">
        <RefreshCw className="h-4 w-4" />
        <span>Refresh Prompts</span>
      </button>
    </div>
  )
}
