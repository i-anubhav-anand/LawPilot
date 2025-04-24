import { useState, useRef, useEffect } from 'react'
import { X, AlertTriangle, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { cn } from "@/lib/utils"

interface Issue {
  id: string
  type: 'warning' | 'error' | 'info'
  title: string
  description: string
}

interface IssuePanelProps {
  isOpen: boolean
  onClose: () => void
  issues: Issue[]
  isLoading?: boolean
  error?: string | null
  className?: string
}

export function IssuePanel({ 
  isOpen, 
  onClose, 
  issues, 
  isLoading = false,
  error = null,
  className 
}: IssuePanelProps) {
  const [animationClass, setAnimationClass] = useState('')
  const panelRef = useRef<HTMLDivElement>(null)

  // Handle outside clicks
  useEffect(() => {
    if (!isOpen) return

    const handleOutsideClick = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        onClose()
      }
    }

    document.addEventListener('mousedown', handleOutsideClick)
    return () => document.removeEventListener('mousedown', handleOutsideClick)
  }, [isOpen, onClose])

  // Handle animations
  useEffect(() => {
    if (isOpen) {
      setAnimationClass('translate-y-0 opacity-100')
    } else {
      setAnimationClass('translate-y-full opacity-0')
    }
  }, [isOpen])

  // Issue type icons
  const getIssueIcon = (type: Issue['type']) => {
    switch (type) {
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-500" />
      case 'info':
        return <CheckCircle className="h-5 w-5 text-blue-500" />
    }
  }

  // Issue type colors
  const getIssueColor = (type: Issue['type']) => {
    switch (type) {
      case 'warning':
        return 'border-yellow-200 bg-yellow-50'
      case 'error':
        return 'border-red-200 bg-red-50'
      case 'info':
        return 'border-blue-200 bg-blue-50'
    }
  }

  if (!isOpen && animationClass === 'translate-y-full opacity-0') {
    return null
  }

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="text-center py-8 text-gray-500">
          <Loader2 className="h-10 w-10 text-purple-500 mx-auto mb-3 animate-spin" />
          <p>Loading document issues...</p>
        </div>
      )
    }
    
    if (error) {
      return (
        <div className="text-center py-8 text-gray-500">
          <AlertCircle className="h-10 w-10 text-red-500 mx-auto mb-3" />
          <p className="text-red-600">{error}</p>
        </div>
      )
    }
    
    if (issues.length === 0) {
      return (
        <div className="text-center py-8 text-gray-500">
          <CheckCircle className="h-10 w-10 text-green-500 mx-auto mb-3" />
          <p>No issues found</p>
        </div>
      )
    }
    
    return (
      <ul className="space-y-3">
        {issues.map((issue) => (
          <li 
            key={issue.id} 
            className={cn(
              "p-3 rounded-md border",
              getIssueColor(issue.type)
            )}
          >
            <div className="flex">
              <div className="flex-shrink-0 mr-3">
                {getIssueIcon(issue.type)}
              </div>
              <div>
                <h4 className="font-medium text-gray-900">{issue.title}</h4>
                <p className="text-sm text-gray-600 mt-1">{issue.description}</p>
              </div>
            </div>
          </li>
        ))}
      </ul>
    )
  }

  return (
    <div 
      className={cn(
        "fixed bottom-20 right-6 z-50 w-80 max-h-[70vh] rounded-lg shadow-xl border border-gray-200 bg-white",
        "transform transition-all duration-300 ease-in-out",
        animationClass,
        className
      )}
      ref={panelRef}
    >
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <h3 className="font-medium text-gray-900">
          Document Issues {!isLoading && !error && `(${issues.length})`}
        </h3>
        <button 
          onClick={onClose}
          className="text-gray-500 hover:text-gray-700 transition-colors rounded-full p-1"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      
      <div className="overflow-y-auto max-h-[calc(70vh-4rem)] p-3">
        {renderContent()}
      </div>
    </div>
  )
} 