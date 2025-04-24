import { useState } from 'react'
import { AlertTriangle } from 'lucide-react'
import { FloatingActionButton } from './ui/floating-action-button'
import { IssuePanel } from './issue-panel'

// Sample data for demonstration purposes
const SAMPLE_ISSUES = [
  {
    id: '1',
    type: 'error' as const,
    title: 'Missing tenant signature',
    description: 'The lease agreement is missing the tenant signature on page 3.'
  },
  {
    id: '2',
    type: 'warning' as const,
    title: 'Unclear rent increase clause',
    description: 'The clause regarding rent increases in section 8.2 is ambiguous and may not be enforceable.'
  },
  {
    id: '3',
    type: 'info' as const,
    title: 'Security deposit limit',
    description: 'In San Francisco, security deposits cannot exceed two months\' rent for unfurnished units.'
  }
]

interface DocumentIssuesFabProps {
  issues?: Array<{
    id: string
    type: 'warning' | 'error' | 'info'
    title: string
    description: string
  }>
  className?: string
}

export function DocumentIssuesFab({ 
  issues = SAMPLE_ISSUES, 
  className 
}: DocumentIssuesFabProps) {
  const [isPanelOpen, setIsPanelOpen] = useState(false)
  
  const togglePanel = () => {
    setIsPanelOpen(prev => !prev)
  }

  return (
    <>
      {/* Floating Action Button */}
      <FloatingActionButton
        size="default"
        variant={issues.length > 0 ? 'error' : 'secondary'}
        icon={
          <>
            <AlertTriangle className="h-5 w-5" />
            {issues.length > 0 && (
              <span className="absolute -top-1 -right-1 bg-red-600 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                {issues.length}
              </span>
            )}
          </>
        }
        onClick={togglePanel}
        className="fixed bottom-5 right-5 z-50 shadow-lg"
        aria-label="Show document issues"
      />

      {/* Issues Panel */}
      <IssuePanel
        isOpen={isPanelOpen}
        onClose={() => setIsPanelOpen(false)}
        issues={issues}
      />
    </>
  )
} 