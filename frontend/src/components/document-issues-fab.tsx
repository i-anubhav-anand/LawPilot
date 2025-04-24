import { useState } from 'react'
import { AlertTriangle, Loader2 } from 'lucide-react'
import { FloatingActionButton } from './ui/floating-action-button'
import { IssuePanel } from './issue-panel'
import { useDocumentIssues, DocumentIssue } from '@/hooks/use-document-issues'

interface DocumentIssuesFabProps {
  documentId?: string
  // Allow passing issues directly in case they're available from a parent component
  issues?: DocumentIssue[]
  className?: string
}

export function DocumentIssuesFab({ 
  documentId,
  issues: passedIssues,
  className 
}: DocumentIssuesFabProps) {
  const [isPanelOpen, setIsPanelOpen] = useState(false)
  const { 
    issues: fetchedIssues, 
    isLoading, 
    error,
    totalCount 
  } = useDocumentIssues(documentId)
  
  // Use passed issues if available, otherwise use fetched issues
  const issues = passedIssues || fetchedIssues
  
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
            {isLoading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <AlertTriangle className="h-5 w-5" />
            )}
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
        isLoading={isLoading}
        error={error}
      />
    </>
  )
} 