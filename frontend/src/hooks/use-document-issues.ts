import { useState, useCallback } from 'react'

export interface DocumentIssue {
  id: string
  type: 'warning' | 'error' | 'info'
  title: string
  description: string
}

export function useDocumentIssues() {
  const [issues, setIssues] = useState<DocumentIssue[]>([])

  const addIssue = useCallback((issue: Omit<DocumentIssue, 'id'>) => {
    const newIssue: DocumentIssue = {
      ...issue,
      id: Math.random().toString(36).substring(2, 9) // Generate a simple unique ID
    }
    setIssues(prev => [...prev, newIssue])
    return newIssue.id
  }, [])

  const removeIssue = useCallback((id: string) => {
    setIssues(prev => prev.filter(issue => issue.id !== id))
  }, [])

  const clearIssues = useCallback(() => {
    setIssues([])
  }, [])

  const getIssuesByType = useCallback((type: DocumentIssue['type']) => {
    return issues.filter(issue => issue.type === type)
  }, [issues])

  return {
    issues,
    addIssue,
    removeIssue,
    clearIssues,
    getIssuesByType,
    errorCount: issues.filter(i => i.type === 'error').length,
    warningCount: issues.filter(i => i.type === 'warning').length,
    infoCount: issues.filter(i => i.type === 'info').length,
    totalCount: issues.length
  }
}