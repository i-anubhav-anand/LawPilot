import { useState, useCallback, useEffect } from 'react'
import { analyzeDocument, DocumentAnalysisResponse } from '@/lib/api';

export interface DocumentIssue {
  id: string
  type: 'warning' | 'error' | 'info'
  title: string
  description: string
}

export function useDocumentIssues(documentId?: string) {
  const [issues, setIssues] = useState<DocumentIssue[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch document issues from the API
  useEffect(() => {
    if (!documentId) return;
    
    const fetchIssues = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const analysis = await analyzeDocument(documentId);
        
        // Convert the API response format to our DocumentIssue format
        const formattedIssues = analysis.issues.map((issue, index) => ({
          id: `${documentId}-issue-${index}`,
          type: issue.type,
          title: issue.title,
          description: issue.description
        }));
        
        setIssues(formattedIssues);
      } catch (err) {
        console.error('Error fetching document issues:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch document issues');
        // Don't clear existing issues on error to maintain UI state
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchIssues();
  }, [documentId]);

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
    isLoading,
    error,
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