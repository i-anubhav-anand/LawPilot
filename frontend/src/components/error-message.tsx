// Create a new component for displaying API connection errors
import { AlertCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

interface ErrorMessageProps {
  message: string
  suggestion?: string
}

export function ErrorMessage({ message, suggestion }: ErrorMessageProps) {
  return (
    <Alert variant="destructive" className="mb-6">
      <AlertCircle className="h-4 w-4" />
      <AlertTitle>Error</AlertTitle>
      <AlertDescription>
        <p>{message}</p>
        {suggestion && <p className="mt-2">{suggestion}</p>}
      </AlertDescription>
    </Alert>
  )
}
