import { AlertTriangle } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

export function LegalDisclaimer() {
  return (
    <Alert className="bg-gray-50 border border-gray-200 mb-6">
      <AlertTriangle className="h-4 w-4 text-gray-500" />
      <AlertDescription className="text-gray-600 text-sm font-normal">
        This AI assistant provides general legal information, not legal advice. For specific legal issues, please
        consult with a qualified attorney.
      </AlertDescription>
    </Alert>
  )
}
