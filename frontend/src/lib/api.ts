// API service functions for the Legal AI Assistant

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
  timestamp: string
}

export interface Source {
  source_type: string
  title: string
  content: string
  citation?: string | null
  relevance_score?: number | null
  document_id?: string | null
}

export interface ChatResponse {
  message: string
  session_id: string
  case_file_id?: string | null
  sources?: Source[]
  next_questions?: string[]
  uploaded_document_id?: string
}

export interface ChatSession {
  session_id: string
  message_count: number
  created_at?: string
}

export interface DocumentResponse {
  document_id: string
  filename: string
  session_id?: string | null
  case_file_id?: string | null
  status: string // "processing", "processed", "failed"
  created_at: string
  processed_at?: string | null
  error?: string | null
  is_global: boolean
}

export interface DocumentAnalysisResponse {
  document_id: string;
  summary: string;
  key_points: string[];
  issues: Array<{
    type: 'warning' | 'error' | 'info';
    title: string;
    description: string;
  }>;
  recommendations: string[];
  relevant_laws: Array<{
    title: string;
    description: string;
    citation?: string;
  }>;
}

// Export this constant so it can be used in other files
export const API_BASE_URL = "http://localhost:8000"

// Utility function to handle fetch errors
async function fetchWithErrorHandling(url: string, options: RequestInit = {}, timeoutMs: number = 30000) {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs) // Configurable timeout with default 30 seconds
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        ...options.headers,
      },
    })

    clearTimeout(timeoutId)

    if (!response.ok) {
      const errorText = await response.text()
      let errorMessage = `API error: ${response.status}`
      
      try {
        const errorJson = JSON.parse(errorText)
        if (errorJson.detail) {
          errorMessage = errorJson.detail
        }
      } catch (e) {
        // If it's not valid JSON, use the raw error text
        if (errorText) {
          errorMessage = errorText
        }
      }
      
      throw new Error(errorMessage)
    }

    return await response.json()
  } catch (error) {
    if (error instanceof TypeError && error.message.includes("Failed to fetch")) {
      throw new Error("Unable to connect to the API server. Please check if the server is running at " + API_BASE_URL)
    }

    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Request timed out. The API server is taking too long to respond.")
    }

    throw error
  } finally {
    clearTimeout(timeoutId)
  }
}

// Send a chat message to the API
export async function sendChatMessage(message: string, sessionId?: string, caseFileId?: string): Promise<ChatResponse> {
  return fetchWithErrorHandling(`${API_BASE_URL}/api/chat/`, {
    method: "POST",
    body: JSON.stringify({
      query: message, // Important: API expects 'query', not 'message'
      session_id: sessionId,
      case_file_id: caseFileId,
    }),
  })
}

// Create a new chat session
export async function createChatSession(caseFileId?: string): Promise<ChatSession> {
  const url = new URL(`${API_BASE_URL}/api/chat/create-session`)
  if (caseFileId) {
    url.searchParams.append("case_file_id", caseFileId)
  }
  
  return fetchWithErrorHandling(url.toString(), {
    method: "POST",
  })
}

// Get chat history for a session
export async function getChatHistory(sessionId: string): Promise<ChatMessage[]> {
  return fetchWithErrorHandling(`${API_BASE_URL}/api/chat/${sessionId}/history/`)
}

// Get a specific session
export async function getSession(sessionId: string): Promise<ChatSession> {
  return fetchWithErrorHandling(`${API_BASE_URL}/api/chat/sessions/${sessionId}`)
}

// Get all chat sessions
export async function getChatSessions(): Promise<ChatSession[]> {
  const response = await fetchWithErrorHandling(`${API_BASE_URL}/api/chat/sessions`)
  return response.sessions || []
}

// Upload a document
export async function uploadDocument(
  file: File, 
  sessionId?: string, 
  caseFileId?: string, 
  isGlobal: boolean = false
): Promise<DocumentResponse> {
  const formData = new FormData()
  formData.append("file", file)
  
  if (sessionId) {
    formData.append("session_id", sessionId)
  }
  
  if (caseFileId) {
    formData.append("case_file_id", caseFileId)
  }
  
  const endpoint = isGlobal ? 
    `${API_BASE_URL}/api/documents/upload/global` : 
    `${API_BASE_URL}/api/documents/upload`
  
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 60000) // 60 second timeout for uploads (increased from 30 seconds)
  
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    })

    clearTimeout(timeoutId)

    if (!response.ok) {
      const errorText = await response.text()
      let errorMessage = `Upload error: ${response.status}`
      
      try {
        const errorJson = JSON.parse(errorText)
        if (errorJson.detail) {
          errorMessage = errorJson.detail
        }
      } catch (e) {
        if (errorText) {
          errorMessage = errorText
        }
      }
      
      throw new Error(errorMessage)
    }

    return await response.json()
  } catch (error) {
    if (error instanceof TypeError && error.message.includes("Failed to fetch")) {
      throw new Error("Unable to connect to the API server. Please check if the server is running.")
    }

    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Upload timed out. The file may be too large or the server is busy.")
    }

    throw error
  } finally {
    clearTimeout(timeoutId)
  }
}

// Get document status
export async function getDocumentStatus(documentId: string): Promise<DocumentResponse> {
  // Make sure we're not trying to get status for "list" or other special paths
  if (documentId === 'list' || documentId === 'global' || documentId === 'processing-status') {
    throw new Error(`Invalid document ID: ${documentId}`);
  }
  return fetchWithErrorHandling(`${API_BASE_URL}/api/documents/${documentId}`)
}

// Get the download URL for a document
export function getDocumentDownloadUrl(documentId: string): string {
  return `${API_BASE_URL}/api/documents/${documentId}/download`;
}

// List all documents
export async function listDocuments(sessionId?: string): Promise<DocumentResponse[]> {
  const url = new URL(`${API_BASE_URL}/api/documents/`)
  if (sessionId) {
    url.searchParams.append("session_id", sessionId)
  }
  
  return fetchWithErrorHandling(url.toString())
}

// List global documents
export async function listGlobalDocuments(): Promise<DocumentResponse[]> {
  return fetchWithErrorHandling(`${API_BASE_URL}/api/documents/global`)
}

// Toggle document global status
export async function toggleDocumentGlobalStatus(documentId: string): Promise<DocumentResponse> {
  return fetchWithErrorHandling(`${API_BASE_URL}/api/documents/${documentId}/toggle-global`, {
    method: "POST",
  })
}

// Check API health
export async function checkApiHealth(): Promise<{ status: string, timestamp: string }> {
  return fetchWithErrorHandling(`${API_BASE_URL}/health`)
}

// Get API server status
export async function getServerStatus(): Promise<any> {
  return fetchWithErrorHandling(`${API_BASE_URL}/status`)
}

// Check document processing status with polling
export async function pollDocumentStatus(
  documentId: string, 
  onStatusUpdate: (status: DocumentResponse) => void,
  interval: number = 2000,
  maxAttempts: number = 60  // Increased from 30 to 60, allowing for 120 seconds of polling
): Promise<DocumentResponse> {
  let attempts = 0;
  let notFoundAttempts = 0; // Track "not found" errors separately
  const MAX_NOT_FOUND_RETRIES = 15; // Increased from 10 to 15 to allow more time for document processing
  
  return new Promise<DocumentResponse>((resolve, reject) => {
    const checkStatus = async () => {
      if (attempts >= maxAttempts) {
        reject(new Error("Document processing timed out"));
        return;
      }
      
      try {
        console.log(`Checking status for document ${documentId} (attempt ${attempts+1}/${maxAttempts})`);
        const status = await getDocumentStatus(documentId);
        // Reset not found counter on successful response
        notFoundAttempts = 0;
        onStatusUpdate(status);
        
        if (status.status === "processed") {
          console.log(`Document ${documentId} processing completed successfully`);
          resolve(status);
          return;
        } else if (status.status === "failed") {
          console.error(`Document ${documentId} processing failed: ${status.error || "Unknown error"}`);
          reject(new Error(`Document processing failed: ${status.error || "Unknown error"}`));
          return;
        }
        
        // Continue polling
        attempts++;
        setTimeout(checkStatus, interval);
      } catch (error) {
        // Special handling for "Document not found" errors
        if (error instanceof Error && error.message.includes("Document not found")) {
          notFoundAttempts++;
          console.warn(`Document ${documentId} not found (attempt ${notFoundAttempts}/${MAX_NOT_FOUND_RETRIES}). This may be normal during processing.`);
          
          // Allow more retries for "not found" with progressively longer delays
          if (notFoundAttempts < MAX_NOT_FOUND_RETRIES) {
            // Use a progressively longer delay for not found errors
            // Starting at 3 seconds and increasing with each attempt
            const notFoundDelay = interval * 1.5 * notFoundAttempts;
            console.log(`Retrying in ${notFoundDelay/1000} seconds...`);
            attempts++;
            setTimeout(checkStatus, notFoundDelay);
            return;
          } else {
            console.error(`Document ${documentId} not found after ${MAX_NOT_FOUND_RETRIES} retries. The document may have failed to upload properly.`);
          }
        } else {
          console.error(`Error checking document status: ${error instanceof Error ? error.message : "Unknown error"}`);
        }
        
        // For other errors or if we've exceeded not found retries
        reject(error);
      }
    };
    
    // Start polling
    checkStatus();
  });
}

// Get server processing statistics
export async function getProcessingStats(): Promise<any> {
  return fetchWithErrorHandling(`${API_BASE_URL}/api/documents/processing-status`);
}

// Analyze a document to get legal issues, recommendations, and related information
export async function analyzeDocument(documentId: string): Promise<DocumentAnalysisResponse> {
  return fetchWithErrorHandling(`${API_BASE_URL}/api/documents/${documentId}/analyze`, {
    method: 'POST',
  });
}

export async function sendChatMessageWithDocumentText(
  message: string, 
  documentText: string, 
  documentName: string,
  sessionId: string, 
  caseFileId?: string
): Promise<ChatResponse> {
  const url = `${API_BASE_URL}/api/chat/with-document-text`;
  
  const requestData = {
    query: message,
    session_id: sessionId,
    case_file_id: caseFileId,
    document_text: documentText,
    document_name: documentName
  };
  
  const response = await fetchWithErrorHandling(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestData),
  });
  
  return response as ChatResponse;
}

// Send a chat message with a file attachment
export async function sendChatMessageWithFile(
  message: string, 
  file: File,
  sessionId: string, 
  caseFileId?: string
): Promise<ChatResponse> {
  const formData = new FormData();
  formData.append("query", message);
  formData.append("session_id", sessionId);
  
  if (caseFileId) {
    formData.append("case_file_id", caseFileId);
  }
  
  if (file) {
    formData.append("file", file);
  }
  
  // Increased timeout for large files
  const timeout = Math.max(60000, file.size / 10000); // 60s minimum, or longer for larger files
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  // Add retry logic for transient errors
  const MAX_RETRIES = 2;
  let retryCount = 0;
  
  const attemptUpload = async (): Promise<ChatResponse> => {
    try {
      console.log(`Sending chat message with file to ${API_BASE_URL}/api/chat/with-file`);
      console.log(`File: ${file.name} (${file.size} bytes, ${file.type})`);
      console.log(`Message: "${message.length > 50 ? message.substring(0, 50) + '...' : message}"`);
      console.log(`Session ID: ${sessionId}`);
      if (caseFileId) console.log(`Case File ID: ${caseFileId}`);
      
      const response = await fetch(`${API_BASE_URL}/api/chat/with-file`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage = `Upload error: ${response.status}`;
        
        try {
          const errorJson = JSON.parse(errorText);
          if (errorJson.detail) {
            errorMessage = errorJson.detail;
          }
        } catch (e) {
          if (errorText) {
            errorMessage = errorText;
          }
        }
        
        console.error(`Error response from server: ${errorMessage}`);
        
        // Check for retriable errors (5xx server errors or specific error messages)
        const isRetriable = 
          (response.status >= 500 && response.status < 600) || 
          errorMessage.includes("timeout") ||
          errorMessage.includes("overloaded");
        
        if (isRetriable && retryCount < MAX_RETRIES) {
          retryCount++;
          console.warn(`Upload failed (${errorMessage}). Retrying (${retryCount}/${MAX_RETRIES})...`);
          
          // Add exponential backoff
          const backoffMs = 1000 * Math.pow(2, retryCount);
          await new Promise(resolve => setTimeout(resolve, backoffMs));
          
          return attemptUpload();
        }
        
        throw new Error(errorMessage);
      }

      const responseData = await response.json();
      console.log(`Successfully received response from server for file upload`);
      return responseData;
    } catch (error: any) {
      if (error instanceof TypeError && error.message.includes("Failed to fetch")) {
        if (retryCount < MAX_RETRIES) {
          retryCount++;
          console.warn(`Connection error. Retrying (${retryCount}/${MAX_RETRIES})...`);
          
          // Add exponential backoff
          const backoffMs = 1000 * Math.pow(2, retryCount);
          await new Promise(resolve => setTimeout(resolve, backoffMs));
          
          return attemptUpload();
        }
        throw new Error("Unable to connect to the API server. Please check if the server is running.");
      }

      if (error instanceof DOMException && error.name === "AbortError") {
        throw new Error("Upload timed out. The file may be too large or the server is busy.");
      }

      throw error;
    }
  };
  
  try {
    return await attemptUpload();
  } finally {
    clearTimeout(timeoutId);
  }
}

// Send a chat message with an image for vision analysis
export async function sendChatMessageWithImage(
  message: string, 
  imageFile: File,
  sessionId: string, 
  caseFileId?: string
): Promise<ChatResponse> {
  // Validate the file is an image
  if (!imageFile.type.startsWith('image/')) {
    throw new Error('File must be an image (jpg, png, gif, etc.)');
  }
  
  const formData = new FormData();
  formData.append("query", message);
  formData.append("session_id", sessionId);
  
  if (caseFileId) {
    formData.append("case_file_id", caseFileId);
  }
  
  formData.append("file", imageFile);
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120000); // 120 second timeout for vision processing (increased from 60 seconds)
  
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/with-image`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorText = await response.text();
      let errorMessage = `Image analysis error: ${response.status}`;
      
      try {
        const errorJson = JSON.parse(errorText);
        if (errorJson.detail) {
          errorMessage = errorJson.detail;
        }
      } catch (e) {
        if (errorText) {
          errorMessage = errorText;
        }
      }
      
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof TypeError && error.message.includes("Failed to fetch")) {
      throw new Error("Unable to connect to the API server. Please check if the server is running.");
    }

    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Image analysis timed out. The image may be too complex or the server is busy.");
    }

    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

// Update case summary from chat history
export async function updateCaseSummaryFromChat(
  caseFileId: string,
  chatHistory: any[],
  forceUpdate: boolean = false
): Promise<{ summary: string }> {
  const url = `${API_BASE_URL}/api/summaries/generate/${caseFileId}`;
  
  return fetchWithErrorHandling(url, {
    method: 'POST',
      headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      chat_history: chatHistory,
      force_update: forceUpdate
    }),
  });
}

// Get formatted case summary
export async function getFormattedCaseSummary(caseFileId: string): Promise<{ summary: string }> {
  return fetchWithErrorHandling(`${API_BASE_URL}/api/summaries/formatted/${caseFileId}`);
}

// Add new function to get summary from session directly with extended timeout
export async function getSummaryFromSession(
  sessionId: string
): Promise<{ summary: string }> {
  // Use a longer timeout (5 minutes) for summary generation since it can be computationally intensive
  return fetchWithErrorHandling(
    `${API_BASE_URL}/api/summaries/from-session/${sessionId}`, 
    {}, 
    300000 // 5 minute timeout (300,000ms)
  );
}
