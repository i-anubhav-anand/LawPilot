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

const API_BASE_URL = "http://localhost:8000"

// Utility function to handle fetch errors
async function fetchWithErrorHandling(url: string, options: RequestInit = {}) {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 30000) // 30 second timeout (increased from 10 seconds)
  
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
  return fetchWithErrorHandling(`${API_BASE_URL}/api/chat/session/${sessionId}/`)
}

// Get all chat sessions
export async function getChatSessions(): Promise<ChatSession[]> {
  const response = await fetchWithErrorHandling(`${API_BASE_URL}/api/chat/sessions/`)
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
  return fetchWithErrorHandling(`${API_BASE_URL}/api/documents/${documentId}`)
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
  
  return new Promise<DocumentResponse>((resolve, reject) => {
    const checkStatus = async () => {
      if (attempts >= maxAttempts) {
        reject(new Error("Document processing timed out"));
        return;
      }
      
      try {
        const status = await getDocumentStatus(documentId);
        onStatusUpdate(status);
        
        if (status.status === "processed") {
          resolve(status);
          return;
        } else if (status.status === "failed") {
          reject(new Error(`Document processing failed: ${status.error || "Unknown error"}`));
          return;
        }
        
        // Continue polling
        attempts++;
        setTimeout(checkStatus, interval);
      } catch (error) {
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
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout for uploads (increased from 30 seconds)
  
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/with-file`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

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
      
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof TypeError && error.message.includes("Failed to fetch")) {
      throw new Error("Unable to connect to the API server. Please check if the server is running.");
    }

    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Upload timed out. The file may be too large or the server is busy.");
    }

    throw error;
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
