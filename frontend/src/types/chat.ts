import { Source } from "@/lib/api";

export interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
  sources?: Source[];
  isError?: boolean;
  fileAttachment?: {
    name: string;
    type: string;
    isProcessing?: boolean;
    documentId?: string;
    error?: string;
  };
} 