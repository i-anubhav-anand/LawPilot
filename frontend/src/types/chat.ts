import { Source } from "@/lib/api";

export interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
  sources?: Source[];
  nextQuestions?: string[];
  isError?: boolean;
} 