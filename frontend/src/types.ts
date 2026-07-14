export interface Citation {
  source: string;
  section: string;
  snippet: string;
}

export interface ChatResponse {
  answer: string;
  citations: Citation[];
  refused: boolean;
  flags: string[];
  latencyMs: number;
}

export interface SourceInfo {
  source: string;
  title: string;
  sections: number;
}

export interface SourcesResponse {
  sources: SourceInfo[];
}

export type MessageRole = "user" | "assistant";

export interface ThreadMessage {
  id: string;
  role: MessageRole;
  text: string;
  citations?: Citation[];
  refused?: boolean;
  flags?: string[];
  latencyMs?: number;
  error?: boolean;
}
