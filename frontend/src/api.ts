import type { ChatResponse, SourcesResponse } from "./types";

export async function postChat(question: string): Promise<ChatResponse> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    throw new Error(`Chat request failed with status ${res.status}`);
  }
  return (await res.json()) as ChatResponse;
}

export async function getSources(): Promise<SourcesResponse> {
  const res = await fetch("/api/sources");
  if (!res.ok) {
    throw new Error(`Sources request failed with status ${res.status}`);
  }
  return (await res.json()) as SourcesResponse;
}
