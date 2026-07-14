import type { ThreadMessage } from "../types";
import CitationChip from "./CitationChip";

interface Props {
  message: ThreadMessage;
}

function flagLabel(flag: string): string {
  return flag.replace(/_/g, " ");
}

export default function MessageBubble({ message }: Props) {
  if (message.role === "user") {
    return (
      <div className="row row-user">
        <div className="bubble bubble-user">
          <p>{message.text}</p>
        </div>
      </div>
    );
  }

  const refused = message.refused === true;
  const flags = message.flags ?? [];
  const citations = message.citations ?? [];

  return (
    <div className="row row-assistant">
      <div
        className={`bubble bubble-assistant ${refused ? "bubble-refused" : ""} ${
          message.error ? "bubble-error" : ""
        }`}
      >
        {refused && (
          <div className="refusal-banner">
            <span className="refusal-dot" aria-hidden="true" />
            Not in the docs
          </div>
        )}
        {flags.length > 0 && (
          <div className="flag-row">
            {flags.map((flag) => (
              <span key={flag} className="flag-tag">
                {flagLabel(flag)}
              </span>
            ))}
          </div>
        )}
        <p className="answer-text">{message.text}</p>
        {citations.length > 0 && (
          <div className="citation-list">
            <span className="citation-list-label">Cited passages</span>
            {citations.map((c, i) => (
              <CitationChip key={`${c.source}-${c.section}-${i}`} citation={c} index={i} />
            ))}
          </div>
        )}
        {typeof message.latencyMs === "number" && (
          <div className="message-meta">
            <span>answered in {(message.latencyMs / 1000).toFixed(2)}s</span>
          </div>
        )}
      </div>
    </div>
  );
}
