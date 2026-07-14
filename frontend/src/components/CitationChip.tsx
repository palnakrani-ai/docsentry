import { useState } from "react";
import type { Citation } from "../types";

interface Props {
  citation: Citation;
  index: number;
}

export default function CitationChip({ citation, index }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className={`citation ${open ? "citation-open" : ""}`}>
      <button
        type="button"
        className="citation-chip"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="citation-index">{String(index + 1).padStart(2, "0")}</span>
        <span className="citation-source">{citation.source}</span>
        <span className="citation-divider" aria-hidden="true" />
        <span className="citation-section">{citation.section}</span>
        <span className="citation-caret" aria-hidden="true">
          {open ? "−" : "+"}
        </span>
      </button>
      {open && (
        <blockquote className="citation-snippet">
          {citation.snippet}
        </blockquote>
      )}
    </div>
  );
}
