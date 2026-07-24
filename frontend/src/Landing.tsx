const HOW = [
  {
    step: "01",
    title: "Grounded retrieval",
    body: "Answers come only from the indexed documents. DocSentry retrieves the relevant passages first, then writes from what it found, so nothing is invented on the way.",
  },
  {
    step: "02",
    title: "Cited every time",
    body: "Each reply shows the exact source passage it leaned on. You can open the citation and read the original wording, so every claim traces back to a document.",
  },
  {
    step: "03",
    title: "Refuses & flags",
    body: "When the docs are silent it says \"not in the documents\" instead of guessing. And it catches prompt-injection attempts, flagging them rather than following them.",
  },
];

export default function Landing({ onStart }: { onStart: () => void }) {
  return (
    <div className="shell landing">
      <div className="backdrop" aria-hidden="true" />

      <header className="lp-topbar">
        <div className="lp-topbar-inner">
          <span className="wordmark">
            Doc<em>Sentry</em>
          </span>
          <span className="tagline">
            Answers from your documents. Nothing made up.
          </span>
        </div>
      </header>

      <main className="lp-main">
        <section className="lp-hero">
          <span className="lp-eyebrow">GROUNDED · CITED · INJECTION-AWARE</span>
          <h1 className="lp-headline">
            Answers straight from your documents.
            <br />
            <em>Nothing invented.</em>
          </h1>
          <p className="lp-sub">
            DocSentry grounds every answer in your indexed documents and cites
            the exact source passage it used. When the documents don't cover a
            question, it refuses instead of guessing, and it flags prompt-injection
            attempts rather than following them.
          </p>
          <div className="lp-cta-row">
            <button type="button" className="send-button lp-cta" onClick={onStart}>
              Open DocSentry
            </button>
            <span className="lp-cta-note">
              Live demo over the Meridian Outfitters handbook
            </span>
          </div>
        </section>

        <section className="lp-how" aria-label="How it works">
          <div className="lp-how-grid">
            {HOW.map((c) => (
              <article key={c.step} className="lp-card">
                <span className="lp-card-step">{c.step}</span>
                <h2 className="lp-card-title">{c.title}</h2>
                <p className="lp-card-body">{c.body}</p>
              </article>
            ))}
          </div>
        </section>
      </main>

      <footer className="lp-footer">
        <p className="composer-note">
          Grounded answers only · citations on every reply · injection attempts
          get flagged
        </p>
      </footer>
    </div>
  );
}
