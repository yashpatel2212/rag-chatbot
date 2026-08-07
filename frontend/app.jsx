import React, { useEffect, useRef, useState } from "https://esm.sh/react@18.3.1";
import { createRoot } from "https://esm.sh/react-dom@18.3.1/client";
import htm from "https://esm.sh/htm@3.1.1";
import {
  Bot,
  FileText,
  LoaderCircle,
  MessageSquareText,
  Paperclip,
  Send,
  Trash2,
  Upload,
} from "https://esm.sh/lucide-react@0.468.0?deps=react@18.3.1";

const html = htm.bind(React.createElement);

function App() {
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [error, setError] = useState("");
  const fileInput = useRef(null);
  const chatEnd = useRef(null);

  const loadDocuments = async () => {
    setIsLoadingDocuments(true);
    try {
      const response = await fetch("/documents");
      if (!response.ok) throw new Error("Could not load documents.");
      const data = await response.json();
      setDocuments(data.documents);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsLoadingDocuments(false);
    }
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isAsking]);

  const uploadDocument = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setError("");
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch("/upload", { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Upload failed.");
      await loadDocuments();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      event.target.value = "";
      setIsUploading(false);
    }
  };

  const deleteDocument = async (document) => {
    if (!window.confirm(`Delete ${document.source}?`)) return;

    setError("");
    try {
      const response = await fetch(`/delete/${document.pdf_id}`, { method: "DELETE" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Delete failed.");
      setDocuments((items) => items.filter((item) => item.pdf_id !== document.pdf_id));
    } catch (requestError) {
      setError(requestError.message);
    }
  };

  const askQuestion = async (event) => {
    event.preventDefault();
    const text = question.trim();
    if (!text || isAsking) return;

    setError("");
    setMessages((items) => [...items, { role: "user", content: text }]);
    setQuestion("");
    setIsAsking(true);

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Unable to answer the question.");
      setMessages((items) => [...items, { role: "assistant", ...data }]);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsAsking(false);
    }
  };

  return html`
    <main className="app-shell">
      <aside className="library" aria-label="Document library">
        <div className="brand">
          <div className="brand-mark"><${Bot} size=${22} strokeWidth=${2.2} /></div>
          <div><p className="eyebrow">Knowledge workspace</p><h1>Document Desk</h1></div>
        </div>

        <section className="upload-area" aria-label="Upload a PDF">
          <input ref=${fileInput} type="file" accept="application/pdf,.pdf" onChange=${uploadDocument} hidden />
          <${Upload} size=${25} />
          <strong>${isUploading ? "Indexing your PDF..." : "Add a PDF"}</strong>
          <span>Upload a policy, handbook, or report</span>
          <button className="secondary-button" type="button" disabled=${isUploading} onClick=${() => fileInput.current?.click()}>
            <${Paperclip} size=${16} /> Choose file
          </button>
        </section>

        <div className="library-heading">
          <div><p className="eyebrow">Library</p><h2>Indexed documents</h2></div>
          <span className="count">${documents.length}</span>
        </div>

        <div className="document-list">
          ${isLoadingDocuments && html`<p className="muted">Loading documents...</p>`}
          ${!isLoadingDocuments && documents.length === 0 && html`<p className="muted">No documents indexed yet.</p>`}
          ${documents.map((document) => html`
            <article className="document-row" key=${document.pdf_id}>
              <div className="document-icon"><${FileText} size=${18} /></div>
              <div className="document-details">
                <strong title=${document.source}>${document.source}</strong>
                <span>${document.chunk_count} searchable chunks</span>
              </div>
              <button className="icon-button danger" type="button" title="Delete document" onClick=${() => deleteDocument(document)}>
                <${Trash2} size=${17} />
              </button>
            </article>
          `)}
        </div>
      </aside>

      <section className="chat-panel" aria-label="Chat with documents">
        <header className="chat-header">
          <div><p className="eyebrow">Grounded answers</p><h2>Ask your documents</h2></div>
          <div className="indexed-status"><${FileText} size=${16} /> ${documents.length} indexed</div>
        </header>

        <div className="conversation">
          ${messages.length === 0 && html`
            <div className="empty-chat">
              <div className="empty-icon"><${MessageSquareText} size=${30} /></div>
              <h3>What do you need to know?</h3>
              <p>Answers are generated from your indexed documents and include their source pages.</p>
              <div className="suggestions">
                <button type="button" onClick=${() => setQuestion("What are the leave policies?")}>Leave policies</button>
                <button type="button" onClick=${() => setQuestion("Summarize travel reimbursement rules.")}>Travel rules</button>
              </div>
            </div>
          `}

          ${messages.map((message, index) => html`
            <article className=${`message ${message.role}`} key=${index}>
              <div className="message-label">${message.role === "user" ? "You" : "Document Desk"}</div>
              <p>${message.content || message.answer}</p>
              ${message.sources?.length > 0 && html`
                <div className="citations">
                  ${message.sources.map((source, sourceIndex) => html`
                    <span key=${sourceIndex}>${source.source}, p. ${source.page}</span>
                  `)}
                </div>
              `}
            </article>
          `)}

          ${isAsking && html`<div className="thinking"><${LoaderCircle} size=${18} /> Searching indexed documents...</div>`}
          <div ref=${chatEnd}></div>
        </div>

        <footer className="composer-area">
          ${error && html`<p className="error-message">${error}</p>`}
          <form className="composer" onSubmit=${askQuestion}>
            <textarea value=${question} onInput=${(event) => setQuestion(event.target.value)} placeholder="Ask a question about your documents..." rows="1" aria-label="Question"></textarea>
            <button className="send-button" type="submit" title="Send question" disabled=${!question.trim() || isAsking}>
              <${Send} size=${19} />
            </button>
          </form>
          <p className="composer-note">Answers use retrieved PDF context and show source pages.</p>
        </footer>
      </section>
    </main>
  `;
}

createRoot(document.getElementById("root")).render(html`<${App} />`);
