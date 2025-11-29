import React, { useState } from "react";
import {
  ingestText,
  ingestUrl,
  ingestUrlDistilled,
  distillText,
  ingestPdf,
} from "../api";

export default function IngestPage() {
  // Raw text
  const [title, setTitle] = useState("");
  const [source, setSource] = useState("");
  const [lang, setLang] = useState("en");
  const [description, setDescription] = useState("");
  const [text, setText] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // URL ingest
  const [url, setUrl] = useState("");
  const [urlStatus, setUrlStatus] = useState<string | null>(null);
  const [urlLoading, setUrlLoading] = useState(false);

  // URL → distilled
  const [distUrl, setDistUrl] = useState("");
  const [distHints, setDistHints] = useState("");
  const [distUrlStatus, setDistUrlStatus] = useState<string | null>(null);
  const [distUrlLoading, setDistUrlLoading] = useState(false);

  // Text → distilled
  const [factText, setFactText] = useState("");
  const [factHints, setFactHints] = useState("");
  const [factStatus, setFactStatus] = useState<string | null>(null);
  const [factLoading, setFactLoading] = useState(false);

  // PDF ingest
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [pdfTitle, setPdfTitle] = useState("");
  const [pdfSource, setPdfSource] = useState("");
  const [pdfDesc, setPdfDesc] = useState("");
  const [pdfStatus, setPdfStatus] = useState<string | null>(null);
  const [pdfLoading, setPdfLoading] = useState(false);

  const sectionStyle: React.CSSProperties = {
    border: "1px solid #e5e7eb",
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
  };

  const inputStyle: React.CSSProperties = {
    width: "100%",
    marginBottom: 6,
    padding: 6,
    borderRadius: 6,
    border: "1px solid #e5e7eb",
    fontSize: 14,
  };

  const labelStyle: React.CSSProperties = {
    marginBottom: 4,
    fontSize: 13,
    color: "#4b5563",
  };

  // ---- Raw text ingest ----
  const onIngestText = async () => {
    if (!title || !source || !text) {
      setStatus("Title, source and text are required.");
      return;
    }
    setLoading(true);
    setStatus("Uploading raw text…");
    try {
      const res: any = await ingestText({
        title,
        source,
        lang,
        description,
        text,
      });
      setStatus(`✅ Ingested. doc_id=${res?.doc_id ?? "?"}`);
    } catch (err: any) {
      console.error(err);
      setStatus(`❌ Error: ${err.message || "failed"}`);
    } finally {
      setLoading(false);
    }
  };

  // ---- URL ingest (raw) ----
  const onIngestUrl = async () => {
    if (!url || !source) {
      setUrlStatus("URL and source are required.");
      return;
    }
    setUrlLoading(true);
    setUrlStatus("Fetching and ingesting URL…");
    try {
      const res: any = await ingestUrl({
        url,
        source,
        lang,
        description,
      });
      setUrlStatus(`✅ URL ingested. doc_id=${res?.doc_id ?? "?"}`);
    } catch (err: any) {
      console.error(err);
      setUrlStatus(`❌ Error: ${err.message || "failed"}`);
    } finally {
      setUrlLoading(false);
    }
  };

  // ---- URL → distilled facts ----
  const onIngestUrlDistilled = async () => {
    if (!distUrl || !source) {
      setDistUrlStatus("URL and source are required.");
      return;
    }
    setDistUrlLoading(true);
    setDistUrlStatus("Fetching URL and distilling facts…");
    try {
      const hints =
        distHints
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean) || [];
      const res: any = await ingestUrlDistilled({
        url: distUrl,
        source,
        lang,
        description,
        entity_hints: hints.length ? hints : undefined,
      });
      setDistUrlStatus(
        `✅ URL distilled. doc_id=${res?.doc_id ?? "?"}, facts=${res?.facts_count ?? "?"}`
      );
    } catch (err: any) {
      console.error(err);
      setDistUrlStatus(`❌ Error: ${err.message || "failed"}`);
    } finally {
      setDistUrlLoading(false);
    }
  };

  // ---- Text → distilled facts ----
  const onDistillText = async () => {
    if (!factText || !source) {
      setFactStatus("Text and source are required.");
      return;
    }
    setFactLoading(true);
    setFactStatus("Distilling facts from pasted text…");
    try {
      const hints =
        factHints
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean) || [];
      const res: any = await distillText({
        text: factText,
        source,
        lang,
        description,
        entity_hints: hints.length ? hints : undefined,
      });
      setFactStatus(
        `✅ Distilled. doc_id=${res?.doc_id ?? "?"}, facts=${res?.facts_count ?? "?"}`
      );
    } catch (err: any) {
      console.error(err);
      setFactStatus(`❌ Error: ${err.message || "failed"}`);
    } finally {
      setFactLoading(false);
    }
  };

  // ---- PDF ingest ----
  const onIngestPdf = async () => {
    if (!pdfFile || !pdfTitle || !pdfSource) {
      setPdfStatus("PDF file, title and source are required.");
      return;
    }
    if (pdfFile.size > 5 * 1024 * 1024) {
      setPdfStatus("PDF must be less than 5MB.");
      return;
    }
    setPdfLoading(true);
    setPdfStatus("Uploading and processing PDF…");
    try {
      const res: any = await ingestPdf(pdfFile, {
        title: pdfTitle,
        source: pdfSource,
        lang,
        description: pdfDesc,
      });
      setPdfStatus(`✅ PDF ingested. doc_id=${res?.doc_id ?? "?"}`);
    } catch (err: any) {
      console.error(err);
      setPdfStatus(`❌ Error: ${err.message || "failed"}`);
    } finally {
      setPdfLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 900 }}>
      <h2>Knowledge Ingestion</h2>

      {/* Shared top hint */}
      <p style={{ fontSize: 13, color: "#6b7280", marginTop: 0 }}>
        Use <b>source</b> like <i>al-barakah</i>, <i>malisha-edu</i>,{" "}
        <i>easylink</i>, <i>brcc</i> so the agent can route questions correctly.
      </p>

      {/* RAW TEXT */}
      <section style={sectionStyle}>
        <h3 style={{ marginTop: 0 }}>1. Raw Text</h3>
        <div style={labelStyle}>Title</div>
        <input
          style={inputStyle}
          placeholder="Title"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
        <div style={labelStyle}>Source</div>
        <input
          style={inputStyle}
          placeholder="al-barakah / malisha-edu / easylink / brcc…"
          value={source}
          onChange={(e) => setSource(e.target.value)}
        />
        <div style={labelStyle}>Language</div>
        <input
          style={inputStyle}
          placeholder="en / bn / cn…"
          value={lang}
          onChange={(e) => setLang(e.target.value)}
        />
        <div style={labelStyle}>Description</div>
        <textarea
          style={inputStyle}
          placeholder="Short description"
          rows={2}
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
        <div style={labelStyle}>Raw Text</div>
        <textarea
          style={inputStyle}
          placeholder="Paste raw text to store in KB"
          rows={6}
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button onClick={onIngestText} disabled={loading}>
          {loading ? "Ingesting…" : "Ingest Text"}
        </button>
        {status && (
          <div style={{ marginTop: 8, fontSize: 14, color: "#4b5563" }}>
            {status}
          </div>
        )}
      </section>

      {/* URL INGEST */}
      <section style={sectionStyle}>
        <h3 style={{ marginTop: 0 }}>2. URL → Raw Text</h3>
        <div style={labelStyle}>URL</div>
        <input
          style={inputStyle}
          placeholder="https://malishaedu.com/..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
        <button onClick={onIngestUrl} disabled={urlLoading}>
          {urlLoading ? "Ingesting URL…" : "Ingest URL"}
        </button>
        {urlStatus && (
          <div style={{ marginTop: 8, fontSize: 14, color: "#4b5563" }}>
            {urlStatus}
          </div>
        )}
      </section>

      {/* URL → DISTILLED FACTS */}
      <section style={sectionStyle}>
        <h3 style={{ marginTop: 0 }}>3. URL → Distilled Facts</h3>
        <div style={labelStyle}>URL</div>
        <input
          style={inputStyle}
          placeholder="https://malishaedu.com/..."
          value={distUrl}
          onChange={(e) => setDistUrl(e.target.value)}
        />
        <div style={labelStyle}>Entity hints (comma separated)</div>
        <input
          style={inputStyle}
          placeholder="malishaedu, dr maruf, korban ali..."
          value={distHints}
          onChange={(e) => setDistHints(e.target.value)}
        />
        <button onClick={onIngestUrlDistilled} disabled={distUrlLoading}>
          {distUrlLoading ? "Distilling from URL…" : "URL → Distilled Facts"}
        </button>
        {distUrlStatus && (
          <div style={{ marginTop: 8, fontSize: 14, color: "#4b5563" }}>
            {distUrlStatus}
          </div>
        )}
      </section>

      {/* TEXT → DISTILLED FACTS */}
      <section style={sectionStyle}>
        <h3 style={{ marginTop: 0 }}>4. Pasted Text → Distilled Facts</h3>
        <div style={labelStyle}>Text</div>
        <textarea
          style={inputStyle}
          placeholder="Paste transcript, speech, etc. The system will extract clean facts."
          rows={6}
          value={factText}
          onChange={(e) => setFactText(e.target.value)}
        />
        <div style={labelStyle}>Entity hints (comma separated)</div>
        <input
          style={inputStyle}
          placeholder="malishaedu, dr maruf, korban ali..."
          value={factHints}
          onChange={(e) => setFactHints(e.target.value)}
        />
        <button onClick={onDistillText} disabled={factLoading}>
          {factLoading ? "Distilling…" : "Distill Text into Facts"}
        </button>
        {factStatus && (
          <div style={{ marginTop: 8, fontSize: 14, color: "#4b5563" }}>
            {factStatus}
          </div>
        )}
      </section>

      {/* PDF INGEST */}
      <section style={sectionStyle}>
        <h3 style={{ marginTop: 0 }}>5. PDF (&lt; 5MB)</h3>
        <div style={labelStyle}>PDF File</div>
        <input
          style={{ marginBottom: 6 }}
          type="file"
          accept="application/pdf"
          onChange={(e) => setPdfFile(e.target.files?.[0] || null)}
        />
        <div style={labelStyle}>Title</div>
        <input
          style={inputStyle}
          placeholder="Title for this PDF"
          value={pdfTitle}
          onChange={(e) => setPdfTitle(e.target.value)}
        />
        <div style={labelStyle}>Source</div>
        <input
          style={inputStyle}
          placeholder="al-barakah / malisha-edu / easylink / brcc…"
          value={pdfSource}
          onChange={(e) => setPdfSource(e.target.value)}
        />
        <div style={labelStyle}>Description</div>
        <textarea
          style={inputStyle}
          placeholder="Short description (optional)"
          rows={2}
          value={pdfDesc}
          onChange={(e) => setPdfDesc(e.target.value)}
        />
        <button onClick={onIngestPdf} disabled={pdfLoading}>
          {pdfLoading ? "Ingesting PDF…" : "Ingest PDF"}
        </button>
        {pdfStatus && (
          <div style={{ marginTop: 8, fontSize: 14, color: "#4b5563" }}>
            {pdfStatus}
          </div>
        )}
      </section>
    </div>
  );
}
