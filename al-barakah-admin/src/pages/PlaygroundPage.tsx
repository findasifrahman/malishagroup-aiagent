import React, { useState } from "react";
import { adminChat } from "../api";

interface Msg {
  role: "user" | "assistant";
  text: string;
}

export default function PlaygroundPage() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [debug, setDebug] = useState<any>(null);

  const send = async () => {
    const text = input.trim();
    if (!text) return;
    setMessages((m) => [...m, { role: "user", text }]);
    setInput("");
    setLoading(true);
    setDebug(null);
    try {
      const res = await adminChat(text);
      setMessages((m) => [...m, { role: "assistant", text: res.answer }]);
      setDebug(res);
    } catch (err: any) {
      setMessages((m) => [
        ...m,
        { role: "assistant", text: "Error: " + (err.message || "failed") },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", gap: 16, height: "100%" }}>
      <div style={{ flex: 2, display: "flex", flexDirection: "column" }}>
        <h2>Admin Playground</h2>
        <div
          style={{
            flex: 1,
            border: "1px solid #e5e7eb",
            borderRadius: 8,
            padding: 8,
            overflowY: "auto",
            background: "#f9fafb",
          }}
        >
          {messages.map((m, i) => (
            <div key={i} style={{ marginBottom: 8 }}>
              <strong>{m.role === "user" ? "You" : "Agent"}:</strong>{" "}
              <span>{m.text}</span>
            </div>
          ))}
          {loading && <div>Assistant is thinking…</div>}
        </div>
        <div style={{ marginTop: 8 }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            rows={2}
            style={{ width: "100%", padding: 6, marginBottom: 6 }}
            placeholder="Ask a question to test the agent"
          />
          <button onClick={send} disabled={loading}>
            {loading ? "Sending…" : "Send"}
          </button>
        </div>
      </div>
      <div style={{ flex: 1 }}>
        <h3>Debug Info</h3>
        <pre
          style={{
            border: "1px solid #e5e7eb",
            borderRadius: 8,
            padding: 8,
            maxHeight: "80vh",
            overflow: "auto",
            background: "#f9fafb",
          }}
        >
          {debug ? JSON.stringify(debug, null, 2) : "No debug yet."}
        </pre>
      </div>
    </div>
  );
}
