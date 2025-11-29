// src/pages/ChatPage.tsx
import React, { useState } from "react";
import { apiChat } from "../api";
import type { AuthUser } from "../api";
import { Link } from "react-router-dom";

type AgentOption = "auto" | "al-barakah" | "malisha-edu" | "easylink" | "brcc";

interface Props {
  user: AuthUser | null;
  checked: boolean;
}

interface ChatMsg {
  role: "user" | "assistant";
  text: string;
  usedWeb?: boolean;
}

export default function ChatPage({ user }: Props) {
  const [messages, setMessages] = useState<ChatMsg[]>([
    {
      role: "assistant",
      text:
        "Assalamu alaikum! I am the Al-Barakah & Malisha assistant.\n\n" +
        "Ask about food, delivery, China travel, MalishaEdu, Easylink, BRCC…",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const send = async () => {
    const text = input.trim();
    if (!text) return;
    setMessages((m) => [...m, { role: "user", text }]);
    setInput("");
    setLoading(true);
    setError(null);
    try {
      const res = await apiChat(text, conversationId || undefined);
      if (!conversationId && res.conversation_id) {
        setConversationId(res.conversation_id);
      }
      setMessages((m) => [
        ...m,
        { role: "assistant", text: res.answer, usedWeb: !!res.used_web },
      ]);
    } catch (err: any) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text:
            "Sorry, I couldn’t reach the server. " +
            (err.message || "Unknown error"),
        },
      ]);
      setError(err.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!loading) send();
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f3f4f6",
        display: "flex",
        alignItems: "stretch",
        justifyContent: "center",
        padding: 16,
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: 900,
          background: "#ffffff",
          borderRadius: 16,
          border: "1px solid #e5e7eb",
          boxShadow: "0 12px 30px rgba(15,23,42,0.12)",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        <header
          style={{
            padding: "10px 16px",
            borderBottom: "1px solid #e5e7eb",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            background: "#f9fafb",
          }}
        >
          <div>
            <div style={{ fontSize: 14, fontWeight: 600 }}>
              Al-Barakah AI Assistant
            </div>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              Chat about food, travel, Malisha group services
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {user ? (
              <>
                <span style={{ fontSize: 12, color: "#4b5563" }}>
                  {user.username} ({user.role})
                </span>
                {user.role === "admin" && (
                  <Link to="/admin/ingest" style={{ fontSize: 12 }}>
                    Admin
                  </Link>
                )}
              </>
            ) : (
              <>
                <Link to="/login" style={{ fontSize: 12 }}>
                  Log in
                </Link>
                <span style={{ fontSize: 12, color: "#9ca3af" }}>|</span>
                <Link to="/signup" style={{ fontSize: 12 }}>
                  Sign up
                </Link>
              </>
            )}
          </div>
        </header>

        <main
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "12px 16px",
            background: "linear-gradient(to bottom,#f9fafb,#ffffff)",
          }}
        >
          {messages.map((m, idx) => (
            <div
              key={idx}
              style={{
                display: "flex",
                justifyContent:
                  m.role === "user" ? "flex-end" : "flex-start",
                marginBottom: 8,
              }}
            >
              <div>
                <div
                  style={{
                    padding: "8px 10px",
                    borderRadius: 10,
                    maxWidth: "80%",
                    background:
                      m.role === "user" ? "#dbeafe" : "#f3f4f6",
                    border:
                      m.role === "user"
                        ? "1px solid #bfdbfe"
                        : "1px solid #e5e7eb",
                    whiteSpace: "pre-wrap",
                    fontSize: 14,
                  }}
                >
                  {m.text}
                </div>
                <div
                  style={{
                    fontSize: 11,
                    color: "#6b7280",
                    marginTop: 2,
                  }}
                >
                  {m.role === "user"
                    ? "You"
                    : m.usedWeb
                    ? "Assistant · Web used ✅"
                    : "Assistant"}
                </div>
              </div>
            </div>
          ))}
          {loading && (
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              Assistant is thinking…
            </div>
          )}
        </main>

        <footer
          style={{
            borderTop: "1px solid #e5e7eb",
            padding: "8px 16px 10px",
            background: "#f9fafb",
          }}
        >
          {error && (
            <div
              style={{
                background: "#fee2e2",
                border: "1px solid #fecaca",
                color: "#b91c1c",
                borderRadius: 8,
                padding: "4px 8px",
                fontSize: 12,
                marginBottom: 4,
              }}
            >
              {error}
            </div>
          )}
          <form
            onSubmit={onSubmit}
            style={{ display: "flex", gap: 8, alignItems: "flex-end" }}
          >
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              rows={2}
              style={{
                flex: 1,
                padding: 8,
                borderRadius: 10,
                border: "1px solid #e5e7eb",
                resize: "none",
                fontSize: 14,
              }}
              placeholder="Type a message…"
            />
            <button
              type="submit"
              disabled={loading}
              style={{
                padding: "8px 14px",
                borderRadius: 999,
                border: "none",
                background: "#22c55e",
                color: "#052e16",
                fontSize: 14,
                fontWeight: 500,
                cursor: loading ? "default" : "pointer",
              }}
            >
              {loading ? "Sending…" : "Send"}
            </button>
          </form>
        </footer>
      </div>
    </div>
  );
}
