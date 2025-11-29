import React, { useEffect, useState } from "react";
import {
  getConversations,
  getConversationMessages,
  getComplaints,
  updateComplaintStatus,
} from "../api";

export default function ConversationsPage() {
  const [convs, setConvs] = useState<any[]>([]);
  const [selectedConvId, setSelectedConvId] = useState<string | null>(null);
  const [messages, setMessages] = useState<any[]>([]);
  const [complaints, setComplaints] = useState<any[]>([]);
  const [status, setStatus] = useState<string | null>(null);

  async function loadConvs() {
    setStatus("Loading conversations…");
    try {
      const data = await getConversations();
      setConvs(data);
      setStatus(null);
    } catch (err: any) {
      setStatus("Error loading convs: " + (err.message || "failed"));
    }
  }

  async function loadMsgs(id: string) {
    setSelectedConvId(id);
    setStatus("Loading messages…");
    try {
      const data = await getConversationMessages(id);
      setMessages(data);
      setStatus(null);
    } catch (err: any) {
      setStatus("Error loading messages: " + (err.message || "failed"));
    }
  }

  async function loadComplaintsFn() {
    try {
      const data = await getComplaints();
      setComplaints(data);
    } catch (err) {
      console.error(err);
    }
  }

  useEffect(() => {
    loadConvs();
    loadComplaintsFn();
  }, []);

  const onUpdateComplaint = async (c: any) => {
    const newStatus = window.prompt(
      "Update status (open, in_progress, resolved):",
      c.status
    );
    if (!newStatus) return;
    try {
      await updateComplaintStatus(c.id, newStatus);
      await loadComplaintsFn();
    } catch (err: any) {
      alert("Error updating complaint: " + (err.message || "failed"));
    }
  };

  return (
    <div style={{ display: "flex", gap: 16 }}>
      <div style={{ flex: 1 }}>
        <h2>Conversations</h2>
        <button onClick={loadConvs} style={{ marginBottom: 8 }}>
          Refresh
        </button>
        <div
          style={{
            border: "1px solid #e5e7eb",
            borderRadius: 8,
            maxHeight: 260,
            overflow: "auto",
          }}
        >
          <table
            style={{ width: "100%", fontSize: 13, borderCollapse: "collapse" }}
          >
            <thead>
              <tr>
                <th>ID (short)</th>
                <th>Domain</th>
                <th>Channel</th>
              </tr>
            </thead>
            <tbody>
              {convs.map((c) => (
                <tr
                  key={c.id}
                  style={{ cursor: "pointer" }}
                  onClick={() => loadMsgs(c.id)}
                >
                  <td>{String(c.id).slice(0, 8)}</td>
                  <td>{c.domain || ""}</td>
                  <td>{c.channel}</td>
                </tr>
              ))}
              {!convs.length && (
                <tr>
                  <td colSpan={3}>No conversations yet.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <h3>Complaints</h3>
        <button onClick={loadComplaintsFn} style={{ marginBottom: 8 }}>
          Refresh
        </button>
        <div
          style={{
            border: "1px solid #e5e7eb",
            borderRadius: 8,
            maxHeight: 260,
            overflow: "auto",
          }}
        >
          <table
            style={{ width: "100%", fontSize: 13, borderCollapse: "collapse" }}
          >
            <thead>
              <tr>
                <th>ID</th>
                <th>Status</th>
                <th>Summary</th>
              </tr>
            </thead>
            <tbody>
              {complaints.map((c) => (
                <tr
                  key={c.id}
                  style={{ cursor: "pointer" }}
                  onClick={() => onUpdateComplaint(c)}
                >
                  <td>{c.id}</td>
                  <td>{c.status}</td>
                  <td>{(c.summary || "").slice(0, 60)}</td>
                </tr>
              ))}
              {!complaints.length && (
                <tr>
                  <td colSpan={3}>No complaints.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {status && (
          <div style={{ marginTop: 8, fontSize: 14, color: "#4b5563" }}>
            {status}
          </div>
        )}
      </div>
      <div style={{ flex: 1 }}>
        <h3>Conversation Messages</h3>
        <div
          style={{
            border: "1px solid #e5e7eb",
            borderRadius: 8,
            padding: 8,
            maxHeight: "80vh",
            overflow: "auto",
            whiteSpace: "pre-wrap",
            fontSize: 13,
          }}
        >
          {selectedConvId ? (
            messages.length ? (
              messages.map((m) => {
                const ts = new Date(m.created_at).toLocaleString();
                return `[${ts}] ${m.role.toUpperCase()} (${m.domain || ""}/${
                  m.intent || ""
                }):\n${m.content}\n\n`;
              })
            ) : (
              "No messages yet."
            )
          ) : (
            "Select a conversation."
          )}
        </div>
      </div>
    </div>
  );
}
