// src/pages/LeadsPage.tsx
import React, { useEffect, useState } from "react";
import { getLeads, type Lead } from "../api";

const domainLabel: Record<string, string> = {
  "al-barakah": "Al-Barakah",
  "malisha-edu": "MalishaEdu",
  easylink: "Easylink",
  brcc: "BRCC / Healthcare",
};

export function LeadsPage() {
  const [leads, setLeads] = useState<Lead[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getLeads(2);
        setLeads(data);
      } catch (err: any) {
        setError(err.message || "Failed to load leads");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const counts = leads.reduce<Record<string, number>>((acc, l) => {
    acc[l.domain] = (acc[l.domain] || 0) + 1;
    return acc;
  }, {});

  return (
    <div style={{ padding: 16 }}>
      <h1 style={{ fontSize: 20, fontWeight: 600, marginBottom: 12 }}>
        Leads (last 2 days)
      </h1>

      <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
        {["al-barakah", "malisha-edu", "easylink", "brcc"].map((d) => (
          <div
            key={d}
            style={{
              padding: 10,
              borderRadius: 8,
              background: "#f3f4f6",
              fontSize: 13,
              minWidth: 140,
            }}
          >
            <div style={{ fontWeight: 500 }}>
              {domainLabel[d] || d}
            </div>
            <div style={{ color: "#4b5563" }}>
              {counts[d] || 0} lead(s)
            </div>
          </div>
        ))}
      </div>

      {loading && <div>Loading…</div>}
      {error && (
        <div style={{ color: "red", marginBottom: 8 }}>Error: {error}</div>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: 12,
        }}
      >
        {leads.map((l) => (
          <div
            key={l.id}
            style={{
              border: "1px solid #e5e7eb",
              borderRadius: 10,
              padding: 10,
              background: "white",
              fontSize: 13,
            }}
          >
            <div
              style={{
                fontSize: 12,
                color: "#6b7280",
                marginBottom: 4,
              }}
            >
              {new Date(l.created_at).toLocaleString()} ·{" "}
              {domainLabel[l.domain] || l.domain}
            </div>
            <div style={{ fontWeight: 500, marginBottom: 4 }}>
              {l.name || "Unnamed lead"}
            </div>
            <div style={{ color: "#4b5563", marginBottom: 4 }}>
              {l.contact && <div>📞 {l.contact}</div>}
              {l.country && <div>🌍 {l.country}</div>}
              {l.age && <div>🎂 {l.age}</div>}
            </div>
            <div style={{ marginBottom: 4 }}>
              <div style={{ fontWeight: 500 }}>First question:</div>
              <div style={{ color: "#374151" }}>{l.first_question}</div>
            </div>
            <div>
              <div style={{ fontWeight: 500 }}>Last question:</div>
              <div style={{ color: "#374151" }}>{l.last_question}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
