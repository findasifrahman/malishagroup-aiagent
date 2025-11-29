import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);

    if (!username || !password) {
      setErr("Please enter username and password (dummy for now).");
      return;
    }

    setLoading(true);

    // For PoC: no backend login; just go to admin
    setTimeout(() => {
      setLoading(false);
      navigate("/admin/ingest");
    }, 300);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#f3f4f6",
        padding: 12,
      }}
    >
      <form
        onSubmit={onSubmit}
        style={{
          width: "100%",
          maxWidth: 360,
          background: "#ffffff",
          padding: 20,
          borderRadius: 12,
          boxShadow: "0 10px 30px rgba(15,23,42,0.12)",
        }}
      >
        <h2 style={{ marginTop: 0, marginBottom: 12 }}>Admin Login (PoC)</h2>
        <p style={{ marginTop: 0, marginBottom: 16, fontSize: 13, color: "#6b7280" }}>
          For now this is just a front-end gate. Any username/password will
          take you to the admin panel.
        </p>

        <label style={{ display: "block", marginBottom: 4, fontSize: 14 }}>
          Username
        </label>
        <input
          style={{
            width: "100%",
            marginBottom: 10,
            padding: 8,
            borderRadius: 8,
            border: "1px solid #e5e7eb",
          }}
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />

        <label style={{ display: "block", marginBottom: 4, fontSize: 14 }}>
          Password
        </label>
        <input
          type="password"
          style={{
            width: "100%",
            marginBottom: 10,
            padding: 8,
            borderRadius: 8,
            border: "1px solid #e5e7eb",
          }}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        {err && (
          <div
            style={{
              marginBottom: 10,
              fontSize: 13,
              color: "#b91c1c",
              background: "#fee2e2",
              borderRadius: 8,
              padding: "4px 8px",
            }}
          >
            {err}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          style={{
            width: "100%",
            padding: 8,
            borderRadius: 999,
            border: "none",
            background: "#22c55e",
            color: "#052e16",
            fontWeight: 500,
            cursor: loading ? "default" : "pointer",
          }}
        >
          {loading ? "Entering..." : "Enter Admin Panel"}
        </button>
      </form>
    </div>
  );
}
