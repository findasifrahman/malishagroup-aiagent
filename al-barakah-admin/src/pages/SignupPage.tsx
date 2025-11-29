import React, { useState } from "react";
import { Link } from "react-router-dom";
import { apiSignup } from "../api";
import type { AuthUser } from "../api";

interface SignupPageProps {
  setUserAndToken: (u: AuthUser | null, token?: string) => void;
}

const SignupPage: React.FC<SignupPageProps> = ({ setUserAndToken }) => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    setLoading(true);
    try {
      const res: any = await apiSignup(username.trim(), password.trim());
      const user: AuthUser = {
        id: res.user.id,
        username: res.user.username,
        role: res.user.role,
      };
      setUserAndToken(user, res.token);
    } catch (error: any) {
      setErr(error.message || "Signup failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        height: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#f3f4f6",
      }}
    >
      <form
        onSubmit={handleSubmit}
        style={{
          width: 320,
          background: "#ffffff",
          padding: 20,
          borderRadius: 8,
          boxShadow: "0 10px 30px rgba(0,0,0,0.08)",
        }}
      >
        <h2 style={{ marginTop: 0 }}>Sign up</h2>

        <label style={{ display: "block", marginBottom: 6, fontSize: 14 }}>
          Username
        </label>
        <input
          style={{ width: "100%", marginBottom: 8, padding: 6 }}
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />

        <label style={{ display: "block", marginBottom: 6, fontSize: 14 }}>
          Password
        </label>
        <input
          style={{ width: "100%", marginBottom: 8, padding: 6 }}
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        {err && (
          <div style={{ marginBottom: 8, fontSize: 13, color: "#b91c1c" }}>
            {err}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          style={{
            width: "100%",
            padding: 8,
            borderRadius: 6,
            border: "none",
            background: "#22c55e",
            color: "#052e16",
            fontWeight: 500,
            cursor: "pointer",
          }}
        >
          {loading ? "Signing up..." : "Sign up"}
        </button>

        <div style={{ marginTop: 8, fontSize: 12 }}>
          Already have an account? <Link to="/login">Login</Link>
        </div>
      </form>
    </div>
  );
};

export default SignupPage;
