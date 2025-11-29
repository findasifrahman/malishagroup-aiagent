import {
  Routes,
  Route,
  Navigate,
  Link,
  Outlet,
  useLocation,
} from "react-router-dom";

import ChatPage from "./pages/ChatPage";
import LoginPage from "./pages/LoginPage";
import IngestPage from "./pages/IngestPage";
import MenuPage from "./pages/MenuPage";
import ConversationsPage from "./pages/ConversationsPage";
import PlaygroundPage from "./pages/PlaygroundPage";
import { LeadsPage } from "./pages/LeadsPage";

function AdminLayout() {
  const location = useLocation();

  const isActive = (path: string) =>
    location.pathname === path || location.pathname.startsWith(path + "/");

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        background: "#f3f4f6",
      }}
    >
      {/* Sidebar */}
      <aside
        style={{
          width: 230,
          background: "#0f172a",
          color: "#e5e7eb",
          padding: "16px 12px",
          display: "flex",
          flexDirection: "column",
          gap: 8,
        }}
      >
        <div style={{ marginBottom: 16, fontWeight: 600, fontSize: 16 }}>
          Admin Panel
        </div>

        <Link
          to="/admin/ingest"
          style={{
            padding: "8px 10px",
            borderRadius: 8,
            textDecoration: "none",
            color: isActive("/admin/ingest") ? "#0f172a" : "#e5e7eb",
            background: isActive("/admin/ingest") ? "#22c55e" : "transparent",
            fontSize: 14,
          }}
        >
          Knowledge Ingest
        </Link>

        <Link
          to="/admin/playground"
          style={{
            padding: "8px 10px",
            borderRadius: 8,
            textDecoration: "none",
            color: isActive("/admin/playground") ? "#0f172a" : "#e5e7eb",
            background: isActive("/admin/playground") ? "#22c55e" : "transparent",
            fontSize: 14,
          }}
        >
          Playground
        </Link>

        <Link
          to="/admin/menu"
          style={{
            padding: "8px 10px",
            borderRadius: 8,
            textDecoration: "none",
            color: isActive("/admin/menu") ? "#0f172a" : "#e5e7eb",
            background: isActive("/admin/menu") ? "#22c55e" : "transparent",
            fontSize: 14,
          }}
        >
          Menu Management
        </Link>

        {/* You can enable this later when conversations API is stable */}
        <Link
          to="/admin/conversations"
          style={{
            padding: "8px 10px",
            borderRadius: 8,
            textDecoration: "none",
            color: isActive("/admin/conversations") ? "#0f172a" : "#e5e7eb",
            background: isActive("/admin/conversations")
              ? "#22c55e"
              : "transparent",
            fontSize: 14,
          }}
        >
          Conversations / Complaints
        </Link>

        <Link
          to="/admin/leads"
          style={{
            padding: "8px 10px",
            borderRadius: 8,
            textDecoration: "none",
            color: isActive("/admin/leads") ? "#0f172a" : "#e5e7eb",
            background: isActive("/admin/leads") ? "#22c55e" : "transparent",
            fontSize: 14,
          }}
        >
          Leads
        </Link>

        <div style={{ flexGrow: 1 }} />

        <Link
          to="/"
          style={{
            padding: "8px 10px",
            borderRadius: 8,
            textDecoration: "none",
            color: "#e5e7eb",
            fontSize: 13,
            opacity: 0.8,
          }}
        >
          ⬅ Back to Chat
        </Link>
      </aside>

      {/* Main content */}
      <main
        style={{
          flexGrow: 1,
          padding: 12,
          display: "flex",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            width: "100%",
            maxWidth: 960,
            background: "#ffffff",
            borderRadius: 12,
            boxShadow: "0 10px 30px rgba(15,23,42,0.08)",
            padding: 16,
          }}
        >
          <Outlet />
        </div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      {/* Public chat page – your GPT-like UI */}
      <Route path="/" element={<ChatPage />} />

      {/* Very simple login page (front-end only for now) */}
      <Route path="/login" element={<LoginPage />} />

      {/* Admin area – NO auth guard for now, just for PoC */}
      <Route path="/admin" element={<AdminLayout />}>
        <Route index element={<Navigate to="ingest" />} />
        <Route path="ingest" element={<IngestPage />} />
        <Route path="playground" element={<PlaygroundPage />} />
        <Route path="menu" element={<MenuPage />} />
        <Route path="conversations" element={<ConversationsPage />} />
        <Route path="leads" element={<LeadsPage />} />
      </Route>

      {/* Fallback */}
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  );
}
