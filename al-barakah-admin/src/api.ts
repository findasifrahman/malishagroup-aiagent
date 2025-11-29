// src/api.ts

const API_BASE = "https://malishagroup-aiagent-production.up.railway.app:8000"//import.meta.env.VITE_API_BASE_URL;// || "http://127.0.0.1:8000";

export type AuthUser = {
  id: number;
  username: string;
  role: string;
};

const TOKEN_KEY = "token";
const USER_KEY = "user";

let token: string | null = localStorage.getItem(TOKEN_KEY);
let cachedUser: AuthUser | null = (() => {
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as AuthUser;
  } catch {
    return null;
  }
})();

export function getToken() {
  return token;
}

export function setToken(t: string | null) {
  token = t;
  if (t) localStorage.setItem(TOKEN_KEY, t);
  else localStorage.removeItem(TOKEN_KEY);
}

export function getUser(): AuthUser | null {
  return cachedUser;
}

export function setUser(u: AuthUser | null) {
  cachedUser = u;
  if (u) localStorage.setItem(USER_KEY, JSON.stringify(u));
  else localStorage.removeItem(USER_KEY);
}

async function request(path: string, options: RequestInit = {}) {
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };

  if (!(options.body instanceof FormData)) {
    // Only set JSON content-type for non-FormData bodies
    headers["Content-Type"] = headers["Content-Type"] || "application/json";
  }

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      if (data && data.detail) detail = data.detail;
    } catch {
      // ignore
    }
    throw new Error(detail || "Request failed");
  }

  if (res.status === 204) return null;
  try {
    return await res.json();
  } catch {
    return null;
  }
}

// ---------- Auth ----------

export async function login(username: string, password: string) {
  return request("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export async function apiSignup(username: string, password: string) {
  return request("/api/auth/signup", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export async function apiMe() {
  return request("/api/auth/me");
}

// ---------- Chat (public) ----------
/* ------------------------------------------------------------------ */
/*  Chat endpoints                                                     */
/* ------------------------------------------------------------------ */

// Public chat – used by ChatPage.tsx
export async function apiChat(
    message: string,
    conversationId?: string | null,
    domainOverride?: string
  ) {
    const body: any = { message };
    if (conversationId) body.conversation_id = conversationId;
    if (domainOverride) body.domain_override = domainOverride; // 👈 NEW
  
    return request("/api/chat", {
      method: "POST",
      body: JSON.stringify(body),
    });
  }
  

// Admin playground chat – used by PlaygroundPage.tsx
export async function adminChat(message: string) {
  return request("/api/admin/chat", {
    method: "POST",
    body: JSON.stringify({ message }),
  });
}


export async function sendChat(message: string, conversationId?: string | null) {
  return request("/api/chat", {
    method: "POST",
    body: JSON.stringify({
      message,
      conversation_id: conversationId || null,
    }),
  });
}

// ---------- Ingestion ----------

export interface Lead {
    id: number;
    created_at: string;
    domain: string;
    name: string | null;
    contact: string | null;
    country: string | null;
    problem_type: string | null;
    visa_support: boolean | null;
    first_question: string | null;
    last_question: string | null;
    age: number | null;
    extra: any;
  }
  
  export async function getLeads(days = 2): Promise<Lead[]> {
    return request(`/api/admin/leads?days=${days}`);
  }
  

// ---------- Admin: conversations / complaints ----------

export async function getConversations() {
  return request("/api/admin/conversations");
}

export async function getConversationMessages(conversationId: string) {
  return request(`/api/admin/conversations/${conversationId}/messages`);
}

export async function getComplaints() {
  return request("/api/admin/complaints");
}

export async function updateComplaintStatus(id: number, status: string) {
  return request(`/api/admin/complaints/${id}/status`, {
    method: "POST",
    body: JSON.stringify({ status }),
  });
}

// ---------- Admin: menu ----------
export interface MenuItemInput {
  category_id: number | null;
  name_en: string;
  name_bn: string | null;
  description: string | null;
  price_cny: number;
  is_available: boolean;
  tags: string[] | null;
}

export async function getMenuItems() {
  return request("/api/admin/menu_items", {
    method: "GET",
  });
}

export async function saveMenuItem(
  id: number | null,
  payload: MenuItemInput
) {
  if (id == null) {
    // create
    return request("/api/admin/menu_items", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  } else {
    // update
    return request(`/api/admin/menu_items/${id}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    });
  }
}

export async function deleteMenuItem(id: number) {
  return request(`/api/admin/menu_items/${id}`, {
    method: "DELETE",
  });
}

// ---------- Knowledge ingestion ----------

export async function ingestText(payload: {
    title: string;
    source: string;
    lang?: string;
    description?: string;
    text: string;
  }) {
    return request("/api/admin/ingest_text", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }
  
  export async function ingestUrl(payload: {
    url: string;
    source: string;
    lang?: string;
    description?: string;
  }) {
    return request("/api/admin/ingest_url", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }
  
  export async function ingestUrlDistilled(payload: {
    url: string;
    source: string;
    lang?: string;
    description?: string;
    entity_hints?: string[];
  }) {
    return request("/api/admin/ingest_url_distilled", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }
  
  export async function distillText(payload: {
    text: string;
    source: string;
    lang?: string;
    description?: string;
    entity_hints?: string[];
  }) {
    return request("/api/admin/distill_text", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }
  
  export async function ingestPdf(
    file: File,
    fields: { title: string; source: string; lang?: string; description?: string }
  ) {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("title", fields.title);
    fd.append("source", fields.source);
    fd.append("lang", fields.lang || "en");
    fd.append("description", fields.description || "");
  
    // IMPORTANT: FormData → let fetch set its own headers; don't force JSON content-type
    return fetch(`${API_BASE}/api/admin/ingest_pdf`, {
      method: "POST",
      body: fd,
    }).then(async (res) => {
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const data = await res.json();
          if (data?.detail) detail = data.detail;
        } catch {}
        throw new Error(detail || "PDF ingest failed");
      }
      try {
        return await res.json();
      } catch {
        return null;
      }
    });
  }
  