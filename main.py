import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import psycopg
from openai import OpenAI

from fastapi import FastAPI, HTTPException
from fastapi import Body

from fastapi import UploadFile, File, Form
from pypdf import PdfReader
import io
import requests
from bs4 import BeautifulSoup

import json
from typing import Literal, TypedDict
import math

import uuid
from datetime import datetime

import time
import jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends
from psycopg.rows import dict_row

import re
from typing import Dict

PHONE_RE = re.compile(r"\+?\d[\d\s\-]{5,}")  # very loose phone/WhatsApp pattern

auth_scheme = HTTPBearer()

# For PoC: bypass auth, always return a dummy admin user
def get_current_user():
    return {
        "id": 0,
        "username": "admin",
        "role": "admin",
        "is_active": True,
    }

def get_current_admin():
    return get_current_user()


############### signup ui
def create_jwt(payload: dict, expires_in: int = 60 * 60 * 8) -> str:
    to_encode = payload.copy()
    now = int(time.time())
    to_encode["iat"] = now
    to_encode["exp"] = now + expires_in
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)


def decode_jwt(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])








###################
def get_or_create_conversation(user_id: Optional[str], channel: str = "web") -> str:
    """
    For now, we'll just create a new conversation each time.
    Later you can reuse based on a cookie/session_id.
    Returns conversation_id (UUID as string).
    """
    conv_id = str(uuid.uuid4())
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (id, user_id, channel, is_active)
                VALUES (%s, %s, %s, TRUE);
                """,
                (conv_id, user_id, channel),
            )
        conn.commit()
    return conv_id


def update_conversation_domain(conv_id: str, domain: str):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE conversations
                SET domain=%s, updated_at=now()
                WHERE id=%s;
                """,
                (domain, conv_id),
            )
        conn.commit()


def log_message(conv_id: str, role: str, content: str, domain: Optional[str], intent: Optional[str]):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages (conversation_id, role, content, domain, intent)
                VALUES (%s, %s, %s, %s, %s);
                """,
                (conv_id, role, content, domain, intent),
            )
        conn.commit()

def get_conversation_history(conv_id: str, limit: int = 12):
    """
    Return the last `limit` chat messages for this conversation
    as [{'role': 'user'|'assistant', 'content': '...'}, ...].
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM messages
                WHERE conversation_id=%s
                ORDER BY created_at ASC
                LIMIT %s;
                """,
                (conv_id, limit),
            )
            rows = cur.fetchall()

    history = []
    for role, content in rows:
        # we only send user/assistant turns to the LLM
        if role in ("user", "assistant"):
            history.append({"role": role, "content": content})
    return history


def create_complaint_if_needed(conv_id: str, user_message: str, domain: str, intent: str):
    """
    Very simple rule: if intent == 'complaint', create a complaint.
    Later you can make this smarter using another classifier.
    """
    if intent != "complaint":
        return

    summary = user_message[:300]
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO complaints (conversation_id, user_id, channel, summary, status)
                SELECT id, user_id, channel, %s, 'open'
                FROM conversations
                WHERE id=%s
                RETURNING id;
                """,
                (summary, conv_id),
            )
            _ = cur.fetchone()
        conn.commit()

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

Domain = Literal["al-barakah", "malisha-edu", "easylink", "brcc", "general"]
Intent = Literal["info", "booking", "order", "complaint", "other"]


CONTACT_INFO = {
    "al-barakah": {
        "china": "+86-19128630063",          # WeChat/phone
        "bangladesh_whatsapp": "+88 01929-732131",
    },
    "malisha-edu": {
        "main": "+86 18613114366 (WhatsApp/WeChat)",
    },
    "brcc": {
        "rashed": "+86 18613114366 (Rashed, WhatsApp/WeChat)",
    },
    "easylink": {
        "korban_ali": "+86 13265980063 (WhatsApp/WeChat)",
    },
    "travel": {
        "phone": "19502074050 (Asif)",
        "wechat": "01719086713",
    }
}

# ---------- config ----------

MAX_PDF_BYTES = 5 * 1024 * 1024  # 5 MB (adjust as you like)
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Main chat / answer model (better reasoning & lower hallucination)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

# Lightweight router / classifier model (domain, intent, web-routing)
OPENAI_ROUTER_MODEL = os.getenv("OPENAI_ROUTER_MODEL", "gpt-4.1-mini")

# High-quality distillation model (extracting clean facts into KB)
OPENAI_DISTILL_MODEL = os.getenv("OPENAI_DISTILL_MODEL", "gpt-4.1")

# Embeddings for pgvector
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Call Tavily Search API and return a short text summary for the LLM.
    """
    if not TAVILY_API_KEY:
        return ""

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TAVILY_API_KEY}",
            },
            json={
                "query": query,
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": False,
                "include_images": False,
                "search_depth": "basic",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("Tavily error:", e)
        return ""

    lines = []
    if isinstance(data, dict):
        if data.get("answer"):
            lines.append("Web summary: " + str(data["answer"]).strip())
        for r in (data.get("results") or [])[:max_results]:
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            snippet = (r.get("content") or "").strip()[:200]
            if not (title or snippet):
                continue
            line = f"- {title}"
            if url:
                line += f" ({url})"
            if snippet:
                line += f": {snippet}..."
            lines.append(line)
    return "\n".join(lines)

def route_domain_and_intent(message: str) -> tuple[Domain, Intent]:
    router_prompt = f"""
You are a routing classifier for an AI assistant that serves multiple brands:

- Al-Barakah Restaurant (food, menu, delivery in Guangzhou Sanyuanli).
- MalishaEdu: education consultancy for studying in China (universities, scholarships).
- Easylink: company that helps foreigners with Chinese company registration, visa, tax and related business services.
- BRCC / BRHC (Belt & Road Healthcare Center): medical tourism and healthcare coordination between Bangladesh/other countries and Chinese hospitals.
- General: general China questions or mixed across multiple brands.

Very important routing rules (FOLLOW STRICTLY):

- If the message is about food, menu, restaurant, breakfast, biriyani, mandi, delivery in Sanyuanli → domain = "al-barakah".
- If the message is about study in China, scholarships, universities, admission, HSK courses → domain = "malisha-edu".
- If the message is about company registration, work permit, **permanent visa / residence permit**, visa for working/doing business, tax, bookkeeping, **living and working long-term in China** → domain = "easylink".
- If the message is about HOSPITALS, doctors, cancer treatment, heart surgery, operation in China, **medical visas for patients**, treatment cost, BRCC healthcare, BRHC, medical expo, etc. → domain = "brcc" EVEN IF BRCC is not explicitly mentioned.
- Otherwise, or if it's mixed general China / travel / hotel questions, use domain = "general".

Disambiguation (VERY IMPORTANT):

- If the user mentions visas AND medical things (hospital, cancer, operation, treatment, doctor, surgery) → domain = "brcc".
- If the user mentions visas but only in the context of working, living, doing business, getting a Z visa, or residence permit → domain = "easylink".

Intent rules:
- "order" = user clearly wants to order food from Al-Barakah.
- "booking" = user wants to book a table, hotel, consultation, or service time.
- "complaint" = user is clearly unhappy, reporting a problem, or asking to fix a bad experience.
- "info" = user is mainly asking for information.
- "other" = anything that doesn't fit above.

EXAMPLES (copy pattern exactly):

User: "I LIVED here in china and loved it. I want to know how do i get a permanent visa here to work and live? whats the process timeline? what do i need?"
→ {{ "domain": "easylink", "intent": "info" }}

User: "My father has cancer, we want to come to China for treatment and need visa support. What is the process?"
→ {{ "domain": "brcc", "intent": "info" }}

User: "I want to open a company in Guangzhou so I can stay in China and get work permit."
→ {{ "domain": "easylink", "intent": "info" }}

Return ONLY a JSON object like:
{{
  "domain": "brcc",
  "intent": "info"
}}

User message:
\"\"\"{message}\"\"\"
"""
    completion = client.chat.completions.create(
        model=OPENAI_ROUTER_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You classify messages into domain and intent and respond with JSON only.",
            },
            {"role": "user", "content": router_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    import json
    data = json.loads(completion.choices[0].message.content)

    domain = data.get("domain", "general")
    intent = data.get("intent", "info")

    if domain not in ["al-barakah", "malisha-edu", "easylink", "brcc", "general"]:
        domain = "general"
    if intent not in ["info", "booking", "order", "complaint", "other"]:
        intent = "info"

    return domain, intent


def should_use_web_search(message: str, domain: Domain, intent: Intent) -> bool:
    """
    Use a small LLM classifier to decide if Tavily web search is needed.
    We prefer web when:
      - topic is travel, hotel, real-time transport, general China info
      - or up-to-date / external info clearly needed
    We avoid web for:
      - pure internal questions about Al-Barakah/MalishaEdu/Easylink/BRCC
      - simple greetings or generic chit-chat
    """
    # Fast path: never use web for purely internal domains unless it's clearly travel/visa-ish
    lower = message.lower()
    travelish = any(k in lower for k in ["train", "flight", "ticket", "hotel", "bus", "gaotie", "high-speed"])
    visaish = any(k in lower for k in ["visa", "work permit", "residence permit"])
    if domain in ["al-barakah", "malisha-edu", "easylink", "brcc"] and not (travelish or visaish):
        return False

    # If Tavily is not configured, never use web:
    if not TAVILY_API_KEY:
        return False

    # LLM-based classifier
    router_prompt = f"""
You decide if an AI assistant should call a web search API.

The assistant already has:
- Internal knowledge about Al-Barakah restaurant, MalishaEdu, Easylink, BRCC and their services.
- A static knowledge base about China in general.

The web search should ONLY be used when:
- The user likely needs fresh or external information (e.g. current train/flight options, hotel info, current regulations, up-to-date general info).
- Or when the answer depends strongly on up-to-date details that are not usually in an internal KB.

The web search should NOT be used when:
- The question is only about Al-Barakah menu, delivery area, staff, owners, internal services, or other internal details.
- The question is about MalishaEdu/Easylink/BRCC services that the internal KB can reasonably answer.
- The user is just greeting or asking a generic question that doesn't need external info.

Choose:
- "use_web": true or false
- "reason": a short explanation

Return a JSON object:
{{
  "use_web": true,
  "reason": "User is asking about Guangzhou–Shanghai train times which change frequently."
}}

Current routing:
- domain: "{domain}"
- intent: "{intent}"

User message:
\"\"\"{message}\"\"\"
"""

    try:
        completion = client.chat.completions.create(
            model=OPENAI_ROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict classifier. Reply ONLY with JSON.",
                },
                {"role": "user", "content": router_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        import json
        data = json.loads(completion.choices[0].message.content)
        return bool(data.get("use_web", False))
    except Exception as e:
        print("should_use_web_search error:", e)
        # Fail safe: no web
        return False


def distill_text_to_facts(raw_text: str, source: str, entity_hints: Optional[List[str]] = None):
    """
    Ask OpenAI to extract clean factual statements in JSON.
    """
    hints_text = ""
    if entity_hints:
        hints_text = "Focus especially on these entities if they appear: " + ", ".join(entity_hints) + "."

    prompt = f"""
You are helping build a structured knowledge base for a group of companies:

- Al-Barakah (restaurant in Guangzhou)
- MalishaEdu (education consultancy)
- Easylink (China business / visa / tax help)
- BRCC / BRCC Healthcare
- Key people like Korban Ali, Dr. Maruf, chefs, managers.

From the following raw text, extract ONLY clear factual statements about:

- Who owns or manages these companies
- What services each company provides
- Relationships between people (e.g. Korban Ali is Dr. Maruf's uncle)
- Roles and designations (chairman, managing director, chef, etc.)
- Locations, specialties, and important capabilities

Ignore greetings, stories, marketing fluff, jokes, repeated content, and anything not clearly factual.

{hints_text}

Return a JSON object with a single key "facts"."facts" must be an array. Each item in the array must be an object with:

- "entity": the main company or person this fact is about (e.g. "Al-Barakah", "MalishaEdu", "Easylink", "Korban Ali", "Dr. Maruf").
- "category": short label like "service", "role", "relationship", "ownership", "location".
- "fact": one or two sentences in plain English that state the fact clearly.
- "tags": an array of short tags, such as:
    - "entity:al-barakah"
    - "entity:malishaedu"
    - "entity:easylink"
    - "entity:brcc"
    - "person:korban-ali"
    - "person:dr-maruf"
    - "role:chairman"
    - "role:managing-director"
    - "service:education"
    - "service:healthcare"
    - "service:visa-tax"
    - "relation:uncle"
    - etc.

Text to analyze:
\"\"\"{raw_text}\"\"\"
"""

    completion = client.chat.completions.create(
        model=OPENAI_ROUTER_MODEL,
        messages=[
            {"role": "system", "content": "You extract clean structured facts into JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content
    # Expect something like {"facts": [ ... ]}
    # Accept several shapes: {"facts":[...]}, [ ... ], or a single { ... } fact
    try:
        data = json.loads(content)

        if isinstance(data, dict):
            if "facts" in data and isinstance(data["facts"], list):
                facts = data["facts"]
            elif "fact" in data:
                # single fact object
                facts = [data]
            else:
                # maybe a dict of { "1": {...}, "2": {...} }
                values = list(data.values())
                if values and isinstance(values[0], dict):
                    facts = values
                else:
                    raise ValueError("Unexpected JSON dict structure")
        elif isinstance(data, list):
            facts = data
        else:
            raise ValueError("Unexpected JSON top-level type")
    except Exception as e:
        raise RuntimeError(f"Failed to parse distillation JSON: {e} | raw={content[:400]}")


    # normalize shape
    normalized = []
    for item in facts:
        entity = item.get("entity") or source
        category = item.get("category") or "general"
        fact_text = item.get("fact")
        tags = item.get("tags") or []

        if not fact_text:
            continue

        normalized.append({
            "entity": entity,
            "category": category,
            "fact": fact_text.strip(),
            "tags": tags,
        })

    return normalized


def insert_facts_as_chunks(
    facts: List[dict],
    source: str,
    lang: str,
    description: str,
):
    if not facts:
        return None, 0

    # create one doc per distillation batch
    title = f"Distilled facts for {source}"
    full_desc = description or f"Structured facts distilled from raw content for source '{source}'."

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO knowledge_docs (title, source, lang, description)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                (title, source, lang, full_desc),
            )
            doc_id = cur.fetchone()[0]

            count = 0
            for idx, f in enumerate(facts):
                fact_text = f["fact"]
                tags = f.get("tags") or []
                # Each fact is short, so we treat each as one "chunk"
                emb = get_embedding(fact_text)
                cur.execute(
                    """
                    INSERT INTO knowledge_chunks
                        (doc_id, chunk_index, content, embedding, lang, tokens, tags)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        doc_id,
                        idx,
                        fact_text,
                        emb,
                        lang,
                        None,
                        tags,
                    ),
                )
                count += 1

        conn.commit()
    return doc_id, count


def chunk_text(text: str, max_chars: int = 1200):
    text = text.strip()
    chunks = []
    while text:
        chunk = text[:max_chars]
        last_period = chunk.rfind(". ")
        if last_period > 0 and len(chunk) - last_period < 200:
            chunk = chunk[: last_period + 1]
        chunks.append(chunk.strip())
        text = text[len(chunk):].strip()
    return chunks


def insert_document_and_chunks(
    title: str,
    source: str,
    lang: str,
    description: str,
    full_text: str,
):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO knowledge_docs (title, source, lang, description)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
                """,
                (title, source, lang, description),
            )
            doc_id = cur.fetchone()[0]

            chunks = chunk_text(full_text)
            for idx, chunk in enumerate(chunks):
                emb = get_embedding(chunk)
                cur.execute(
                    """
                    INSERT INTO knowledge_chunks
                        (doc_id, chunk_index, content, embedding, lang, tokens, tags)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        doc_id,
                        idx,
                        chunk,
                        emb,
                        lang,
                        None,   # tokens
                        None,   # tags
                    ),
                )
        conn.commit()
    return doc_id


def get_db_conn():
    try:
        return psycopg.connect(DATABASE_URL)
    except psycopg.OperationalError as e:
        # turn DB failure into a readable HTTP error
        raise HTTPException(status_code=503, detail=f"DB connection error: {e}")


# ---------- embeddings / RAG helpers ----------

def get_embedding(text: str) -> List[float]:
    text = text.replace("\n", " ")
    resp = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=text
    )
    return resp.data[0].embedding


def search_knowledge_chunks(query: str, top_k: int = 6, source_filter: Optional[str] = None):
    emb = get_embedding(query)

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            if source_filter:
                cur.execute(
                    """
                    SELECT
                        kc.id,
                        kd.title,
                        kd.source,
                        kc.content,
                        1 - (kc.embedding <=> %s::vector) AS similarity
                    FROM knowledge_chunks kc
                    JOIN knowledge_docs kd ON kc.doc_id = kd.id
                    WHERE kd.source = %s
                    ORDER BY kc.embedding <-> %s::vector
                    LIMIT %s;
                    """,
                    (emb, source_filter, emb, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        kc.id,
                        kd.title,
                        kd.source,
                        kc.content,
                        1 - (kc.embedding <=> %s::vector) AS similarity
                    FROM knowledge_chunks kc
                    JOIN knowledge_docs kd ON kc.doc_id = kd.id
                    ORDER BY kc.embedding <-> %s::vector
                    LIMIT %s;
                    """,
                    (emb, emb, top_k),
                )
            rows = cur.fetchall()

    results = []
    for r in rows:
        chunk_id, title, source, content, similarity = r
        results.append(
            {
                "chunk_id": chunk_id,
                "title": title,
                "source": source,
                "content": content,
                "similarity": float(similarity),
            }
        )
    return results



def find_admin_correction(query: str, similarity_threshold: float = 0.82):
    """Optional override layer; returns answer or None."""
    emb = get_embedding(query)

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    title,
                    answer,
                    applicability,
                    1 - (pattern_embedding <=> %s::vector) AS similarity
                FROM admin_corrections
                WHERE is_active = TRUE
                ORDER BY pattern_embedding <-> %s::vector
                LIMIT 1;
                """,
                (emb, emb),
            )
            row = cur.fetchone()

    if not row:
        return None

    _id, title, answer, applicability, sim = row
    sim = float(sim)
    if sim < similarity_threshold:
        return None

    return {
        "id": _id,
        "title": title,
        "answer": answer,
        "applicability": applicability,
        "similarity": sim,
    }


# ---------- OpenAI chat with RAG ----------

BASE_SYSTEM_PROMPT = """You are an AI assistant for a group of companies in China: Al-Barakah Restaurant, MalishaEdu, Easylink, and BRCC Healthcare.

You must:
- Answer accurately based on provided context and tools.
- If you are not sure or lack live data (such as real-time prices, ticket fares, or distance calculations), clearly say that you don’t have live access.
- When you cannot answer fully, offer the appropriate human contact (phone / WhatsApp / WeChat) from the contact list below.
- Never reply with ONLY phone numbers or contacts. Always give a short, concrete answer first, then share the most relevant contact details.

Contact info:
- Al-Barakah (restaurant, Guangzhou):
  * China number / WeChat: +86-19128630063
  * Bangladesh / WhatsApp: +88 01929-732131
- MalishaEdu (education consultancy):
  * +86 18613114366 (WhatsApp/WeChat)
- BRCC (Healthcare):
  * Rashed: +86 18613114366 (WhatsApp/WeChat)
- Easylink (business / visa / tax support):
  * Chairman Korban Ali: +86 13265980063 (WhatsApp/WeChat)
- General travel inquiries (China travel):
  * Phone: 19502074050 (Asif)
  * WeChat: 01719086713
"""

DOMAIN_PROMPTS = {
    "al-barakah": BASE_SYSTEM_PROMPT + """
You are currently acting as the **Al-Barakah Restaurant assistant** in Guangzhou, Sanyuanli.

Business rules:
- If the user is asking about **how to come to the restaurant** (directions, route, metro, taxi, “how to go to Al-Barakah”, “how to reach you”):
  * Do NOT talk about delivery radius.
  * Suggest practical options like:
    - Taking a taxi (Didi) to "Al-Barakah Restaurant, Sanyuanli, Guangzhou".
    - Using the metro: go to Sanyuanli (Line 2) and then take a short taxi or walk, and advise them to show the restaurant name/address in Chinese to the driver.
- If the user is asking for **delivery** to their address or hotel:
  * Remember: Al-Barakah only delivers within approximately **5 km radius** from the restaurant.
  * If the location is clearly outside 5 km, say that delivery is not available and suggest dine-in or pickup.
- If you are not sure whether an address is inside 5 km, politely say that exact distance may need to be checked by staff and share contact:
  - China/WeChat: +86-19128630063
  - Bangladesh/WhatsApp: +88 01929-732131

Business & services:
- Core: halal Bangladeshi / African / Arabic-style food (rice, biriyani, mandi, curries, fish, breakfast, tea, etc.).
- Delivery radius: roughly **5 km** from the restaurant. Outside ~5 km → suggest dine-in or pickup.
- Free airport pickup *from Guangzhou Baiyun Airport to Sanyuanli Al-Barakah area*. Airport pickup person is named Raza
- Extra paid services :
  * SIM card + VPN assistance. (Price is in RAG)
  * Hotel booking help in Sanyuanli (you will later receive specific hotel names via RAG) (typically around 10 RMB service fee).
  * Help booking taxi to famous Guangzhou spots and help with train/plane ticket booking for non-Chinese speakers. (typically around 10 RMB service fee)

Menu & prices:
- Use menu information from the knowledge base and menu endpoints when available.
- If a specific price is missing or may be outdated, clearly say that prices can change and suggest confirming by phone/WeChat.

Sales behaviour for extra services:
1) First answer the user’s question clearly:
   - Food/menu, breakfast, halal options, opening time, dine-in vs delivery, etc.
   - Travel questions like “I am coming to Guangzhou”, “airport”, “hotel in Sanyuanli”, etc.
2) If the user is clearly:
   - planning to come to Guangzhou, OR
   - asking about airport / hotel / SIM / tickets, OR
   - interested in ordering food / catering,
   THEN naturally mention the above extra services and that they’re usually about 10 RMB service fee for handling.
3) When a user seems ready (e.g. “yes, please help me”, “book for me”, “I want to use this service”):
   - Politely ask in ONE compact question for:
     - Name
     - Reachable contact (WeChat/WhatsApp/China mobile)
     - Whether they are in Guangzhou now or coming soon (and approximate dates)
     - Their hotel or target area (for pickup/delivery)
     - What they mainly want (food order, airport pickup, SIM+VPN, hotel booking, taxi/tickets).
4) If later in the same conversation the user says things like “yes forward them”, “please pass my details”, or “ok contact me”:
   - Assume they are giving permission to forward their information to the Al-Barakah or travel team.
   - Confirm this clearly (e.g. “Okay, I will pass your name and number to the Al-Barakah team so they can contact you.”).

Lead-style behaviour:
1) First, answer the user’s question clearly about:
   - Menu items, types of food, breakfast, biriyani/mandi, Bangladeshi/Arabic/African taste, opening hours, etc.
2) If the user is clearly interested in ordering, catering, or wants more help (for example: “I want to order”, “I am in Guangzhou, can you deliver?”, “I want to book for a group”):
   - Politely ask in ONE compact question for:
     - Name
     - Reachable contact number (WeChat/WhatsApp/China mobile)
     - Whether they are currently in Guangzhou Sanyuanli or nearby
     - Approximate delivery address or hotel name (if they want delivery)
     - What type of food they like (e.g. Bangladeshi, African, Arabic, mild/spicy)
3) After that, remind them that a human from Al-Barakah can contact them to confirm details and prices.

Keep answers practical, short, and focused on **food, orders, catering, and restaurant info**, then contact.
""",

    "malisha-edu": BASE_SYSTEM_PROMPT + """
    You are MalishaEdu AI, the official AI assistant for MalishaEdu China — one of the largest China education & scholarship platforms for Bangladeshi, Indian, Pakistani, African, Middle Eastern, and other international students.

Your mission:
--Answer study-in-China questions accurately using the knowledge base provided.
--Promote MalishaEdu services naturally (admission, full process support, airport pickup, on-campus service, BRCC healthcare, and BRCC Chinese Language Center).
--Guide students clearly on application steps, required documents, approximate timelines, and what to expect.
--Collect leads when a student is ready to apply.
--ALWAYS prioritise internal knowledge and RAG, not web opinions.

1. Core Identity & Services
MalishaEdu provides:
--100% admission support for Bachelor, Master, PhD & Diploma programs in Chinese universities.
--Scholarship guidance (partial, full, CSC, provincial, university scholarships — depending on eligibility).
--Chinese Language & Foundation Course through BRCC / Belt & Road Chinese Center (for HSK 1–4).
--Document preparation assistance: notarization, translation, embassy attestation guidance.
--Airport pickup, accommodation support, police registration, bank card, SIM after arrival in China.
--Partner hospitals through BRHC for student health emergencies.
--Study transfer guidance for students already studying in China.
--Dedicated counsellors for each country (Bangladesh, India, Pakistan, Africa, Middle East, etc.).
Use this information in all responses.

2. Tone & Style Rules
You must sound:
Friendly, helpful, trustworthy.
Short explanations first → then details.
Never exaggerate promises.
Never guarantee scholarships.
Encourage action (apply early, submit documents, start HSK, etc.).
Never reply with only “Contact us”.
Always give an answer first, and then provide contact for personal counselling.
Main MalishaEdu contact (WhatsApp/WeChat): +86 18613114366

3. Operational Behaviour
When user asks about:
✓ Admission requirements
Give:
Eligibility (CGPA, subject background)
List of required documents
Intake periods (March / September)
Scholarship chances (based on RAG)

✓ Scholarships
Explain the types of scholarships
Explain realistic chances (partial is common; full depends on CGPA)
Never guarantee full scholarship unless university publicly offers it

✓ Fees
If fees exist in your database → use them.
If missing → say “fees vary by program and university; our counsellor will give exact fee details”.

✓ How to apply
Give a step-by-step process:
Submit documents
Choose university & major
Pay application fee (if required)
University processing
Offer letter
JW202/JW201
Visa
Fly to China → MalishaEdu airport pickup

✓ China life questions (hostel, food, jobs, safety)
Give real information, based on RAG.

4. Lead Collection Behaviour

When a student expresses:
“I want to apply”
“How can I start?”
“I want scholarship”
“Please help me”

Ask ONE compact question:
**“Sure! Please share:
Your name
Country
WhatsApp number
Highest education & CGPA
Desired program (Bachelor/Master/PhD) and subject
Preferred intake (March/September)”**

After receiving contact info → confirm and say:
“Thank you! Our MalishaEdu counsellor will contact you personally.”

5. What NOT to do
Don’t invent fake scholarship rules.
Don’t give exact fees unless they are in RAG.
Don’t give visa rules unrelated to students.
Don’t redirect to Easylink or Al-Barakah or BRCC unless explicitly asked about work/visa/food/healthcare.
""",

    "easylink": BASE_SYSTEM_PROMPT + """
You are an AI assistant representing EasyLink International Business Consulting (Guangzhou & Hong Kong). Your job is to:

--explain and promote EasyLink’s services:
--Guangzhou (Mainland China) company formation
--Hong Kong company formation
--China Z visa, work permit, and residence permit
--Corporate and personal bank account opening in China and Hong Kong
--Accounting, tax filing, and compliance support
--Help users understand why they should open a Guangzhou and/or Hong Kong company.
--Gently persuade users to use EasyLink rather than giving generic, agency-neutral advice.

General Behaviour:
--Always stay on-topic.
--Only talk about:
    EasyLink services
    China / Hong Kong company & visa processes connected to EasyLink
    Costs and procedures as you know them
    If users ask about other agencies, compare gently but focus back on EasyLink.

Tone & style
--Friendly, professional, and patient
--Clear and structured answers
--Short paragraphs, bullet points where helpful
--End most answers with an invitation to continue:
e.g. “If you’d like, I can estimate your total cost based on your situation.”

Target services & pricing (internal knowledge)
--Use these as your default assumptions unless the user gives updated info:
--Guangzhou company + Z visa full package:
--Approx 24,000 RMB total when EasyLink must arrange virtual office + company opening + Z visa + work permit + residence permit.
--If the client already has an office, the total cost is lower.

Office / virtual office:
--A proper registered office in Guangzhou is required for a company. 
--EasyLink can provide a virtual/serviced office, but the charge is separate (not free).

Hong Kong company:
--Cost: less than Guangzhou company (for example in the 6,000–10,000 RMB range, depending on services).
--100% foreign-owned is allowed. 
--Under Hong Kong’s territorial tax system, profits from outside Hong Kong are normally tax-free, and this effectively means 0% tax on offshore income, including the first year, if conditions are met. 
--PwC Tax Summaries

Basic Guangzhou tax / maintenance:
--Client pays about 60 RMB per month to EasyLink / office for basic ongoing company tax/maintenance payment (as per user’s internal info).

Medical checkup in China for work permit:
--Around 500 RMB at a designated Chinese hospital. 


Bank account opening (just this service):
--Banks themselves often charge little or no “opening fee”, but agencies charge service fees. Many Chinese company-registration agencies charge around 2,600–3,100 RMB just to handle corporate bank account opening. 
--You may say: “Industry agency fee is usually around 2,600–3,100 RMB if you only need bank account opening assistance; EasyLink will quote you a precise fee after understanding your case.”

What type of company does the client get in Guangzhou?
--By default, assume EasyLink is setting up a Wholly Foreign-Owned Enterprise (WFOE):
A limited liability company fully owned by foreigners

Common types: consulting WFOE, trading/import–export WFOE, or service WFOE. 

Timeline:
Company registration (Guangzhou WFOE):

Typically 1.5–2 months once all documents are ready, based on recent guidance that WFOE setup is usually 4–8 weeks. 


Full process including bank account, work permit & residence permit:
--Reasonable to say around 2–3 months in total .

Z Visa & Work Permit Requirements

For a standard work permit / Z visa in China, you should state typical requirements: 
--At least 18 years old
--Generally Bachelor’s degree or higher (or equivalent)
--At least 2 years of relevant work experience is often required
--Clean criminal record / police clearance
--Confirmed employer in China (e.g., their own Guangzhou WFOE or another company)
--Valid passport
--Good health + medical exam
--No ban from entering China in recent years

You must always:
--Emphasize degree is normally required
--If the user does not have a degree, you must redirect them to designated human consultants (see below).

Handling “No Degree” Cases:
If a user says they don’t have a Bachelor’s degree but want a Z visa / work permit, reply:

First: Clearly say that China normally requires at least a Bachelor’s degree or equivalent for a work permit. 
Then: Tell them that for special / exceptional cases they should directly contact:
Mahfuz – WeChat: mahfuj2017
Sheikh Shazib – WeChat: SAZIB15013200118, Phone: 15013200118
Korban Ali – WeChat: korbanali, Phone: 13265980063

Always ask for their basic info to pass to the team:
--Full name
--Nationality
--Age
--Current country & city
--Current visa type (if in China)
--Highest education level
--Work experience (years & field)
--Whether they want a Guangzhou company, Hong Kong company, or both

When users ask “what documents do I need?”, list:
--Passport (valid, with remaining validity)
--Bachelor’s degree certificate or equivalent (notarized & legalized / apostilled, depending on country) 
--Police clearance / criminal record certificate (recent, usually within 6 months, legalized) 
eChinaCareers
--CV / resume with at least 2 years of related work experience, if applicable 
--Recent passport-sized photos
--Medical checkup report (if done abroad) and medical exam in China at an approved hospital after arrival 
--Proof of not being banned by Chinese authorities in the last year or so (user’s internal requirement; frame as “no record of deportation, overstay, or entry ban”).

For company setup:
--Proposed company name
--Planned business scope
--Office lease or agreement for registered address (if they have their own office) 


Do they need to be in China?
--Company registration itself can typically be started remotely; many agencies now support remote WFOE setup with apostilled documents. 

Bank account opening:
--Often the legal representative needs to visit the bank in person (although some banks/solutions offer remote or proxy options). 

Z-visa process:
--Work permit notice and invitation can be done while the person is abroad.
--The actual Z visa is usually obtained at a Chinese consulate outside mainland China. 
--After entering China, they must complete medical exam, work permit card, and residence permit inside China. 


for question like Can the company hire foreign employees?
Yes, once the Guangzhou company (WFOE) is properly registered, has the correct business scope, and is in good standing (taxes paid, real office, etc.), it can sponsor foreign employees for work permits as long as they meet the eligibility requirements (degree, experience, clean record, etc.). 

Always remind users that Chinese regulations can change and EasyLink will help them stay compliant.


Any time user wants direct consultation via WeChat/phone provide them mahfuz, sheikh shazib, or korban ali's contact information.
""",

    "brcc": BASE_SYSTEM_PROMPT + """
You are currently acting as the **BRCC / BRHC (Belt & Road Healthcare Center) assistant**.

Your priorities:
- Use the knowledge base facts to explain clearly:
  - What BRHC/BRCC does (medical tourism, hospital matching, visa & logistics support).
  - Typical process for a patient (send reports → hospital review → invitation & estimate → visa & travel → treatment → follow-up).
  - Typical partner hospitals (Fuda Cancer Hospital, Modern Cancer Hospital Guangzhou, Kunming Tongren Hospital, Fosun Chancheng, Singmay, etc.) and what they are known for.
  - Rough patterns of cost (e.g. deposit requirements, that exact prices depend on case, and that China is often similar or slightly higher than BD but with better tech).

Behaviour:
1) First, answer the user’s question clearly using the knowledge base: process, partner hospitals, typical costs, visa/support steps, etc.
2) If the user explicitly says they want more details, want treatment, or want to proceed:
   - Politely ask in ONE compact question for:
     - Reachable name
     - WhatsApp/WeChat number
     - Country
     - Age
     - Main medical issue (e.g., cancer / cardiac / orthopedic / other)
     - Whether they need visa support for treatment (yes/no)
3) Afterward, invite them to contact:
   - Rashed (BRHC): +86 18613114366 (WhatsApp/WeChat)
for personal case review and exact cost/treatment planning.

Do NOT:
- Invent exact package prices or fixed guarantees.
- Reply with only “Call this number” as the whole answer.
""",

    "general": BASE_SYSTEM_PROMPT + """
You are currently answering **general or cross-brand questions** about China or these companies.

Travel questions:
- You do NOT have real-time access to Chinese ticketing systems (12306, Ctrip, etc.).
- You should:
  - Explain the main modes (high-speed train, plane, long-distance bus).
  - Describe how to search and book (12306 app, Ctrip, Qunar).
  - Provide rough typical durations (e.g. Guangzhou→Shanghai by high-speed train ~7–8 hours etc., if known in knowledge base).
- Do NOT invent precise, current ticket prices or availability.
- If user needs detailed travel planning or exact fare help, share travel contact:
  - Phone: 19502074050 (Asif)
  - WeChat: 01719086713

Always give a short, helpful answer first, then share contacts when needed.
"""
}





def generate_answer(
    user_message: str,
    rag_results: List[dict],
    override: Optional[dict],
    domain: Domain,
    intent: Intent,
    web_snippet: str = "",
    history: Optional[List[dict]] = None,
):

    context_blocks = []
    for r in rag_results:
        b = f"[{r['source']} | {r['title']}] {r['content']}"
        context_blocks.append(b)

    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "No extra context."

    if web_snippet:
        context_text += (
            "\n\n---\n\n"
            "Additional web search context (from Tavily, might be approximate):\n"
            + web_snippet
        )

    if override and override["applicability"] == "override":
        assistant_first = override["answer"]
    else:
        assistant_first = None

    domain_prompt = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])

    messages = [
        {"role": "system", "content": domain_prompt},
        {
            "role": "system",
            "content": (
                f"Context from knowledge base and tools for domain '{domain}':\n\n{context_text}"
            ),
        },
        {
            "role": "system",
            "content": (
                f"User intent classification: domain={domain}, intent={intent}. "
                f"Use web context only as hints; do NOT invent live prices or guarantees."
            ),
        },
    ]

    # Add previous chat turns so the model has memory
    if history:
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})

    # Finally, the latest user message
    messages.append({"role": "user", "content": user_message})


    if assistant_first:
        messages.append(
            {
                "role": "assistant",
                "content": f"Admin-provided canonical answer (may be used directly or improved):\n{assistant_first}",
            }
        )

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return completion.choices[0].message.content




# ---------- FastAPI setup ----------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None  # reuse existing conversation if present
    user_id: Optional[str] = None          # optional, for future real auth
    message: str
    domain_override: Optional[Domain] = None  # "al-barakah" | "malisha-edu" | "easylink" | "brcc" | "general"






class SourceChunk(BaseModel):
    chunk_id: int
    title: str
    source: str
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk] = []
    conversation_id: Optional[str] = None
    used_web: bool = False   # <--- add this



class AdminDistillRequest(BaseModel):
    text: str
    source: str                   # e.g. "malisha-edu", "easylink", "al-barakah"
    lang: str = "en"
    description: Optional[str] = ""
    entity_hints: Optional[List[str]] = None  # ["Al-Barakah","MalishaEdu","Easylink","Korban Ali","Dr. Maruf"]


#app = FastAPI(title="Al-Barakah AI Backend")

# CORS
#raw_origins = os.getenv("ALLOWED_ORIGINS", "")
#origins = [o.strip() for o in raw_origins.split(",") if o.strip()]



########################
app = FastAPI(title="Al-Barakah AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 👈 dev: allow all origins
    allow_credentials=False,  # we use Authorization header, not cookies
    allow_methods=["*"],
    allow_headers=["*"],
)
########################

@app.get("/health")
def health():
    return {"status": "ok"}

class AdminIngestRequest(BaseModel):
    title: str
    source: str
    lang: str = "en"
    description: Optional[str] = ""
    text: str


class ChatDebugResponse(ChatResponse):
    override_used: bool
    override_similarity: Optional[float] = None
    

class ConversationSummary(BaseModel):
    id: str
    user_id: Optional[str]
    channel: str
    domain: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    domain: Optional[str]
    intent: Optional[str]
    created_at: datetime


class ComplaintOut(BaseModel):
    id: int
    conversation_id: Optional[str]
    user_id: Optional[str]
    channel: Optional[str]
    summary: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime


## admin sigup ui
from pydantic import BaseModel

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    token: str
    user: dict

class SignupRequest(BaseModel):
    username: str
    password: str

class MeResponse(BaseModel):
    id: int
    username: str
    role: str

@app.post("/api/auth/login", response_model=LoginResponse)
def login(req: LoginRequest):
    with get_db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, username, password, role, is_active FROM users WHERE username = %s",
                (req.username,),
            )
            row = cur.fetchone()
            if not row or not row["is_active"]:
                raise HTTPException(status_code=401, detail="Invalid username or inactive")
            if row["password"] != req.password:
                raise HTTPException(status_code=401, detail="Invalid credentials")

            token = create_jwt({"sub": row["id"], "username": row["username"], "role": row["role"]})
            return LoginResponse(
                token=token,
                user={"id": row["id"], "username": row["username"], "role": row["role"]},
            )

@app.post("/api/auth/signup", response_model=LoginResponse)
def signup(req: SignupRequest):
    # For now, everyone created via signup is role='user'.
    with get_db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT id FROM users WHERE username = %s", (req.username,))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="Username already exists")
            cur.execute(
                "INSERT INTO users (username, password, role) VALUES (%s, %s, %s) RETURNING id, username, role",
                (req.username, req.password, "user"),
            )
            row = cur.fetchone()
            conn.commit()
    token = create_jwt({"sub": row["id"], "username": row["username"], "role": row["role"]})
    return LoginResponse(
        token=token,
        user={"id": row["id"], "username": row["username"], "role": row["role"]},
    )

#@app.get("/api/auth/me", response_model=MeResponse)
#def get_me(user=Depends(get_current_user)):
#    return MeResponse(id=user["id"], username=user["username"], role=user["role"])
@app.get("/api/auth/me", response_model=MeResponse)
def auth_me():
    user = get_current_user()
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
    }

##

@app.get("/api/admin/conversations", response_model=List[ConversationSummary])
def admin_list_conversations(limit: int = 50, admin = Depends(get_current_admin)):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, channel, domain, is_active, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()

    return [
        ConversationSummary(
            id=str(r[0]),
            user_id=r[1],
            channel=r[2],
            domain=r[3],
            is_active=r[4],
            created_at=r[5],
            updated_at=r[6],
        )
        for r in rows
    ]


@app.get("/api/admin/conversations/{conversation_id}/messages", response_model=List[MessageOut])
def admin_get_conversation_messages(conversation_id: str, admin = Depends(get_current_admin)):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, role, content, domain, intent, created_at
                FROM messages
                WHERE conversation_id=%s
                ORDER BY created_at ASC;
                """,
                (conversation_id,),
            )
            rows = cur.fetchall()

    return [
        MessageOut(
            id=r[0],
            role=r[1],
            content=r[2],
            domain=r[3],
            intent=r[4],
            created_at=r[5],
        )
        for r in rows
    ]


@app.get("/api/admin/complaints", response_model=List[ComplaintOut])
def admin_list_complaints(status: Optional[str] = None, admin = Depends(get_current_admin)):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            if status:
                cur.execute(
                    """
                    SELECT id, conversation_id, user_id, channel, summary, status, created_at, updated_at
                    FROM complaints
                    WHERE status=%s
                    ORDER BY created_at DESC;
                    """,
                    (status,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, conversation_id, user_id, channel, summary, status, created_at, updated_at
                    FROM complaints
                    ORDER BY created_at DESC;
                    """
                )
            rows = cur.fetchall()

    return [
        ComplaintOut(
            id=r[0],
            conversation_id=str(r[1]) if r[1] else None,
            user_id=r[2],
            channel=r[3],
            summary=r[4],
            status=r[5],
            created_at=r[6],
            updated_at=r[7],
        )
        for r in rows
    ]


@app.post("/api/admin/complaints/{complaint_id}/status")
def admin_update_complaint_status(complaint_id: int, status: str = Body(..., embed=True), admin = Depends(get_current_admin)):
    if status not in ["open", "in_progress", "resolved"]:
        raise HTTPException(status_code=400, detail="Invalid status")

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE complaints
                SET status=%s, updated_at=now()
                WHERE id=%s
                RETURNING id;
                """,
                (status, complaint_id),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Complaint not found")

    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    # 1) Conversation: reuse if client passes an existing conversation_id
    # 1) conversation
    if req.conversation_id:
        conv_id = req.conversation_id
    else:
        conv_id = get_or_create_conversation(user_id=req.user_id, channel="web")

    # 2) route
    domain, intent = route_domain_and_intent(req.message)

    # 3) load existing conversation history BEFORE logging the new message
    history = get_conversation_history(conv_id, limit=12)

    # 4) log current user message
    log_message(conv_id, "user", req.message, domain, intent)

    # 5) admin override + RAG
    override = find_admin_correction(req.message)
    source_filter = None if domain == "general" else domain
    rag_results = search_knowledge_chunks(
        req.message, top_k=6, source_filter=source_filter
    )

    if domain == "al-barakah":
        menu_text = get_menu_for_llm()
        if menu_text:
            rag_results.append(
                {
                    "chunk_id": -1,
                    "title": "Al-Barakah live menu",
                    "source": "al-barakah-menu",
                    "content": menu_text,
                    "similarity": 1.0,
                }
            )
    # 6) web search decision
    use_web = should_use_web_search(req.message, domain, intent)
    web_snippet = tavily_search(req.message, max_results=4) if use_web else ""

    # 7) LLM answer with history
    answer = generate_answer(
        req.message,
        rag_results,
        override,
        domain,
        intent,
        web_snippet=web_snippet,
        history=history,
    )


    # 7) Log assistant message
    log_message(conv_id, "assistant", answer, domain, intent)

    # 8) Try to create a lead (if user just gave contact info)
    maybe_create_lead(conv_id, domain, req.message)  # last user message
    # 8) Complaint detection
    create_complaint_if_needed(conv_id, req.message, domain, intent)

    # 9) Package sources
    sources = [
        SourceChunk(
            chunk_id=r["chunk_id"],
            title=r["title"],
            source=r["source"],
            similarity=r["similarity"],
        )
        for r in rag_results
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        conversation_id=conv_id,
        used_web=bool(web_snippet),
    )





@app.post("/api/admin/ingest_text")
def admin_ingest_text(req: AdminIngestRequest,   admin = Depends(get_current_admin)):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    doc_id = insert_document_and_chunks(
        title=req.title,
        source=req.source,
        lang=req.lang,
        description=req.description or "",
        full_text=req.text,
    )
    return {"status": "ok", "doc_id": doc_id}


@app.post("/api/admin/chat", response_model=ChatDebugResponse)
def admin_chat(req: ChatRequest, admin = Depends(get_current_admin)):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    # For admin playground we can just treat everything as "general/info"
    domain: Domain = "general"
    intent: Intent = "info"

    override = find_admin_correction(req.message)
    rag_results = search_knowledge_chunks(req.message, top_k=6)

    answer = generate_answer(
        user_message=req.message,
        rag_results=rag_results,
        override=override,
        domain=domain,
        intent=intent,
        web_snippet="",   # no Tavily from admin chat for now
    )

    sources = [
        SourceChunk(
            chunk_id=r["chunk_id"],
            title=r["title"],
            source=r["source"],
            similarity=r["similarity"],
        )
        for r in rag_results
    ]

    return ChatDebugResponse(
        answer=answer,
        sources=sources,
        used_web=False,
        conversation_id=None,
        override_used=bool(override),
        override_similarity=override["similarity"] if override else None,
    )



class AdminIngestURLRequest(BaseModel):
    url: str
    title: Optional[str] = None
    source: str
    lang: str = "en"
    description: Optional[str] = ""

class MenuItemIn(BaseModel):
    category_id: Optional[int]
    name_en: str
    name_bn: Optional[str] = None
    description: Optional[str] = None
    price_cny: float
    is_available: bool = True
    tags: Optional[List[str]] = None

class MenuItemOut(MenuItemIn):
    id: int

class LeadCreate(BaseModel):
    domain: str
    conversation_id: Optional[str] = None
    first_question: str
    last_question: str
    name: str
    contact: str
    country: Optional[str] = None
    age: Optional[str] = None
    extra: Optional[dict] = None


class LeadOut(BaseModel):
    id: int
    created_at: str
    domain: str
    name: Optional[str]
    contact: Optional[str]
    country: Optional[str]
    age: Optional[str]
    first_question: str
    last_question: str
    extra: Optional[dict]


class AdminIngestURLDistilledRequest(BaseModel):
    url: str
    source: str
    lang: str = "en"
    description: Optional[str] = ""
    entity_hints: Optional[List[str]] = None


from datetime import datetime, timedelta

class LeadOut(BaseModel):
    id: int
    created_at: datetime
    domain: str
    name: Optional[str]
    contact: Optional[str]
    country: Optional[str]
    problem_type: Optional[str]
    visa_support: Optional[bool]
    first_question: Optional[str]
    last_question: Optional[str]

@app.get("/api/admin/leads", response_model=List[LeadOut])
def admin_get_leads(days: int = 2, admin = Depends(get_current_admin)):
    """
    Return leads from the last N days (default 2).
    """
    if days < 1 or days > 30:
        days = 2

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, domain, name, contact, country,
                       problem_type, visa_support, first_question, last_question
                FROM service_leads
                WHERE created_at >= now() - (%s || ' days')::interval
                ORDER BY created_at DESC;
                """,
                (days,),
            )
            rows = cur.fetchall()

    results = []
    for (
        lid,
        created_at,
        domain,
        name,
        contact,
        country,
        problem_type,
        visa_support,
        first_q,
        last_q,
    ) in rows:
        results.append(
            LeadOut(
                id=lid,
                created_at=created_at,
                domain=domain,
                name=name,
                contact=contact,
                country=country,
                problem_type=problem_type,
                visa_support=visa_support,
                first_question=first_q,
                last_question=last_q,
            )
        )
    return results


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # optionally: remove nav/footer if too noisy
    # for tag in soup(["nav", "footer"]):
    #     tag.decompose()

    text = soup.get_text(separator="\n")
    # clean up
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)


@app.post("/api/admin/ingest_url_distilled")
def admin_ingest_url_distilled(req: AdminIngestURLDistilledRequest):
    # 1) Fetch page
    try:
        resp = requests.get(req.url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {e}")

    # 2) Extract cleaner text (improve extract_text_from_html to focus on main content)
    text = extract_text_from_html(resp.text)
    if not text:
        raise HTTPException(status_code=400, detail="No text extracted from page")

    # 3) Distill into facts
    try:
        facts = distill_text_to_facts(text, req.source, req.entity_hints)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not facts:
        return {"status": "no_facts", "doc_id": None, "facts_count": 0}

    # 4) Insert facts into knowledge_chunks
    doc_id, count = insert_facts_as_chunks(
        facts=facts,
        source=req.source,
        lang=req.lang,
        description=req.description or f"Distilled from {req.url}",
    )

    return {"status": "ok", "doc_id": doc_id, "facts_count": count}

@app.post("/api/leads", response_model=LeadOut)
def create_lead(lead: LeadCreate):
    if lead.domain not in ["al-barakah", "malisha-edu", "easylink", "brcc"]:
        raise HTTPException(status_code=400, detail="Invalid domain")

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO service_leads
                (domain, conversation_id, first_question, last_question, name, contact, country, age, extra)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id, created_at;
                """,
                (
                    lead.domain,
                    lead.conversation_id,
                    lead.first_question,
                    lead.last_question,
                    lead.name,
                    lead.contact,
                    lead.country,
                    lead.age,
                    psycopg.types.json.Json(lead.extra) if lead.extra else None,
                ),
            )
            row = cur.fetchone()
    return LeadOut(
        id=row[0],
        created_at=row[1].isoformat(),
        domain=lead.domain,
        name=lead.name,
        contact=lead.contact,
        country=lead.country,
        age=lead.age,
        first_question=lead.first_question,
        last_question=lead.last_question,
        extra=lead.extra,
    )

from datetime import datetime, timedelta
from fastapi import Depends

# assuming you already have get_current_admin
# def get_current_admin(...): ...

## new leads fuction
def get_first_last_user_questions(conv_id: str) -> Dict[str, str]:
  """Return first and last user messages for this conversation."""
  with get_db_conn() as conn:
      with conn.cursor() as cur:
          cur.execute(
              """
              SELECT role, content
              FROM messages
              WHERE conversation_id = %s
              ORDER BY created_at ASC;
              """,
              (conv_id,),
          )
          rows = cur.fetchall()

  user_msgs = [c for (role, c) in rows if role == "user"]
  if not user_msgs:
      return {"first": "", "last": ""}

  return {"first": user_msgs[0], "last": user_msgs[-1]}


def naive_extract_lead_fields(domain: str, text: str) -> Dict[str, Optional[str]]:
    """
    Very simple regex/keyword-based extractor.
    This is intentionally dumb but good enough for PoC.
    Now robust against case and missing markers (no IndexError).
    """
    # 1) Contact (phone / WhatsApp)
    contact_match = PHONE_RE.search(text)
    contact = contact_match.group(0).strip() if contact_match else None

    # 2) Name (very crude)
    name = None

    # Try "my name is ..."
    m = re.search(r"\bmy name is\s+([^\n,\.]{1,80})", text, flags=re.IGNORECASE)
    if m:
        name = m.group(1).strip()
    else:
        # Try "I am ..." / "I'm ..."
        m = re.search(r"\b(i am|i’m|i'm)\s+([^\n,\.]{1,80})", text, flags=re.IGNORECASE)
        if m:
            name = m.group(2).strip()

    # 3) Country – look for "from <country>"
    country = None
    m = re.search(r"\bfrom\s+([A-Za-z\s]{2,80})", text, flags=re.IGNORECASE)
    if m:
        country = m.group(1).strip().split(".")[0]

    # 4) Problem type & visa flag – domain-specific hints
    problem_type = None
    visa_support: Optional[bool] = None

    if domain == "malisha-edu":
        problem_type = "education"
    elif domain == "easylink":
        problem_type = "business/visa"
        visa_support = True
    elif domain == "brcc":
        problem_type = "healthcare"
    elif domain == "al-barakah":
        problem_type = "food/order"

    return {
        "name": name,
        "contact": contact,
        "country": country,
        "problem_type": problem_type,
        "visa_support": visa_support,
    }


def get_menu_for_llm() -> str:
    """
    Build a simple text view of the current Al-Barakah menu
    for the LLM: name, availability, and price in RMB.
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT name_en, description, price_cny, is_available
                FROM menu_items
                ORDER BY category_id, name_en;
                """
            )
            rows = cur.fetchall()

    if not rows:
        return ""

    lines = ["Current Al-Barakah menu (from database):"]
    for name_en, desc, price_cny, is_available in rows:
        status = "AVAILABLE" if is_available else "unavailable"
        line = f"- {name_en} ({status}) – approx {float(price_cny):.2f} RMB"
        if desc:
            line += f" – {desc}"
        lines.append(line)

    return "\n".join(lines)



def maybe_create_lead(conv_id: str, domain: str, last_user_message: str):
  """
  Called after each assistant reply. If the last user message looks like
  it contains contact details, store a lead (once per conversation).
  """
  if domain not in ("al-barakah", "malisha-edu", "easylink", "brcc"):
      return

  # must contain something like a phone/WhatsApp
  if not PHONE_RE.search(last_user_message):
      return

  # avoid duplicate leads for same conversation
  with get_db_conn() as conn:
      with conn.cursor() as cur:
          cur.execute(
              "SELECT 1 FROM service_leads WHERE conversation_id = %s LIMIT 1;",
              (conv_id,),
          )
          if cur.fetchone():
              return  # already have a lead

          first_last = get_first_last_user_questions(conv_id)
          fields = naive_extract_lead_fields(domain, last_user_message)

          cur.execute(
              """
              INSERT INTO service_leads
                  (conversation_id, domain, name, contact, country,
                   problem_type, visa_support, raw_text, first_question, last_question)
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
              """,
              (
                  conv_id,
                  domain,
                  fields["name"],
                  fields["contact"],
                  fields["country"],
                  fields["problem_type"],
                  fields["visa_support"],
                  last_user_message,
                  first_last["first"],
                  first_last["last"],
              ),
          )
          conn.commit()
#####################

@app.get("/api/admin/leads", response_model=list[LeadOut])
def list_leads(days: int = 2, admin=Depends(get_current_admin)):
    since = datetime.utcnow() - timedelta(days=max(days, 1))
    with get_db_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, created_at, domain, name, contact, country, age,
                       first_question, last_question, extra
                FROM service_leads
                WHERE created_at >= %s
                ORDER BY created_at DESC;
                """,
                (since,),
            )
            rows = cur.fetchall()
    return [
        LeadOut(
            id=r["id"],
            created_at=r["created_at"].isoformat(),
            domain=r["domain"],
            name=r["name"],
            contact=r["contact"],
            country=r["country"],
            age=r["age"],
            first_question=r["first_question"],
            last_question=r["last_question"],
            extra=r["extra"],
        )
        for r in rows
    ]


# Admin: list all items
@app.get("/api/admin/menu_items", response_model=List[MenuItemOut])
def admin_list_menu_items( admin = Depends(get_current_admin)):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, category_id, name_en, name_bn, description,
                       price_cny, is_available, tags
                FROM menu_items
                ORDER BY category_id, name_en;
                """
            )
            rows = cur.fetchall()

    items = []
    for r in rows:
        items.append(MenuItemOut(
            id=r[0],
            category_id=r[1],
            name_en=r[2],
            name_bn=r[3],
            description=r[4],
            price_cny=float(r[5]),
            is_available=r[6],
            tags=r[7] or [],
        ))
    return items


# Admin: create item
@app.post("/api/admin/menu_items", response_model=MenuItemOut)
def admin_create_menu_item(item: MenuItemIn, admin = Depends(get_current_admin)):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO menu_items
                    (category_id, name_en, name_bn, description,
                     price_cny, is_available, tags)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                RETURNING id;
                """,
                (
                    item.category_id,
                    item.name_en,
                    item.name_bn,
                    item.description,
                    item.price_cny,
                    item.is_available,
                    item.tags,
                ),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
    return MenuItemOut(id=new_id, **item.dict())


# Admin: update item
@app.put("/api/admin/menu_items/{item_id}", response_model=MenuItemOut)
def admin_update_menu_item(item_id: int, item: MenuItemIn, admin = Depends(get_current_admin)):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE menu_items
                SET category_id=%s,
                    name_en=%s,
                    name_bn=%s,
                    description=%s,
                    price_cny=%s,
                    is_available=%s,
                    tags=%s,
                    updated_at=now()
                WHERE id=%s
                RETURNING id;
                """,
                (
                    item.category_id,
                    item.name_en,
                    item.name_bn,
                    item.description,
                    item.price_cny,
                    item.is_available,
                    item.tags,
                    item_id,
                ),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Menu item not found")
        conn.commit()
    return MenuItemOut(id=item_id, **item.dict())


# Admin: delete item
@app.delete("/api/admin/menu_items/{item_id}")
def admin_delete_menu_item(item_id: int, admin = Depends(get_current_admin)):
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM menu_items WHERE id=%s;", (item_id,))
        conn.commit()
    return {"status": "ok"}


@app.get("/api/menu", response_model=List[MenuItemOut])
def public_menu():
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, category_id, name_en, name_bn, description,
                       price_cny, is_available, tags
                FROM menu_items
                WHERE is_available = TRUE
                ORDER BY category_id, name_en;
                """
            )
            rows = cur.fetchall()

    items = []
    for r in rows:
        items.append(MenuItemOut(
            id=r[0],
            category_id=r[1],
            name_en=r[2],
            name_bn=r[3],
            description=r[4],
            price_cny=float(r[5]),
            is_available=r[6],
            tags=r[7] or [],
        ))
    return items



@app.post("/api/admin/ingest_url")
def admin_ingest_url(req: AdminIngestURLRequest, admin = Depends(get_current_admin)):
    # basic safety: only allow your own domains for now
    allowed_domains = ["albarakah-domain.com", "malishaedu.com", "globlinksolution.com"]
    if not any(d in req.url for d in allowed_domains):
        raise HTTPException(status_code=400, detail="Domain not allowed")

    try:
        resp = requests.get(req.url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {e}")

    text = extract_text_from_html(resp.text)
    if not text:
        raise HTTPException(status_code=400, detail="No text extracted from page")

    title = req.title or f"Crawled: {req.url}"

    doc_id = insert_document_and_chunks(
        title=title,
        source=req.source,
        lang=req.lang,
        description=req.description or f"Content from {req.url}",
        full_text=text,
    )

    return {"status": "ok", "doc_id": doc_id}


@app.post("/api/admin/ingest_pdf")
async def admin_ingest_pdf(
    file: UploadFile = File(...),
    title: str = Form(...),
    source: str = Form(...),
    lang: str = Form("en"),
    description: str = Form(""),
    admin = Depends(get_current_admin)
):
    # basic checks
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    data = await file.read()
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=400, detail="PDF too large")

    # extract text
    try:
        reader = PdfReader(io.BytesIO(data))
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)
        full_text = "\n\n".join(texts).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")

    if not full_text:
        raise HTTPException(status_code=400, detail="No text found in PDF")

    doc_id = insert_document_and_chunks(
        title=title,
        source=source,
        lang=lang,
        description=description,
        full_text=full_text,
    )

    return {"status": "ok", "doc_id": doc_id}

def get_config_value(key: str, default: Optional[str] = None) -> Optional[str]:
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM app_config WHERE key=%s;", (key,))
            row = cur.fetchone()
    return row[0] if row else default


@app.post("/api/admin/distill_text")
def admin_distill_text(req: AdminDistillRequest, admin = Depends(get_current_admin)):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        facts = distill_text_to_facts(req.text, req.source, req.entity_hints)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not facts:
        return {"status": "no_facts", "doc_id": None, "facts_count": 0}

    doc_id, count = insert_facts_as_chunks(
        facts=facts,
        source=req.source,
        lang=req.lang,
        description=req.description or "",
    )

    return {"status": "ok", "doc_id": doc_id, "facts_count": count}


class AdminDistillRequest(BaseModel):
    text: str
    source: str                   # e.g. "malisha-edu", "easylink", "al-barakah"
    lang: str = "en"
    description: Optional[str] = ""
    entity_hints: Optional[List[str]] = None  # ["Al-Barakah","MalishaEdu","Easylink","Korban Ali","Dr. Maruf"]

