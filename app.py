"""
Season Radar — Streamlit Web App
Seasonal travel decision engine powered by Claude.

Run locally:
    streamlit run app.py

Deploy: Streamlit Community Cloud (share.streamlit.io)
"""

import os
import json
from datetime import datetime

import anthropic
import streamlit as st
from dotenv import load_dotenv

from scoring import rank_cities, format_results_for_claude

# ─── Config ───────────────────────────────────────────────────────────────────

load_dotenv()

st.set_page_config(
    page_title="Season Radar",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="expanded",
)

CURRENT_MONTH = datetime.now().month
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

EXAMPLE_QUERIES = [
    "Where is spring right now?",
    "Best beach destinations in April with low crowds",
    "I want warm, dry, shoulder season -- not Europe",
    "I'm in Dubai, where should I go next month?",
    "Cool mountain destinations in August under 20C",
    "Top off-season city breaks for October",
]

# ─── Data & Client (cached) ───────────────────────────────────────────────────

@st.cache_resource
def load_cities():
    data_path = os.path.join(os.path.dirname(__file__), "data", "cities.json")
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)["cities"]


@st.cache_resource
def get_client():
    # Resolve API key: Streamlit secrets first, then env
    api_key = (
        st.secrets.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not api_key:
        st.error(
            "**ANTHROPIC_API_KEY not found.**\n\n"
            "- **Local dev**: add it to `.streamlit/secrets.toml`\n"
            "- **Streamlit Cloud**: add it under App Settings → Secrets"
        )
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


CITIES = load_cities()
client = get_client()

# ─── System Prompt & Tools ────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are Season Radar, a seasonal travel decision engine. Today is {datetime.now().strftime('%B %Y')}.

Your role is to help flexible travellers (digital nomads, remote workers, slow travellers) choose destinations based on weather, seasonality, and crowd levels — NOT bookings, flights, or prices.

You have access to structured climate data for {len(CITIES)} global cities. When a user asks a travel timing question, you MUST call the `search_destinations` tool to retrieve ranked data, then explain the results conversationally.

Behaviour rules:
- ALWAYS call the tool for any travel destination query — never answer from memory alone
- Cite the actual temperatures and crowd status from the tool result
- Be concise: present 3–5 cities by default unless more are asked for
- When the query is ambiguous (no month, no preferences), ask ONE focused clarifying question
- If the user mentions where they currently are, use exclude_regions to filter it out
- Focus on timing and seasonality insight — explain WHY each destination suits the criteria
- Mention shoulder/off-season context as a practical benefit
- Do not discuss bookings, prices, hotels, or flights

Response format (use this structure):
**[City, Country]** — X°C avg, [season status]
Brief 1–2 sentence reason why it fits.

Then a short closing note about timing or alternatives."""

TOOLS = [
    {
        "name": "search_destinations",
        "description": (
            "Search and rank global destinations from the climate dataset based on "
            "the user's travel timing preferences. Always call this tool before "
            "recommending destinations — never rely on general knowledge alone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "travel_month": {
                    "type": "integer",
                    "description": (
                        f"Month of intended travel (1-12). "
                        f"Default to current month ({CURRENT_MONTH}) if not specified. "
                        f"If user says 'next month', use {(CURRENT_MONTH % 12) + 1}."
                    ),
                    "minimum": 1,
                    "maximum": 12,
                },
                "temp_min": {
                    "type": "number",
                    "description": (
                        "Minimum preferred temperature in Celsius. "
                        "Infer: 'hot'->28, 'warm'->22, 'mild'->15, 'cool'->8, 'cold'->0. "
                        "Omit if no preference."
                    ),
                },
                "temp_max": {
                    "type": "number",
                    "description": (
                        "Maximum preferred temperature in Celsius. "
                        "Infer: 'hot'->38, 'warm'->30, 'mild'->24, 'cool'->18, 'cold'->12. "
                        "Omit if no preference."
                    ),
                },
                "rain_tolerance": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "low=dry preferred, medium=moderate rain ok, high=rain not a concern.",
                },
                "crowd_preference": {
                    "type": "string",
                    "enum": ["off_peak", "shoulder", "any"],
                    "description": "off_peak=avoid crowds, shoulder=moderate ok, any=no preference.",
                },
                "environment_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Preferred types: beach, city, mountain, island, tropical, "
                        "cultural, ski, nature, desert, coastal, history, food, adventure, "
                        "diving, romantic. Omit for no preference."
                    ),
                },
                "exclude_regions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Regions/countries to exclude, e.g. ['Europe'] or ['UAE'].",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10).",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["travel_month", "crowd_preference"],
        },
    }
]

# ─── Tool Execution ───────────────────────────────────────────────────────────

def execute_search(tool_input: dict) -> str:
    month = tool_input.get("travel_month", CURRENT_MONTH)
    month_name = MONTH_NAMES[month - 1]
    ranked = rank_cities(CITIES, tool_input)
    return format_results_for_claude(ranked, month_name)


def process_tool_call(name: str, tool_input: dict) -> str:
    if name == "search_destinations":
        return execute_search(tool_input)
    return f"[Unknown tool: {name}]"


# ─── Agentic Loop ─────────────────────────────────────────────────────────────

def run_agentic_turn(conversation: list) -> str:
    """Drive one full agentic turn. Returns final text response."""
    MAX_ITERATIONS = 6

    for _ in range(MAX_ITERATIONS):
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=2048,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=conversation,
        ) as stream:
            response = stream.get_final_message()

        conversation.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return "".join(
                block.text
                for block in response.content
                if hasattr(block, "text") and block.type == "text"
            )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = process_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            conversation.append({"role": "user", "content": tool_results})

    return "Sorry, I hit an internal limit. Please try rephrasing your question."


# ─── Session State Init ───────────────────────────────────────────────────────

if "conversation" not in st.session_state:
    st.session_state.conversation = []   # full Anthropic message history
if "messages" not in st.session_state:
    st.session_state.messages = []       # display-only {role, content: str}
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌍 Season Radar")
    st.markdown(
        f"Seasonal travel intelligence for **{datetime.now().strftime('%B %Y')}**.\n\n"
        "Find destinations that match your ideal weather, crowd level, and vibe."
    )

    st.divider()
    st.markdown("**Try an example:**")

    for example in EXAMPLE_QUERIES:
        if st.button(example, use_container_width=True, key=f"ex_{example[:20]}"):
            st.session_state.pending_query = example

    st.divider()
    st.markdown(
        "<small>Built on real climate normals for "
        f"{len(CITIES)} global destinations. "
        "Powered by Claude.</small>",
        unsafe_allow_html=True,
    )

    if st.button("Clear conversation", use_container_width=True, type="secondary"):
        st.session_state.conversation = []
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

# ─── Main Chat UI ─────────────────────────────────────────────────────────────

st.markdown("## Season Radar")
st.caption("Tell me when you want to travel and what conditions you prefer.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Resolve the next prompt (typed or clicked example)
prompt = st.chat_input("Where should I go? Try: 'mild weather in May, low crowds'")
if st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None

# Process the prompt
if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add to Anthropic conversation history
    st.session_state.conversation.append({"role": "user", "content": prompt})

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching seasonal data..."):
            try:
                reply = run_agentic_turn(st.session_state.conversation)
            except anthropic.APIStatusError as e:
                reply = f"API error ({e.status_code}): {e.message}"
            except anthropic.APIConnectionError:
                reply = "Connection error — check your internet connection and try again."

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Rerun to clear the pending_query state cleanly
    st.rerun()
