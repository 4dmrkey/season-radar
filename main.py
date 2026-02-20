"""
Season Radar — Seasonal Travel Decision Engine
A Claude-powered CLI that recommends destinations based on weather,
seasonality, and crowd levels.

Usage:
    python main.py
"""

import os
import json
import sys
from datetime import datetime

import anthropic
from dotenv import load_dotenv

from scoring import rank_cities, format_results_for_claude

# ─── Setup ────────────────────────────────────────────────────────────────────

load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    print("Error: ANTHROPIC_API_KEY is not set.")
    print("Copy .env.example to .env and add your key.")
    sys.exit(1)

client = anthropic.Anthropic(api_key=API_KEY)

# Load city dataset
_data_path = os.path.join(os.path.dirname(__file__), "data", "cities.json")
with open(_data_path, encoding="utf-8") as _f:
    CITIES = json.load(_f)["cities"]

CURRENT_MONTH = datetime.now().month
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# ─── System Prompt ────────────────────────────────────────────────────────────

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

# ─── Tool Definition ──────────────────────────────────────────────────────────

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
                        f"Month of intended travel (1–12). "
                        f"Default to current month ({CURRENT_MONTH}) if not specified. "
                        f"If user says 'next month', use {(CURRENT_MONTH % 12) + 1}."
                    ),
                    "minimum": 1,
                    "maximum": 12,
                },
                "temp_min": {
                    "type": "number",
                    "description": (
                        "Minimum preferred temperature in °C. "
                        "Infer from descriptors: 'hot'→28, 'warm'→22, 'mild'→15, "
                        "'cool'→10, 'cold'→0. Omit if no temperature preference."
                    ),
                },
                "temp_max": {
                    "type": "number",
                    "description": (
                        "Maximum preferred temperature in °C. "
                        "Infer from descriptors: 'hot'→38, 'warm'→30, 'mild'→24, "
                        "'cool'→18, 'cold'→12. Omit if no temperature preference."
                    ),
                },
                "rain_tolerance": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": (
                        "low = dry conditions preferred (< ~30mm/month); "
                        "medium = moderate rain OK; "
                        "high = rain is not a concern. "
                        "Default: medium."
                    ),
                },
                "crowd_preference": {
                    "type": "string",
                    "enum": ["off_peak", "shoulder", "any"],
                    "description": (
                        "off_peak = strongly avoid crowds/peak season; "
                        "shoulder = moderate tourist traffic acceptable; "
                        "any = no crowd preference."
                    ),
                },
                "environment_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Preferred environment types from: beach, city, mountain, "
                        "island, tropical, cultural, ski, nature, desert, coastal, "
                        "history, food, adventure, diving, romantic. "
                        "Omit for no preference."
                    ),
                },
                "exclude_regions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Regions or countries to exclude "
                        "(e.g. ['Europe'] if user is already there, "
                        "or ['UAE'] if user is in Dubai). "
                        "Omit if no exclusions needed."
                    ),
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of top destinations to return (default 5, max 10).",
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
    """Run the ranking engine and return formatted results."""
    month     = tool_input.get("travel_month", CURRENT_MONTH)
    month_name = MONTH_NAMES[month - 1]
    ranked    = rank_cities(CITIES, tool_input)
    return format_results_for_claude(ranked, month_name)


def process_tool_call(name: str, tool_input: dict) -> str:
    if name == "search_destinations":
        return execute_search(tool_input)
    return f"[Unknown tool: {name}]"


# ─── Agentic Conversation Loop ────────────────────────────────────────────────

def run_agentic_turn(conversation: list) -> str:
    """
    Drive one full agentic turn:
    1. Call Claude (may respond with tool_use)
    2. Execute any tools and feed results back
    3. Repeat until end_turn
    Returns the final text response.
    """
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

        # Append assistant turn to history
        conversation.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract and return text
            return "".join(
                block.text for block in response.content
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

            # Feed tool results back as a user turn
            conversation.append({"role": "user", "content": tool_results})
            continue  # loop back for Claude's final answer

    return "[Season Radar hit an iteration limit. Please try again.]"


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

BANNER = f"""
================================================================
         SEASON RADAR  --  Seasonal Travel Decision Engine
================================================================

Powered by real climate data for {len(CITIES)} global destinations.
Helping flexible travellers go somewhere worth going.

Try asking:
  - Where is spring right now?
  - Best beach destinations in April with low crowds
  - I want warm weather, dry, shoulder season -- not Europe
  - I'm in Dubai, where should I go next month to escape the heat?
  - Good places in October under 25C with little rain?

Type 'quit' or Ctrl+C to exit.
{"=" * 64}
"""


def main():
    print(BANNER)
    conversation = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! Safe travels.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q", "bye"):
            print("\nGoodbye! Safe travels.")
            break

        conversation.append({"role": "user", "content": user_input})

        print("\nSeason Radar: ", end="", flush=True)
        try:
            reply = run_agentic_turn(conversation)
        except anthropic.APIStatusError as e:
            reply = f"[API error {e.status_code}: {e.message}]"
        except anthropic.APIConnectionError:
            reply = "[Connection error — check your internet and try again.]"

        print(reply)
        print(f"\n{'─' * 62}\n")


if __name__ == "__main__":
    main()
