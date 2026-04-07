"""
Inference script for the Support Triage Environment.

Required by the hackathon. Uses an LLM agent (via OpenAI-compatible client)
to autonomously resolve support tickets through the multi-step environment.

Emits strictly formatted [START], [STEP], and [END] logs as required.

Usage:
    python inference.py

Environment variables (see .env):
    API_BASE_URL   - LLM API base URL (OpenAI-compatible)
    MODEL_NAME     - Model identifier
    LLM_API_KEY    - API key (or use HF_TOKEN as fallback)
    HF_TOKEN       - Hugging Face token (used as api_key if LLM_API_KEY not set)
    MAX_STEPS      - Max steps per episode (default 10)
    NUM_TASKS      - Number of tasks to evaluate (default 3)
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load .env file if present
load_dotenv()

# Config

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("HF_TOKEN", "")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "10"))
NUM_TASKS = int(os.environ.get("NUM_TASKS", "3"))
SERVER_URL = f"http://{os.environ.get('HOST', 'localhost')}:{os.environ.get('PORT', '8000')}"

VALID_ACTIONS = [
    "extract_info",
    "ask_clarification",
    "issue_refund",
    "escalate_to_human",
    "close_ticket",
]

# Approximate reward envelopes for each scenario in the environment.
TASK_REWARD_BOUNDS: Dict[str, tuple[float, float]] = {
    "easy_order_status": (-1.5, 1.0),
    "medium_damaged_product": (-1.9, 1.0),
    "hard_refund_conflict": (-2.0, 1.0),
}

SYSTEM_PROMPT = """You are an expert AI customer support agent.

You are operating in a multi-step RL environment. At each turn you receive:
•⁠  ⁠The customer's original ticket text
•⁠  ⁠Customer metadata (tier, order value)
•⁠  ⁠Feedback from your last action
•⁠  ⁠A history of all your previous actions and their outcomes

You must choose ONE action from the following list and provide an argument:

1.⁠ ⁠extract_info     — Look up a specific hidden field (e.g. "order_status", "refund_eligible", "defect_confirmed")
2.⁠ ⁠ask_clarification — Ask the customer a follow-up question
3.⁠ ⁠issue_refund      — Issue a refund. Argument must be a numeric dollar amount (e.g. "89.50")
4.⁠ ⁠escalate_to_human — Only use as a last resort with a clear reason
5.⁠ ⁠close_ticket      — Provide a final response to the customer

IMPORTANT RULES:
•⁠  ⁠Always prefer the most efficient resolution with fewest steps.
•⁠  ⁠For damaged or defective items, verify eligibility before issuing refunds.
•⁠  ⁠For high-value orders, extract relevant data fields first.
•⁠  ⁠close_ticket should include a complete, helpful response to the customer.

Respond ONLY with a valid JSON object like this:
{"action_type": "extract_info", "argument": "order_status"}

Do not include any other text, explanation, or markdown.
"""

def normalize_score(raw_reward: float, task_id: str) -> float:
    """Normalize raw cumulative reward to a strict (0, 1) score."""
    default_bounds = (-2.0, 1.0)
    min_reward, max_reward = TASK_REWARD_BOUNDS.get(task_id, default_bounds)
    span = max_reward - min_reward
    if span <= 0:
        normalized = 0.5
    else:
        normalized = (raw_reward - min_reward) / span

    eps = 1e-4
    return round(max(eps, min(1.0 - eps, normalized)), 4)


def build_user_message(obs: Dict[str, Any]) -> str:
    """Build the current state description for the LLM."""
    history_text = ""
    if obs.get("history"):
        lines = []
        for h in obs["history"]:
            lines.append(
                f"  Step {h['step']}: [{h['action_type']}] '{h['argument']}' "
                f"→ {h['feedback']} (reward: {h.get('step_reward', '?')})"
            )
        history_text = "\nAction History:\n" + "\n".join(lines)

    return f"""=== Current Support Ticket ===
{obs.get('ticket_text', '')}

Customer Tier: {obs.get('customer_tier', 'standard')}
Order Value: ${obs.get('order_value', 0):.2f}
Task ID: {obs.get('task_id', '')}
Steps Taken: {obs.get('step_count', 0)} / {obs.get('max_steps', 10)}
Cumulative Reward So Far: {obs.get('cumulative_reward', 0.0):.3f}

Last Action Feedback: {obs.get('action_feedback', 'None')}
{history_text}

What is your next action? Respond with valid JSON only."""


async def call_llm(client: AsyncOpenAI, messages: List[Dict]) -> Dict[str, str]:
    """Call the LLM and parse the JSON action response."""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "256")),
    )
    raw = response.choices[0].message.content.strip()

    if raw.startswith("⁠ "):
        raw = raw.split(" ⁠")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        action = json.loads(raw)
        if action.get("action_type") not in VALID_ACTIONS:
            raise ValueError(f"Invalid action_type: {action.get('action_type')}")
        return action
    except Exception as e:
        print(f"  [WARN] Could not parse LLM response: {e}. Defaulting to close_ticket.", file=sys.stderr)
        return {"action_type": "close_ticket", "argument": "I'm sorry, let me escalate this for you."}


async def run_episode(episode_num: int, llm_client: AsyncOpenAI) -> Dict[str, Any]:
    """Run a single episode against the environment via HTTP."""
    import httpx

    async with httpx.AsyncClient(base_url=SERVER_URL, timeout=30.0) as http:

        reset_resp = await http.post("/reset")
        reset_resp.raise_for_status()
        obs = reset_resp.json().get("observation", reset_resp.json())

        task_id = obs.get("task_id", f"episode_{episode_num}")
        total_reward = 0.0
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        print("[START] " + json.dumps({
            "event": "START",
            "episode": episode_num,
            "task_id": task_id,
            "ticket": obs.get("ticket_text", "")[:120] + "...",
            "customer_tier": obs.get("customer_tier"),
            "order_value": obs.get("order_value"),
        }))
        sys.stdout.flush()

        done = obs.get("is_terminated", False) or obs.get("done", False)
        step = 0

        while not done and step < MAX_STEPS:
            step += 1

            user_msg = build_user_message(obs)
            messages.append({"role": "user", "content": user_msg})

            action = await call_llm(llm_client, messages)
            messages.append({"role": "assistant", "content": json.dumps(action)})

            step_resp = await http.post("/step", json={"action": action})
            step_resp.raise_for_status()
            step_data = step_resp.json()

            reward = step_data.get("reward", 0.0)
            total_reward += reward
            obs = step_data.get("observation", step_data)
            done = step_data.get("done", False) or obs.get("is_terminated", False)

            print("[STEP] " + json.dumps({
                "event": "STEP",
                "episode": episode_num,
                "step": step,
                "action_type": action.get("action_type"),
                "argument": action.get("argument", "")[:80],
                "feedback": obs.get("action_feedback", "")[:120],
                "step_reward": round(reward, 3),
                "cumulative_reward": round(obs.get("cumulative_reward", total_reward), 3),
                "done": done,
            }))
            sys.stdout.flush()

        raw_score = obs.get("cumulative_reward", total_reward)
        final_score = normalize_score(raw_score, task_id)

        print("[END] " + json.dumps({
            "event": "END",
            "episode": episode_num,
            "task_id": task_id,
            "total_steps": step,
            "total_reward": round(obs.get("cumulative_reward", total_reward), 4),
            "final_score": final_score,
        }))
        sys.stdout.flush()

        return {
            "task_id": task_id,
            "steps": step,
            "total_reward": round(obs.get("cumulative_reward", total_reward), 4),
            "final_score": final_score,
        }


async def main():
    if not API_KEY:
        print("[ERROR] No API key found. Set LLM_API_KEY or HF_TOKEN in your .env file.", file=sys.stderr)
        sys.exit(1)

    llm_client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[INFO] Using model: {MODEL_NAME} @ {API_BASE_URL}", file=sys.stderr)
    print(f"[INFO] Running {NUM_TASKS} episodes against {SERVER_URL}", file=sys.stderr)

    results = []
    for i in range(1, NUM_TASKS + 1):
        try:
            result = await run_episode(episode_num=i, llm_client=llm_client)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Episode {i} failed: {e}", file=sys.stderr)
            fallback_task_id = f"episode_{i}"
            results.append({
                "task_id": fallback_task_id,
                "steps": 0,
                "total_reward": 0.0,
                "final_score": normalize_score(0.0, fallback_task_id),
            })

    avg_score = round(sum(r["final_score"] for r in results) / len(results), 4) if results else 0.0
    print(f"\n[SUMMARY] Average final score across {NUM_TASKS} episodes: {avg_score}", file=sys.stderr)
    for r in results:
        print(f"  {r['task_id']}: steps={r['steps']}, score={r['final_score']}", file=sys.stderr)


if _name_ == "_main_":
    asyncio.run(main())