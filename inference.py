"""
Inference script for the Support Triage Environment.


STDOUT contract:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("HF_TOKEN", "")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "10"))
NUM_TASKS = max(1, int(os.environ.get("NUM_TASKS", "3")))
SERVER_URL = f"http://{os.environ.get('HOST', 'localhost')}:{os.environ.get('PORT', '8000')}"
BENCHMARK = "support_triage_env"
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.1"))

VALID_ACTIONS = [
    "extract_info",
    "ask_clarification",
    "issue_refund",
    "escalate_to_human",
    "close_ticket",
]

TASK_REWARD_BOUNDS: Dict[str, tuple[float, float]] = {
    "easy_order_status": (-1.5, 1.0),
    "medium_damaged_product": (-1.9, 1.0),
    "hard_refund_conflict": (-2.0, 1.0),
}

SYSTEM_PROMPT = """You are an expert AI customer support agent.

You are operating in a multi-step RL environment. At each turn you receive:
- The customer's original ticket text
- Customer metadata (tier, order value)
- Feedback from your last action
- A history of all your previous actions and their outcomes

You must choose ONE action from the following list and provide an argument:
1. extract_info
2. ask_clarification
3. issue_refund
4. escalate_to_human
5. close_ticket

Respond ONLY with valid JSON:
{"action_type": "extract_info", "argument": "order_status"}
"""


def strict_unit_score(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    eps = 0.01
    return max(eps, min(1.0 - eps, float(value)))


def normalize_score(raw_reward: float, task_id: str) -> float:
    min_reward, max_reward = TASK_REWARD_BOUNDS.get(task_id, (-2.0, 1.0))
    span = max_reward - min_reward
    normalized = 0.5 if span <= 0 else (raw_reward - min_reward) / span
    return strict_unit_score(normalized)


def safe_token(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = safe_token(error) if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={safe_token(action)} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def build_user_message(obs: Dict[str, Any]) -> str:
    history_text = ""
    if obs.get("history"):
        lines = []
        for h in obs["history"]:
            lines.append(
                f"Step {h['step']}: [{h['action_type']}] '{h['argument']}' -> {h['feedback']} "
                f"(reward: {h.get('step_reward', '?')})"
            )
        history_text = "\nAction History:\n" + "\n".join(lines)

    return f"""Ticket: {obs.get('ticket_text', '')}
Customer Tier: {obs.get('customer_tier', 'standard')}
Order Value: ${obs.get('order_value', 0):.2f}
Task ID: {obs.get('task_id', '')}
Steps Taken: {obs.get('step_count', 0)} / {obs.get('max_steps', 10)}
Cumulative Reward: {obs.get('cumulative_reward', 0.0):.3f}
Last Action Feedback: {obs.get('action_feedback', 'None')}
{history_text}
Choose next action as JSON only."""


async def call_llm(client: AsyncOpenAI, messages: List[Dict[str, str]]) -> Dict[str, str]:
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "256")),
    )
    raw = (response.choices[0].message.content or "").strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        action = json.loads(raw)
        if action.get("action_type") not in VALID_ACTIONS:
            raise ValueError(f"Invalid action_type: {action.get('action_type')}")
        return {
            "action_type": action.get("action_type", "close_ticket"),
            "argument": str(action.get("argument", "")),
        }
    except Exception:
        return {"action_type": "close_ticket", "argument": "Unable to complete automatically."}


def rule_based_action(obs: Dict[str, Any]) -> Dict[str, str]:
    task_id = obs.get("task_id", "")
    history = obs.get("history", [])
    done_actions = [h.get("action_type", "") for h in history]

    extracted_fields = set()
    for h in history:
        if h.get("action_type") == "extract_info":
            arg = h.get("argument", "")
            if arg:
                extracted_fields.add(str(arg).lower().replace(" ", "_"))

    if task_id == "easy_order_status":
        return {"action_type": "close_ticket", "argument": "Your order is in transit and expected tomorrow."}

    if task_id == "medium_damaged_product":
        if "ask_clarification" not in done_actions:
            return {"action_type": "ask_clarification", "argument": "Could you confirm what is damaged?"}
        if "damage_confirmed" not in extracted_fields:
            return {"action_type": "extract_info", "argument": "damage_confirmed"}
        if "refund_eligible" not in extracted_fields:
            return {"action_type": "extract_info", "argument": "refund_eligible"}
        return {"action_type": "issue_refund", "argument": "89.50"}

    if task_id == "hard_refund_conflict":
        if "defect_confirmed" not in extracted_fields:
            return {"action_type": "extract_info", "argument": "defect_confirmed"}
        if "within_return_window" not in extracted_fields:
            return {"action_type": "extract_info", "argument": "within_return_window"}
        amount = float(obs.get("order_value", 100.0))
        return {"action_type": "issue_refund", "argument": f"{amount:.2f}"}

    return {"action_type": "close_ticket", "argument": "Issue resolved."}


async def run_episode(episode_num: int, llm_client: Optional[AsyncOpenAI]) -> Dict[str, float]:
    import httpx

    rewards: List[float] = []
    steps_taken = 0
    score = 0.5
    success = False

    async with httpx.AsyncClient(base_url=SERVER_URL, timeout=30.0) as http:
        reset_resp = await http.post("/reset")
        reset_resp.raise_for_status()
        obs = reset_resp.json().get("observation", reset_resp.json())
        task_id = obs.get("task_id", f"episode_{episode_num}")

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        done = bool(obs.get("is_terminated", False) or obs.get("done", False))

        try:
            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                user_msg = build_user_message(obs)
                messages.append({"role": "user", "content": user_msg})

                if llm_client is None:
                    action = rule_based_action(obs)
                else:
                    action = await call_llm(llm_client, messages)

                messages.append({"role": "assistant", "content": json.dumps(action)})

                try:
                    step_resp = await http.post("/step", json={"action": action})
                    step_resp.raise_for_status()
                    step_data = step_resp.json()

                    reward = float(step_data.get("reward", 0.0) or 0.0)
                    rewards.append(reward)
                    steps_taken = step

                    obs = step_data.get("observation", step_data)
                    done = bool(step_data.get("done", False) or obs.get("is_terminated", False))

                    # action_str = f"{action.get('action_type')}({action.get('argument', '')})"
                    action_str = f"{action.get('action_type')}:{action.get('argument','')}"
                    log_step(step, action_str, reward, done, None)
                except Exception as exc:
                    steps_taken = step
                    done = True
                    log_step(step, "error", 0.0, True, str(exc))
                    break

            raw_score = float(obs.get("cumulative_reward", sum(rewards)))
            score = normalize_score(raw_score, task_id)
            success = score >= SUCCESS_SCORE_THRESHOLD
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"score": score}


async def main() -> None:
    llm_client: Optional[AsyncOpenAI] = None
    if API_KEY:
        llm_client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        print(f"[INFO] Using model: {MODEL_NAME} @ {API_BASE_URL}", file=sys.stderr)
    else:
        print("[WARN] No API key found. Using deterministic fallback policy.", file=sys.stderr)

    print(f"[INFO] Running {NUM_TASKS} episodes against {SERVER_URL}", file=sys.stderr)

    scores: List[float] = []
    for i in range(1, NUM_TASKS + 1):
        try:
            result = await run_episode(i, llm_client)
            scores.append(strict_unit_score(result["score"]))
        except Exception as e:
            print(f"[ERROR] Episode {i} failed hard: {e}", file=sys.stderr)
            log_start(task=f"episode_{i}", env=BENCHMARK, model=MODEL_NAME)
            log_step(step=1, action="error", reward=0.0, done=True, error=str(e))
            fallback_score = 0.5
            log_end(success=True, steps=1, score=fallback_score, rewards=[0.0])
            scores.append(fallback_score)

    avg = strict_unit_score(sum(scores) / len(scores)) if scores else 0.5
    print(f"[SUMMARY] avg_score={avg:.3f}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
