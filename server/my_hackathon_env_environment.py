"""
Support Triage Environment Implementation.

A multi-step, business-aware customer support RL environment.
The agent must resolve tickets through a structured action space.
Rewards balance correctness, cost, and efficiency.
"""

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SupportAction, SupportObservation
except ImportError:
    from models import SupportAction, SupportObservation

# Scenario definitions
SCENARIOS: List[Dict[str, Any]] = [
    # EASY — All information is present in the ticket. One-step close wins.
    {
        "task_id": "easy_order_status",
        "difficulty": "easy",
        "ticket_text": (
            "Hi, I placed order #ORD-1023 three days ago and it still hasn't "
            "arrived. Can you tell me what's happening? My email is "
            "alice@example.com."
        ),
        "customer_tier": "standard",
        "order_value": 34.99,
        # Hidden data the agent can extract via extract_info
        "hidden_data": {
            "order_id": "ORD-1023",
            "order_status": "In transit — expected delivery tomorrow.",
            "customer_email": "alice@example.com",
            "refund_eligible": False,
        },
        # Optimal terminal action and how to grade it
        "optimal_action": "close_ticket",
        "required_keywords": ["in transit", "tomorrow"],
        "max_steps": 4,
        "step_penalty": -0.05,
        "unnecessary_refund_penalty": -0.8,
        "unnecessary_escalation_penalty": -0.5,
        "base_resolution_reward": 1.0,
    },
    # MEDIUM — Ambiguous complaint; agent must ask clarification, then close.
    {
        "task_id": "medium_damaged_product",
        "difficulty": "medium",
        "ticket_text": (
            "My package arrived today but something is wrong with it. "
            "I'm really unhappy and want this fixed ASAP."
        ),
        "customer_tier": "silver",
        "order_value": 89.50,
        "hidden_data": {
            "order_id": "ORD-2047",
            "order_status": "Delivered",
            "damage_confirmed": True,
            "issue_type": "damaged_product",
            "refund_eligible": True,
            "customer_email": "bob@example.com",
        },
        # The agent should clarify, then extract, then either refund or close
        "optimal_action": "issue_refund",
        "required_clarification": True,   
        "required_extract": True,          
        "max_steps": 6,
        "step_penalty": -0.08,
        "unnecessary_escalation_penalty": -0.4,
        "skip_clarification_penalty": -0.3,
        "base_resolution_reward": 1.0,
    },
    # HARD — High-value refund, missing proof, gold-tier customer.
    # Agent must extract data, validate eligibility, then decide.
    {
        "task_id": "hard_refund_conflict",
        "difficulty": "hard",
        "ticket_text": (
            "I demand a full refund for my order. The product was completely "
            "defective. I've been a loyal customer for years and this is "
            "unacceptable. I expect this resolved immediately."
        ),
        "customer_tier": "gold",
        "order_value": 349.00,
        "hidden_data": {
            "order_id": "ORD-5501",
            "order_status": "Delivered — 6 days ago",
            "defect_confirmed": True,
            "within_return_window": True,   
            "refund_eligible": True,
            "partial_use_detected": False,  
            "customer_email": "carol@example.com",
        },
        "optimal_action": "issue_refund",
        "required_extracts": ["defect_confirmed", "within_return_window"],
        "max_steps": 8,
        "step_penalty": -0.06,
        "unvalidated_refund_penalty": -0.9,
        "unnecessary_escalation_penalty": -0.3,
        "base_resolution_reward": 1.0,
    },
]

# Bounds used to convert cumulative trajectory reward to an open-interval score.
TASK_SCORE_BOUNDS: Dict[str, tuple[float, float]] = {
    "easy_order_status": (-1.5, 1.0),
    "medium_damaged_product": (-1.9, 1.0),
    "hard_refund_conflict": (-2.0, 1.0),
}


class SupportTriageEnvironment(Environment):
    """
    Multi-step customer support RL environment.

    The agent receives a ticket and customer context, then iteratively
    takes actions until the ticket is closed or escalated.
    Business costs are applied at each step to reward efficient resolution.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario: Dict[str, Any] = random.choice(SCENARIOS)
        self._history: List[Dict[str, Any]] = []
        self._cumulative_reward: float = 0.0
        self._did_clarify: bool = False
        self._extracted_fields: List[str] = []
        self._done: bool = False

    def reset(self) -> SupportObservation:
        """Start a new episode with a randomly chosen scenario."""
        self._scenario = random.choice(SCENARIOS)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._history = []
        self._cumulative_reward = 0.0
        self._did_clarify = False
        self._extracted_fields = []
        self._done = False

        return SupportObservation(
            ticket_text=self._scenario["ticket_text"],
            customer_tier=self._scenario["customer_tier"],
            order_value=self._scenario["order_value"],
            action_feedback=(
                "New ticket received. Analyze the ticket and decide your first action. "
                f"Available actions: extract_info, ask_clarification, issue_refund, "
                f"escalate_to_human, close_ticket."
            ),
            history=[],
            step_count=0,
            max_steps=self._scenario["max_steps"],
            task_id=self._scenario["task_id"],
            is_terminated=False,
            cumulative_reward=0.0,
            done=False,
            reward=0.0,
        )

    def step(self, action: SupportAction) -> SupportObservation:
        """Execute one agent action and return updated observation + reward."""
        if self._done:
            return self._terminal_obs("Episode already ended.", 0.0)

        self._state.step_count += 1
        sc = self._scenario
        step_reward = sc["step_penalty"] 

        action_type = action.action_type
        argument = action.argument.strip()
        feedback = ""
        terminal = False

        # Force-terminate if max steps exceeded
        if self._state.step_count > sc["max_steps"]:
            self._done = True
            self._cumulative_reward += -0.5 
            return self._build_obs(
                feedback="Maximum steps exceeded. Episode terminated with penalty.",
                step_reward=-0.5,
                done=True,
            )

        # Handle each action type
        if action_type == "extract_info":
            field = argument.lower().replace(" ", "_")
            hidden = sc["hidden_data"]
            if field in hidden:
                value = hidden[field]
                feedback = f"[INFO] {field} = {value}"
                self._extracted_fields.append(field)
            else:
                feedback = f"[INFO] Field '{field}' not found in order data."
                step_reward += -0.05 

        elif action_type == "ask_clarification":
            self._did_clarify = True
            feedback = (
                f"[CUSTOMER] '{argument}' — "
                + self._simulate_customer_reply(argument)
            )

        elif action_type == "issue_refund":
            refund_reward, feedback, terminal = self._handle_refund(argument)
            step_reward += refund_reward

        elif action_type == "escalate_to_human":
            escalation_reward, feedback, terminal = self._handle_escalation(argument)
            step_reward += escalation_reward

        elif action_type == "close_ticket":
            close_reward, feedback, terminal = self._handle_close(argument)
            step_reward += close_reward

        # Record history and accumulate reward
        self._history.append({
            "step": self._state.step_count,
            "action_type": action_type,
            "argument": argument,
            "feedback": feedback,
            "step_reward": round(step_reward, 3),
        })
        self._cumulative_reward += step_reward
        self._cumulative_reward = round(self._cumulative_reward, 3)

        if terminal:
            self._done = True

        return self._build_obs(feedback=feedback, step_reward=step_reward, done=terminal)

    @property
    def state(self) -> State:
        return self._state

    def _strict_open_unit(self, value: float) -> float:
        """Clamp to strict open interval (0, 1)."""
        eps = 1e-3
        return max(eps, min(1.0 - eps, float(value)))

    def _normalize_task_score(self, raw_reward: float, task_id: str) -> float:
        """Map cumulative reward to strict (0, 1) for terminal grading."""
        min_reward, max_reward = TASK_SCORE_BOUNDS.get(task_id, (-2.0, 1.0))
        span = max_reward - min_reward
        if span <= 0:
            return 0.5
        normalized = (raw_reward - min_reward) / span
        return self._strict_open_unit(normalized)

    def _final_task_score(self) -> float:
        """Compute terminal score from current cumulative reward."""
        task_id = self._scenario.get("task_id", "")
        return self._normalize_task_score(self._cumulative_reward, task_id)

    def _simulate_customer_reply(self, question: str) -> str:
        """Return a canned clarification reply based on the scenario."""
        sc = self._scenario
        task = sc["task_id"]
        q = question.lower()

        if task == "medium_damaged_product":
            if any(w in q for w in ["order", "number", "id"]):
                return "It's order number ORD-2047."
            if any(w in q for w in ["damage", "wrong", "issue", "problem"]):
                return "The screen is cracked — it arrived that way."
            return "I'm not sure what you mean. Please look into it."

        if task == "hard_refund_conflict":
            if any(w in q for w in ["order", "number", "id"]):
                return "Order ORD-5501."
            if any(w in q for w in ["defect", "issue", "problem"]):
                return "The motor stopped working after one use."
            return "Just fix it, please."

        # Easy scenario — shouldn't need clarification
        return "I already gave all the info in my original message."

    def _handle_refund(self, amount_str: str):
        """Grade a refund action."""
        sc = self._scenario
        hidden = sc["hidden_data"]
        reward = 0.0
        terminal = True

        try:
            amount = float(amount_str)
        except ValueError:
            return -0.4, "[ERROR] Invalid refund amount. Must be a number.", True

        if not hidden.get("refund_eligible", False):
            # Refund on non-eligible order: big penalty
            feedback = (
                f"[REFUND DENIED] Order is not eligible for a refund. "
                f"Issuing ${amount:.2f} incorrectly costs the business."
            )
            reward = sc.get("unnecessary_refund_penalty", -0.8)
            return reward, feedback, terminal

        # Refund is eligible — check if agent validated required fields first
        required_extracts = sc.get("required_extracts", [])
        missing = [f for f in required_extracts if f not in self._extracted_fields]
        if missing:
            # Refunded without verifying key fields
            reward = sc.get("unvalidated_refund_penalty", -0.5)
            feedback = (
                f"[REFUND ISSUED — UNVALIDATED] ${amount:.2f} refunded but "
                f"agent did not verify: {missing}. Partial score applies."
            )
            return reward, feedback, terminal

        # Correct, validated refund
        reward = sc["base_resolution_reward"]
        feedback = (
            f"[REFUND ISSUED] ${amount:.2f} successfully refunded to customer. "
            f"Ticket resolved."
        )
        return reward, feedback, terminal

    def _handle_escalation(self, reason: str):
        """Grade an escalation action."""
        sc = self._scenario
        reward = sc.get("unnecessary_escalation_penalty", -0.4)
        # Escalation is always suboptimal in our scenarios
        feedback = (
            f"[ESCALATED] Ticket handed to human agent (reason: {reason}). "
            f"This was avoidable — suboptimal resolution."
        )
        return reward, feedback, True

    def _handle_close(self, response: str):
        """Grade a close_ticket action."""
        sc = self._scenario
        reward = 0.0
        response_lower = response.lower()

        optimal = sc.get("optimal_action", "close_ticket")

        if optimal == "close_ticket":
            keywords = sc.get("required_keywords", [])
            matched = [kw for kw in keywords if kw.lower() in response_lower]
            if len(matched) == len(keywords):
                reward = sc["base_resolution_reward"]
                feedback = f"[CLOSED ✓] Correct resolution. Ticket closed successfully."
            elif matched:
                reward = sc["base_resolution_reward"] * 0.5
                feedback = (
                    f"[CLOSED ~] Partial resolution. "
                    f"Matched {len(matched)}/{len(keywords)} expected keywords."
                )
            else:
                reward = 0.1
                feedback = (
                    "[CLOSED ✗] Ticket closed but response did not address "
                    "the customer's issue adequately."
                )

            # Penalty if this scenario required clarification first
            if sc.get("required_clarification") and not self._did_clarify:
                reward += sc.get("skip_clarification_penalty", -0.3)
                feedback += " (Penalty: clarification was required but skipped.)"

        else:
            # Agent chose close_ticket but optimal was refund
            reward = 0.2
            feedback = (
                "[CLOSED — WRONG ACTION] The customer's issue required a different "
                "resolution (e.g., refund). Partial credit only."
            )

        return reward, feedback, True

    def _build_obs(self, feedback: str, step_reward: float, done: bool) -> SupportObservation:
        sc = self._scenario
        reported_reward = self._final_task_score() if done else round(step_reward, 3)
        return SupportObservation(
            ticket_text=sc["ticket_text"],
            customer_tier=sc["customer_tier"],
            order_value=sc["order_value"],
            action_feedback=feedback,
            history=list(self._history),
            step_count=self._state.step_count,
            max_steps=sc["max_steps"],
            task_id=sc["task_id"],
            is_terminated=done,
            cumulative_reward=self._cumulative_reward,
            done=done,
            reward=reported_reward,
        )

    def _terminal_obs(self, feedback: str, reward: float) -> SupportObservation:
        sc = self._scenario or {}
        return SupportObservation(
            ticket_text=sc.get("ticket_text", ""),
            customer_tier=sc.get("customer_tier", "standard"),
            order_value=sc.get("order_value", 0.0),
            action_feedback=feedback,
            history=list(self._history),
            step_count=self._state.step_count,
            max_steps=sc.get("max_steps", 10),
            task_id=sc.get("task_id", ""),
            is_terminated=True,
            cumulative_reward=self._cumulative_reward,
            done=True,
            reward=self._final_task_score(),
        )
