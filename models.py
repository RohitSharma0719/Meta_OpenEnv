"""
Data models for the Support Triage Environment.

The agent must resolve customer support tickets through a sequence of
business-aware actions. Rewards balance correctness, cost, and efficiency.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field

ActionType = Literal[
    "extract_info",       
    "ask_clarification",  
    "issue_refund",       
    "escalate_to_human",  
    "close_ticket",       
]


class SupportAction(Action):
    """
    Action submitted by the agent at each step.

    The agent picks an action_type and provides an argument:
      - extract_info:       argument = field name to look up (e.g. "order_status")
      - ask_clarification:  argument = the question to ask the customer
      - issue_refund:       argument = amount as a numeric string (e.g. "49.99")
      - escalate_to_human:  argument = reason string
      - close_ticket:       argument = the final response message to the customer
    """

    action_type: ActionType = Field(
        ...,
        description=(
            "The action to perform. One of: extract_info, ask_clarification, "
            "issue_refund, escalate_to_human, close_ticket."
        ),
    )
    argument: str = Field(
        ...,
        description=(
            "The argument for the action. For extract_info: a field name. "
            "For ask_clarification: a question string. "
            "For issue_refund: a numeric dollar amount. "
            "For escalate_to_human: a reason. "
            "For close_ticket: the final response message."
        ),
    )


class SupportObservation(Observation):
    """
    Observation returned by the environment after each step.

    Contains the current ticket context plus feedback from the last action.
    """

    ticket_text: str = Field(
        default="",
        description="The customer's support message or ticket text.",
    )
    customer_tier: str = Field(
        default="standard",
        description="Customer tier: standard, silver, or gold.",
    )
    order_value: float = Field(
        default=0.0,
        description="Value of the order in USD related to this ticket.",
    )

    action_feedback: str = Field(
        default="",
        description="Result or feedback from the last action the agent took.",
    )

    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of previous (action_type, argument, feedback) tuples.",
    )
    step_count: int = Field(
        default=0,
        description="Number of steps taken so far in this episode.",
    )
    max_steps: int = Field(
        default=10,
        description="Maximum steps allowed before the episode is force-terminated.",
    )
    task_id: str = Field(
        default="",
        description="Identifier of the current task scenario (easy/medium/hard).",
    )

    is_terminated: bool = Field(
        default=False,
        description="True when the episode has ended (ticket closed or escalated).",
    )

    cumulative_reward: float = Field(
        default=0.0,
        description="Total reward accumulated so far in the episode.",
    )
