"""Support Triage Environment Client."""

from typing import Any, Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SupportAction, SupportObservation


class SupportTriageEnv(
    EnvClient[SupportAction, SupportObservation, State]
):
    """
    Client for the Support Triage Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated session on the server.

    Example (async):
        async with SupportTriageEnv(base_url="http://localhost:8000") as client:
            obs = await client.reset()
            print(obs.ticket_text)

            result = await client.step(
                SupportAction(action_type="extract_info", argument="order_status")
            )
            print(result.observation.action_feedback)

    Example (sync):
        with SupportTriageEnv(base_url="http://localhost:8000").sync() as client:
            obs = client.reset()
            result = client.step(
                SupportAction(action_type="close_ticket", argument="Your order is on its way!")
            )
    """

    def _step_payload(self, action: SupportAction) -> Dict:
        """Convert SupportAction to JSON payload for the step WebSocket message."""
        return {
            "action_type": action.action_type,
            "argument": action.argument,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SupportObservation]:
        """Parse server response into StepResult[SupportObservation]."""
        obs_data = payload.get("observation", {})

        observation = SupportObservation(
            ticket_text=obs_data.get("ticket_text", ""),
            customer_tier=obs_data.get("customer_tier", "standard"),
            order_value=obs_data.get("order_value", 0.0),
            action_feedback=obs_data.get("action_feedback", ""),
            history=obs_data.get("history", []),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 10),
            task_id=obs_data.get("task_id", ""),
            is_terminated=obs_data.get("is_terminated", False),
            cumulative_reward=obs_data.get("cumulative_reward", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
