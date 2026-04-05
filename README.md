---
title: Support Triage Environment
emoji: 🎧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Support Triage Environment

A multi-step, business-aware Reinforcement Learning environment built for the Meta-PyTorch Hackathon via the OpenEnv framework.

---

## Description

Unlike standard classification tasks, this environment simulates a real-world customer support agent trajectory. The agent receives a customer ticket and must take a sequence of actions to resolve it.

This environment simulates an **enterprise-grade AI support agent** that operates under **partial observability**, interacts with hidden system data, and performs **multi-step decision-making**.

The agent is required to balance:
- **Resolution correctness** (understanding and solving the issue)
- **Operational cost** (refunds, escalations, unnecessary steps)
- **Efficiency** (minimizing steps and redundant actions)

This transforms the problem from a simple NLP task into a **decision-making system** aligned with real-world LLM agent workflows.

---

## Key Insight

This environment models real-world LLM agents that must act under uncertainty, validate information before taking actions, and optimize for both correctness and business cost — mirroring how AI systems operate in production environments.

---

### Difficulty Tiers (Task Scenarios)

1. **Easy:** Fully observable scenario where all required information is present. Tests direct resolution and efficiency.

2. **Medium:** Partially observable scenario with ambiguity. The agent must decide whether to ask clarification or extract hidden information before acting.

3. **Hard:** High-stakes, high-value scenario involving business risk. The agent must validate critical conditions (e.g., defect confirmation, return window) before issuing costly actions like refunds.

---

## Action Space

Each action represents a real operational decision with associated business cost and impact.

The agent can choose from 5 actions, passing a string `argument` for each:
- `extract_info`: Retrieves specific database fields (e.g. `order_status`, `refund_eligible`).
- `ask_clarification`: Asks the customer a follow-up question.
- `issue_refund`: Issues a monetary refund. Argument must be a numeric amount.
- `escalate_to_human`: Hands off the ticket to an L2 agent (incurs a penalty).
- `close_ticket`: Resolves the ticket with a final written response.

---

## Observation Space

The environment returns the following state to the agent on every step:
- `ticket_text` (str): The customer's message.
- `customer_tier` (str): Account status (e.g. standard, gold).
- `order_value` (float): Monetary value of the order (used for cost-aware decisions).
- `action_feedback` (str): Result of the previous action (e.g. database output or simulated customer reply).
- `history` (list): Running list of all actions, feedback, and step-level rewards.
- `step_count` (int): Current step number in the episode.
- `cumulative_reward` (float): Total reward accumulated so far.

---

## Environment Design Highlights

- **Partial Observability:** Critical information (e.g., refund eligibility, defect status) is hidden and must be explicitly retrieved.
- **Multi-Step Reasoning:** Agents must plan sequences of actions rather than produce a single output.
- **Cost-Aware Decisions:** Actions like refunds and escalations incur penalties, simulating real business trade-offs.
- **Validation Before Action:** Agents are penalized for acting (e.g., issuing refunds) without verifying required conditions.
- **Dynamic Feedback Loop:** Each action produces feedback that influences subsequent decisions.


---

## Evaluation Logic

The environment uses a reward-based grading system that evaluates not just outcomes, but the quality of decision-making.

The reward function incorporates:
- **Correctness:** Accurate understanding and resolution of the issue
- **Efficiency:** Penalizing unnecessary steps through step-level penalties
- **Business Cost:** Financial and operational penalties for actions like refunds and escalations
- **Validation Discipline:** Penalizing actions taken without verifying required conditions

This ensures that agents are rewarded for behaving like effective real-world support systems, not just producing correct outputs.

---

---

### 2. Run Locally

```bash
bash run.sh

## Local Setup & Usage

### Prerequisites
- Python 3.10+
- `uv` package manager (optional, but recommended)

---

### 1. Environment Variables

Create a `.env` file in the root folder based on `.env.example`:

```env
LLM_API_KEY=your_api_key
MODEL_NAME=llama-3.3-70b-versatile
API_BASE_URL=https://api.groq.com/openai/v1