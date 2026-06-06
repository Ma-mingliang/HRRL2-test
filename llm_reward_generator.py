"""LLM-driven reward function generator using Eureka-style prompts."""

import json
import os
import re
import traceback
from pathlib import Path
from typing import Optional

import httpx


class LLMRewardGenerator:
    """Generate reward function modifications using LLM + paper pool data."""

    def __init__(self, api_key: str, base_url: str, model: str = "mimo-v2.5-pro"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = 120

    def generate_modification(
        self,
        current_reward_code: str,
        method: dict,
        env_context: str,
        baseline_metrics: dict,
        history: list,
        error_feedback: Optional[str] = None,
    ) -> Optional[dict]:
        """Generate a reward modification using LLM.

        Args:
            current_reward_code: Current __calculate_reward method code.
            method: Method dict from method_pool with core_idea, reward_formula, etc.
            env_context: Environment class description (observation space, key attributes).
            baseline_metrics: Baseline training metrics for context.
            history: List of previous version results for iterative feedback.
            error_feedback: If syntax check failed, the error message.

        Returns:
            Dict with 'code', 'name', 'description', 'category' or None if failed.
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            current_reward_code, method, env_context, baseline_metrics, history, error_feedback
        )

        try:
            response = self._call_llm(system_prompt, user_prompt)
            return self._parse_response(response, method)
        except Exception as e:
            print(f"[LLM] Error: {e}")
            traceback.print_exc()
            return None

    def fix_syntax_error(
        self,
        broken_code: str,
        error_message: str,
        method: dict,
    ) -> Optional[dict]:
        """Ask LLM to fix syntax errors in generated code."""
        system_prompt = (
            "You are a Python expert. Fix the syntax error in the reward function code below. "
            "Return ONLY the corrected Python code in a ```python ... ``` block. "
            "Do not change the logic, only fix syntax issues."
        )
        user_prompt = (
            f"## Broken Code\n```python\n{broken_code}\n```\n\n"
            f"## Error Message\n{error_message}\n\n"
            "Fix the syntax error and return the corrected code."
        )

        try:
            response = self._call_llm(system_prompt, user_prompt)
            code = self._extract_code(response)
            if code:
                return {
                    "code": code,
                    "name": method.get("method_name", "Unknown"),
                    "description": f"Syntax-fixed: {method.get('core_idea', '')}",
                    "category": method.get("category", ""),
                    "method_id": method.get("method_id", ""),
                }
        except Exception as e:
            print(f"[LLM Fix] Error: {e}")
        return None

    def _build_system_prompt(self) -> str:
        return """You are a reward engineer for reinforcement learning. Your task is to write reward functions for a bicycle self-balancing control system.

## Key Rules
1. The reward function must be a method named `__calculate_reward(self, state_last, state, target_handle_angle=0.0)`.
2. Use `self.__observation_reduction(state)` to get raw state: [dis_angle, theta0, w0, v].
3. Available self attributes: `self.epoch_num` (episode count), `self.step_num` (current step), `self.max_step_num` (1000), `self.FAILURE_PENALTY` (-10.0), `self.EARLY_TERMINATION_PENALTY` (-20.0).
4. Use `math` module for math operations (already imported).
5. You MAY use `hasattr(self, ...)` and `self.xxx` to store persistent state across steps.
6. The function must return a single float `reward`.
7. Do NOT import any modules.
8. Do NOT modify any other methods or class attributes.
9. Keep the indentation consistent with class method (4 spaces indent).

## Output Format
Return the complete `__calculate_reward` method in a ```python ... ``` block."""

    def _build_user_prompt(
        self,
        current_reward_code: str,
        method: dict,
        env_context: str,
        baseline_metrics: dict,
        history: list,
        error_feedback: Optional[str],
    ) -> str:
        parts = []

        # Environment context
        parts.append(f"## Environment\n{env_context}")

        # Current reward function
        parts.append(f"## Current Reward Function\n```python\n{current_reward_code}\n```")

        # Method from paper pool
        core_idea = method.get("core_idea", "")
        reward_formula = method.get("reward_formula", "")
        impl_template = method.get("implementation_template", "")
        risks = method.get("risks", [])
        paper_md = method.get("_paper_md")

        method_section = f"""## Research Method to Apply
- **Core Idea**: {core_idea}
- **Reward Formula**: {reward_formula}
- **Implementation Template**: {impl_template}
- **Risks**: {', '.join(risks[:3]) if risks else 'N/A'}"""
        parts.append(method_section)

        # Paper content (if available)
        if paper_md:
            parts.append(f"## Reference Paper (excerpt)\n{paper_md[:6000]}")

        # Baseline metrics
        parts.append(f"## Baseline Metrics\n- Mean Reward: {baseline_metrics.get('mean_reward', 'N/A')}")

        # History (last 5 versions)
        if history:
            history_text = "## Recent Version History\n"
            for h in history[-5:]:
                status = "ACCEPTED" if h.get("accepted") else "REJECTED"
                history_text += (
                    f"- v{h['version']}: {h.get('name', 'N/A')} → "
                    f"reward={h.get('mean_reward', 0):.2f} ({h.get('change_pct', 0):+.1f}%) [{status}]\n"
                )
            parts.append(history_text)

        # Error feedback (if retrying)
        if error_feedback:
            parts.append(f"## Syntax Error to Fix\n{error_feedback}")
            parts.append("Please fix the syntax error in the code you generate.")

        # Instruction
        parts.append(
            "## Task\n"
            "Based on the research method above, modify the current reward function to incorporate the key innovation. "
            "Keep the existing structure that works well (tracking reward, bonus reward, improvement reward). "
            "Add or modify components based on the research method. "
            "Return ONLY the complete `__calculate_reward` method in a ```python ... ``` block."
        )

        return "\n\n".join(parts)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM API."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 4096,
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            message = data["choices"][0]["message"]
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")
            return content if content else reasoning

    def _extract_code(self, text: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        patterns = [
            r'```python\s*\n(.*?)```',
            r'```\s*\n(.*?)```',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                code = match.group(1).strip()
                # Ensure it starts with def
                lines = code.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("def __calculate_reward"):
                        return "\n".join(lines[i:])
                return code
        return None

    def _parse_response(self, response: str, method: dict) -> Optional[dict]:
        """Parse LLM response into a modification dict."""
        code = self._extract_code(response)
        if not code:
            print("[LLM] Could not extract code from response")
            return None

        # Validate it looks like a reward function
        if "def __calculate_reward" not in code:
            print("[LLM] Code does not contain __calculate_reward method")
            return None
        if "return reward" not in code and "return " not in code:
            print("[LLM] Code does not return a reward value")
            return None

        return {
            "code": code,
            "name": method.get("method_name", "LLM Generated"),
            "description": method.get("core_idea", "LLM-generated modification"),
            "category": method.get("category", ""),
            "method_id": method.get("method_id", ""),
        }
