"""
eac_synth/synthesis/verbalizer.py
Path-to-QA verbalization -- Stage 4 of EAC-Synth.

Converts an extracted reasoning chain P* into a (question, steps, answer)
triple by prompting the teacher model M_T.

Key design choices (§3.5.1):
  - Intermediate entities are *obfuscated* as domain-specific clues,
    not named directly -- forces multi-hop traversal rather than entity lookup.
  - Terminal entity e_L is the target answer y.
  - Sampling uses temperature tau=0.85, top_p=0.90 for lexical diversity.
  - Chain-of-thought steps are interleaved with the answer in the assistant
    turn during training (Eq.1).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from eac_synth.graph.graph_builder import Edge
from eac_synth.graph.chain_extractor import format_chain_for_prompt




_VERBALIZE_SYSTEM = (
    "You are an expert question writer for a {domain} training dataset.\n"
    "Your task: given a multi-hop reasoning chain, write a challenging\n"
    "question whose answer requires traversing every step of the chain.\n\n"
    "Rules:\n"
    "1. Do NOT name intermediate entities directly -- describe them via\n"
    "   domain-specific clues (role, property, mechanism, statute number, etc.).\n"
    "2. The final answer must be the terminal entity in the chain.\n"
    "3. Include a step-by-step reasoning trace (chain-of-thought).\n"
    "4. Output valid JSON only -- no prose outside the JSON object.\n"
)

_VERBALIZE_USER = (
    "Domain: {domain}\n\n"
    "Reasoning chain:\n"
    "{chain}\n\n"
    "Output JSON with keys:\n"
    '  "question": str   -- the multi-hop question\n'
    '  "steps":    list  -- ordered list of reasoning step strings\n'
    '  "answer":   str   -- the terminal entity (final answer)\n'
)




class Verbalizer:
    """Converts a reasoning chain P* into a QA training sample.

    Args:
        teacher_fn: callable(system: str, user: str, temperature: float,
            top_p: float) -> dict.  Should return parsed JSON matching the
            schema {"question": str, "steps": list[str], "answer": str}.
        domain: domain label inserted into the prompt.
        temperature: sampling temperature (default 0.85, paper §3.5.1).
        top_p: nucleus sampling probability (default 0.90).
        max_retries: number of retries on parse failure (default 2).
    """

    def __init__(
        self,
        teacher_fn: Callable[..., Any],
        domain: str,
        temperature: float = 0.85,
        top_p: float = 0.90,
        max_retries: int = 2,
    ):
        self._teacher = teacher_fn
        self.domain = domain
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries

    def verbalize(self, path: List[Edge]) -> Optional[Dict[str, Any]]:
        """Convert *path* into a QA sample dict.

        Args:
            path: list of Edge objects representing P*.

        Returns:
            Dict with keys "question", "steps", "answer", and "path"
            (the raw edge list as dicts), or None on failure.
        """
        chain_str = format_chain_for_prompt(path)
        system = _VERBALIZE_SYSTEM.format(domain=self.domain)
        user = _VERBALIZE_USER.format(domain=self.domain, chain=chain_str)

        for attempt in range(self.max_retries + 1):
            try:
                raw = self._teacher(
                    system,
                    user,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                # raw may already be a dict or a JSON string
                if isinstance(raw, str):
                    raw = raw.strip()
                    # strip markdown code fences if present
                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                    result = json.loads(raw)
                else:
                    result = raw

                # Validate required keys
                if not all(k in result for k in ("question", "steps", "answer")):
                    raise ValueError(f"Missing keys in result: {result}")

                result["path"] = [e.to_dict() for e in path]
                return result

            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                if attempt == self.max_retries:
                    # Log and return None -- the sample is skipped
                    import warnings
                    warnings.warn(
                        f"Verbalization failed after {self.max_retries + 1} "
                        f"attempts: {exc}"
                    )
                    return None

        return None  # unreachable, satisfies type checker




def make_openai_teacher(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Callable:
    """Return a teacher function backed by the OpenAI Chat Completions API.

    The returned callable has signature:
        teacher_fn(system, user, temperature, top_p) -> dict

    Args:
        model: model identifier (default "gpt-4o-mini", paper §4.3).
        api_key: OpenAI API key; falls back to OPENAI_API_KEY env var.

    Returns:
        Callable compatible with both GraphBuilder edge scorer,
        LongestChainExtractor scorer, and Verbalizer.
    """
    import os
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("pip install openai") from e

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def _call(
        system: str,
        user: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Any:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            top_p=top_p,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    return _call
