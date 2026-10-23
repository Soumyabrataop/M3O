"""
Google Gemini / Gemma API Diagnostic Engine for M3O.

Calls gemma-4-26b-a4b-it directly via the Gemini API to analyze layer telemetry,
representation geometry stability, and collapse prevention without fallback.
"""

import os
from typing import Any, Dict, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

TARGET_MODEL = "gemma-4-26b-a4b-it"


def diagnose_model_surgery_with_gemini(
    metrics: Dict[str, Any],
    api_key: Optional[str] = None,
) -> str:
    """
    Call gemma-4-26b-a4b-it directly via Google Gemini API.
    """
    resolved_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not resolved_key:
        return (
            "⚠️ **Gemini API Key Required**\n\n"
            "To run live `gemma-4-26b-a4b-it` diagnostics, get a key at "
            "[aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) "
            "and set `GEMINI_API_KEY=...` in your `.env` file."
        )

    prompt = f"""You are an elite Mechanistic Interpretability AI Diagnostician.
Analyze this real-time layer telemetry from an incremental weight surgery pass on Gemma 4:

• Task Vector Scaling Factor (α): {metrics.get('alpha', 'N/A')}
• Evaluated Projection Layers: {metrics.get('total_layers', 0)}
• Mean Cosine Similarity: {metrics.get('mean_cos_sim', 1.0):.6f} (Min: {metrics.get('min_cos_sim', 1.0):.6f})
• Mean Frobenius Norm Drift: {metrics.get('mean_frob_drift', 0.0):.6f}
• Layer Health Distribution: {metrics.get('stable_count', 0)} STABLE | {metrics.get('degraded_count', 0)} DEGRADED | {metrics.get('critical_count', 0)} CRITICAL

Provide a crisp, professional mechanistic assessment in Markdown:
1. Representation Geometry Assessment: Subspace alignment and drift impact.
2. Collapse Risk: Interpretation of Cosine Similarity against thresholds (0.85 Degraded, 0.65 Critical).
3. Alpha Recommendation: Direct guidance on tuning α for optimal capability retention vs stability."""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 2048,
        },
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{TARGET_MODEL}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": resolved_key,
    }

    import time
    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=45,
            )
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return "No analysis returned from Gemma 4."
                parts = candidates[0].get("content", {}).get("parts", [])
                # Extract actual output, excluding internal chain-of-thought blocks
                text_blocks = [p["text"] for p in parts if not p.get("thought") and "text" in p]
                if text_blocks:
                    return "\n\n".join(text_blocks).strip()
                return "\n\n".join(p["text"] for p in parts if "text" in p).strip()
            elif resp.status_code in (500, 503) and attempt < 2:
                time.sleep(1.5)
                continue
            else:
                return f"⚠️ **Gemma 4 API Error (HTTP {resp.status_code}):** {resp.text}"
        except Exception as e:
            if attempt < 2:
                time.sleep(1.5)
                continue
            return f"⚠️ **Connection Error:** {e}"
