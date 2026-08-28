from typing import Dict, List
from openai import OpenAI
import os


SYSTEM_PROMPT = """
You are a NASA mission intelligence expert specializing in Apollo 11,
Apollo 13, and the Space Shuttle Challenger mission.

Answer questions using the retrieved NASA source material provided to you.

Rules:
1. Base factual claims only on the provided retrieved context.
2. Do not invent mission facts that are not supported by the context.
3. If the context is insufficient, clearly say that the retrieved sources
   do not provide enough information.
4. Cite the supplied source names when explaining factual information.
5. Keep answers clear, concise, and technically accurate.
6. Use previous conversation turns only for conversational continuity;
   factual mission claims must still be grounded in retrieved context.
"""


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo",
) -> str:
    """Generate a grounded NASA mission response using OpenAI."""

    if not openai_key:
        raise ValueError("An OpenAI API key is required.")

    if not user_message or not user_message.strip():
        raise ValueError("User message cannot be empty.")

    client = OpenAI(
        api_key=openai_key,
        base_url=os.getenv("OPENAI_BASE_URL")
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # Add previous conversation history
    for message in conversation_history:
        if (
            isinstance(message, dict)
            and message.get("role") in {"user", "assistant"}
            and message.get("content")
        ):
            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )

    # Add retrieved context and current question
    context_text = context.strip() if context else "No relevant context was retrieved."

    messages.append(
        {
            "role": "user",
            "content": f"""
Retrieved NASA context:

{context_text}

Current question:
{user_message}

Answer using the retrieved context. Cite relevant sources where possible.
If the context is insufficient, say so clearly.
""".strip(),
        }
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()