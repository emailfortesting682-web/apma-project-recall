import json
from openai import OpenAI

SYSTEM_PROMPT = """
You convert user instructions into a structured summary template.

Return ONLY valid JSON in this format:

{
  "sections": ["Section 1", "Section 2", "..."],
  "tone": "simple | detailed | technical | executive",
  "length": "short | medium | long"
}

Rules:
- Infer logical sections from instructions
- Default tone = simple
- Default length = short
"""

def parse_summary_instructions(instructions: str) -> dict:
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instructions}
        ],
        temperature=0.2
    )

    content = resp.choices[0].message.content.strip()
    return json.loads(content)
