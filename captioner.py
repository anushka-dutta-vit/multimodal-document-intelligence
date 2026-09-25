"""Generates a text caption for an image region using Claude's vision.

The caption is what actually gets embedded and indexed for that region —
this is the "fusion" step that lets a chart or photo be retrieved by a
plain-text question (see README: why captioning instead of CLIP).
"""
import base64
from io import BytesIO

from anthropic import Anthropic
from PIL import Image

from backend.config import settings

_client = Anthropic(api_key=settings.anthropic_api_key)

CAPTION_PROMPT = (
    "Describe this image from a document in 2-4 sentences, written so it can "
    "be searched by keyword later. If it is a chart or graph, state its type, "
    "what it plots, and the key numbers/trend visible. If it is a table "
    "rendered as an image, transcribe the key rows/values. If it is a photo "
    "or diagram, describe what it shows. Be concrete and specific — mention "
    "actual numbers, labels, and titles you can read. Do not add commentary."
)


def _image_to_b64(image: Image.Image) -> tuple[str, str]:
    buf = BytesIO()
    fmt = "PNG"
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/png"


def caption_image(image: Image.Image) -> str:
    if not settings.anthropic_api_key:
        return "[image region — caption unavailable: no ANTHROPIC_API_KEY set]"

    data, media_type = _image_to_b64(image)

    response = _client.messages.create(
        model=settings.claude_model,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    },
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }
        ],
    )

    parts = [b.text for b in response.content if b.type == "text"]
    return " ".join(parts).strip() or "[no caption generated]"


def table_rows_to_text(rows: list[list[str]]) -> str:
    """Flattens an extracted table into a compact text representation for
    embedding (kept separate from image captioning since we already have
    structured cell data — no need to spend a vision call on it)."""
    lines = [" | ".join(cell.strip() for cell in row) for row in rows]
    return "\n".join(lines)