"""Get a 256-dimensional embedding using Qwen.

Install:
    python -m pip install -U openai

Set DASHSCOPE_API_KEY or replace YOUR_API_KEY below, then run:
    python get_qwen_embedding.py
"""

import os
from openai import OpenAI


def main():
    api_key = os.getenv("DASHSCOPE_API_KEY") or "YOUR_API_KEY"
    if api_key == "YOUR_API_KEY":
        raise ValueError(
            "Set the DASHSCOPE_API_KEY environment variable "
            "or replace YOUR_API_KEY in this file."
        )

    input_text = "房间很干净，但是隔音不太好。"

    client = OpenAI(
        api_key=api_key,
        base_url="https://llm-s67p6tocy99yx6an.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )

    response = client.embeddings.create(
        model="qwen3.7-text-embedding",
        input=input_text,
        dimensions=256,  # Request a 256-dimensional vector
        encoding_format="float",
    )

    embedding = response.data[0].embedding
    assert len(embedding) == 256

    print("Input text:", input_text)
    print("Dimensions:", len(embedding))
    print("First 10 values:", embedding[:10])
    print("Full embedding:", embedding)


if __name__ == "__main__":
    main()
