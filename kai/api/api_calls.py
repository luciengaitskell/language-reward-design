from openai import OpenAI
from settings import Settings

openai_client = OpenAI(api_key=Settings().openai_key)

def vision(prompt_text: str, img_base64: str):
    """Run a GPT-4 vision model on the prompt text and image.

    ```
    from PIL import Image
    im = Image.fromarray(r)
    vision("what do you see?", image_to_base64(im))
    ```
    """
    gpt_model = "gpt-4-turbo"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                    },
                },
            ],
        }
    ]

    response = openai_client.chat.completions.create(
        model=gpt_model,
        messages=messages,
        temperature=0.2,
        max_tokens=300,
    )
    return response


def complete(prompt_text: str):
    """Run a GPT-4 model on the prompt text."""
    gpt_model = "gpt-4-turbo"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    response = openai_client.chat.completions.create(
        model=gpt_model,
        messages=messages,
        temperature=0.2,
        max_tokens=300,
    )
    return response
