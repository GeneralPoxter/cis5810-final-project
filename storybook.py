from constants import *

import argparse
import time
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH
os.environ["HF_TOKEN"] = HF_TOKEN

from google.cloud import vision
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig, Image

vertexai.init(project=PROJECT_ID, location=REGION)

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed
import torch


def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f"Google Vision API error: {response.error.message}")
    if len(texts) == 0:
        raise Exception(f"No text detected in image: {path}")

    description = texts[0].description.strip()

    return description


def load_llm_and_tokenizer(
    model_name=LLM_NAME,
    model_kwargs={
        "quantization_config": (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if torch.cuda.is_available()
            else None
        ),
        "max_length": 1024,
        "torch_dtype": torch.float16,
        "device_map": "auto",
    },
):
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.padding_side = "left"
    return model, tokenizer


@torch.no_grad()
def prompt_llm(
    prompt,
    model,
    tokenizer,
    generation_kwargs={
        "min_new_tokens": 1,
        "max_new_tokens": 100,
        "do_sample": False,
    },
):
    set_seed(LLM_SEED)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    response = model.generate(**inputs, **generation_kwargs)
    generated_ids = response[0][inputs.input_ids.shape[-1] :]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    output = output.strip()
    return output


def summarize_text(text, model, tokenizer):
    prompt = f"Summarize the following text as four storyboard panels in 100 tokens:\n{text}\nFour panels (do not output anything else):"
    output = prompt_llm(prompt, model, tokenizer)
    output = "\n".join(output.split("\n")[:4])
    return output


def summarize_context(context, model, tokenizer):
    prompt = f"Summarize the following text in 50 tokens:\n{context}\nSummary:"
    output = prompt_llm(
        prompt,
        model,
        tokenizer,
        generation_kwargs={
            "min_new_tokens": 1,
            "max_new_tokens": 50,
            "do_sample": False,
        },
    )
    return output.split("\n")[0]


def summarize_character(path):
    model = GenerativeModel("gemini-1.5-flash-002")
    config = GenerationConfig(seed=SEED)
    image = Image.load_from_file(path)
    prompt = "Describe the character shown in this image in 10 words or less, capturing both physical and personality traits, but keep things positive"
    response = model.generate_content([prompt, image], generation_config=config)

    if len(response.candidates) == 0:
        raise Exception(f"No description generated for image: {path}")

    description = response.candidates[0].content.parts[0].text.strip()

    return description


def storybook(
    story_image_path,
    character_image_paths,
    llm,
    tokenizer,
    seed=SEED,
    cache_title=None,
    context_cache_file=None,
    character_data=[],
    character_cache_files=[],
    output_path=None,
    debug=False,
):
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

    story = detect_text(story_image_path)

    context = None
    if context_cache_file:
        with open(context_cache_file, "r") as f:
            text = f.read().strip()
            context = summarize_context(text, llm, tokenizer)

    if cache_title:
        with open(f"cache/context/{cache_title} ({timestamp}).txt", "w") as f:
            if context:
                f.write(f"{context}\n{story}")
            else:
                f.write(story)

    story = summarize_text(story, llm, tokenizer)

    character_prompt = "With the following characters:"
    format_prompt = lambda name, desc: f"\n* {name}: {desc}" if name else f"\n* {desc}"

    for i, character_image_path in enumerate(character_image_paths):
        name = None
        description = summarize_character(character_image_path)
        if character_data and i < len(character_data):
            name, save_cache = character_data[i]
            if save_cache:
                path = f"cache/character/{name if name else 'Unnamed character'} ({timestamp}).txt"
                with open(path, "w") as f:
                    f.write(f"{name}\n{description}")
        character_prompt += format_prompt(name, description)

    if character_cache_files:
        for character_cache_file in character_cache_files:
            with open(character_cache_file, "r") as f:
                name = f.readline().strip()
                description = f.read().strip()
                character_prompt += format_prompt(name, description)

    prompt = f"Create a four-panel comic of the following story:\n{story}\nDo not generate people."
    if character_image_paths or character_cache_files:
        prompt = f"{character_prompt}\n{prompt}"
    if context:
        prompt = f"With the following context:\n{context}\n{prompt}"

    if debug:
        print(f"Generating image with prompt:\n{prompt}")

    if output_path is None:
        output_path = f"output/output_storybook_{timestamp}.jpg"

    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
    images = model.generate_images(
        prompt=prompt,
        negative_prompt="text and speech bubbles",
        number_of_images=1,
        language="en",
        # can't use a seed value and watermark at the same time.
        add_watermark=False,
        seed=seed,
        aspect_ratio="1:1",
        # safety_filter_level="block_some",
        # person_generation="allow_adult",
        # person_generation="dont_allow",
    )

    try:
        images[0].save(location=output_path, include_generation_parameters=False)
    except:
        raise Exception("Generated image was blocked by Imagen 3.0 API")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Path to text image", required=True)
    parser.add_argument("--characters", nargs="+", help="Paths to character images")
    parser.add_argument("--output", type=str, help="Path to output image")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    llm, tokenizer = load_llm_and_tokenizer()
    output = storybook(
        args.text,
        args.characters,
        llm,
        tokenizer,
        output_path=args.output,
        debug=args.debug,
    )
    print(f"Save at {output}")
