from storybook import load_llm_and_tokenizer
from storybook import storybook
from example_configs import *

# TODO: include desired `Config`s before running
RUN_CONFIGS = [CatConfig3()]

llm, tokenizer = load_llm_and_tokenizer()

for i, config in enumerate(RUN_CONFIGS):
    story_image_path = f"assets/text/{config.text}"
    character_image_paths = [
        f"assets/character/{image}" for image in config.character_images
    ]
    character_data = [(name, False) for name in config.character_names]
    context_cache_file = (
        f"cache/context/{config.cache_context}.txt" if config.cache_context else None
    )
    character_cache_files = [
        f"cache/character/{name}.txt" for name in config.cache_characters
    ]
    output_path = f"output/{config.name}.jpg"

    print()
    print(f"===== Running Config {i+1} =====")
    try:
        _ = storybook(
            story_image_path,
            character_image_paths,
            llm,
            tokenizer,
            seed=config.seed,
            context_cache_file=context_cache_file,
            character_data=character_data,
            character_cache_files=character_cache_files,
            output_path=output_path,
            debug=True,
        )
    except Exception as e:
        print(f"Example failed with exception:\n{e}")
