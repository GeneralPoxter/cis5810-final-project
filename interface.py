from storybook import load_llm_and_tokenizer, storybook
from stylization import load_rev_network, image_transfer

import gradio as gr

# Hugging Face LM
llm, tokenizer = load_llm_and_tokenizer()

# CAP-VST Reversible Network
RevNetwork = load_rev_network()


# Gradio interface setup
def interface(
    text_image,
    character_images,
    style_image,
    cache_title,
    context_cache_file,
    character_data,
    character_cache_files,
):
    character_image_paths = (
        [path for path, _ in character_images] if character_images else []
    )
    character_data = [
        (name, save == "y") for name, save in character_data.values.tolist()
    ]
    output = storybook(
        text_image,
        character_image_paths,
        llm,
        tokenizer,
        cache_title=cache_title,
        context_cache_file=context_cache_file,
        character_data=character_data,
        character_cache_files=character_cache_files,
        debug=True,
    )
    if style_image:
        output = image_transfer(output, style_image, RevNetwork)
    return output


def initialize_character_data(character_images):
    if character_images:
        return [(None, "n")] * len(character_images)
    return []


# Create the Gradio app
gr_interface = gr.Blocks(theme=gr.themes.Soft())

with gr_interface:
    gr.Markdown(
        """
        # 🎨 Text-to-Comic Generator
        ### Turn your imagination into a comic strip with stylized characters and stories!
        """
    )

    with gr.Row(elem_id="input-section"):
        text_image = gr.Image(type="filepath", label="📝 Upload Text Image")
        character_images = gr.Gallery(
            type="filepath",
            label="🖼️ Upload Character Images (multiple allowed)",
        )

    with gr.Accordion("Context Cache (optional)", open=False):
        with gr.Row():
            cache_title = gr.Textbox(
                None,
                label="Story title (leave blank to not save to cache)",
            )
            context_cache_file = gr.FileExplorer(
                "*.txt",
                root_dir="./cache/context",
                file_count="single",
                label="Add cached story context",
            )
        with gr.Row():
            character_data = gr.Dataframe(
                [],
                label="Character settings (in upload order)",
                headers=["Name (optional)", "Save to cache? (y/n)"],
                col_count=2,
                interactive=True,
            )
            character_cache_files = gr.FileExplorer(
                "*.txt",
                root_dir="./cache/character",
                file_count="multiple",
                label="Add cached characters",
            )

    character_images.upload(
        fn=initialize_character_data,
        inputs=[character_images],
        outputs=[character_data],
    )
    character_images.change(
        fn=initialize_character_data,
        inputs=[character_images],
        outputs=[character_data],
    )

    with gr.Row(elem_id="input-section"):
        style_image = gr.Image(
            type="filepath", label="🎨 Upload Style Reference (optional)"
        )

    generate_btn = gr.Button("✨ Generate Comic", elem_id="generate-btn")

    with gr.Row(elem_id="output-section"):
        output = gr.Image(label="Generated Comic")

    generate_btn.click(
        fn=interface,
        inputs=[
            text_image,
            character_images,
            style_image,
            cache_title,
            context_cache_file,
            character_data,
            character_cache_files,
        ],
        outputs=output,
    )

# Launch the Gradio app
gr_interface.launch(share=True)
