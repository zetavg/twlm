import gradio as gr
import json
import matplotlib
from . import get_page_data

# https://github.com/matplotlib/matplotlib/issues/14304/#issuecomment-545717061
matplotlib.use('agg')


def clear_outputs():
    return '', '', ''


def handle_get_page(page_title, lang):
    if not page_title:
        raise gr.Error(f"Please enter a page title.")
    if not lang:
        raise gr.Error(f"Please select a language.")
    try:
        page_data = get_page_data(page_title, lang=lang)
        page_metadata = {
            key: value for key, value
            in page_data.items() if key not in ["html", "markdown"]
        }
        page_metadata_json = json.dumps(
            page_metadata, indent=2, ensure_ascii=False)
        return (
            page_data['markdown'],
            page_data['html'],
            page_metadata_json,
        )
    except Exception as e:
        raise gr.Error(f"Error when getting page '{page_title}': {e}")


def render_content(markdown, html):
    # To prevent invalid syntax crashing everything.
    return (markdown, html)


def handle_example_select(evt: gr.SelectData):
    return evt.value[1], evt.value[2]


css = """
.group_box, .group_box > * {
    padding: 0 !important;
    gap: 0 !important;
}
.group_box > * > .form,
.group_box > * > * > .form,
.group_box > * > .block,
.group_box > * > * > .block {
    border: 0;
    box-shadow: none;
}
.group_box > .with_group_box_element_padding,
.group_box > * > .with_group_box_element_padding,
.group_box > * > * > .with_group_box_element_padding {
    padding: var(--block-padding) !important;
}

#example_selections {
    padding-top: 0 !important;
}
"""

examples = [
    ('網際網路', 'zh-tw'),
    ('Internet', 'en'),
    ('梯度下降法', 'zh-tw'),
    ('Gradient descent', 'en'),
    ('傅立葉轉換', 'zh-tw'),
    ('Fourier transformation', 'en'),
    ('Haskell', 'zh-tw'),
    ('Haskell', 'en'),
]


with gr.Blocks(title="Wikipedia", css=css) as demo:
    gr.Markdown("""
        # Wikipedia
    """)
    with gr.Box(elem_classes="group_box"):
        with gr.Row():
            title = gr.Textbox(
                label="Page Title",
                placeholder="Machine Learning",
                lines=1)
            lang = gr.Dropdown(
                label="Language",
                choices=["en", "zh-tw"],
                value="zh-tw",
                allow_custom_value=True)
        example_buttons = gr.Dataset(
            label="Examples",
            components=[gr.Textbox(visible=False)],
            samples=[
                [f"{example[0]} ({example[1]})", example[0], example[1]]
                for example in examples
            ],
            elem_classes="with_group_box_element_padding",
            elem_id="example_selections"
        )
    get_page_btn = gr.Button("Get Page!", variant="primary")
    # Syntax error in LaTeX will break the app (ParseFatalException).
    # with gr.Tab(label="Markdown"):
    #     page_content_markdown = gr.Markdown("")
    with gr.Tab(label="Markdown (Source)"):
        page_content_markdown_source = gr.Code(
            "",
            label="Markdown",
            language="markdown",
            interactive=False
        )
    # with gr.Tab(label="HTML"):
    #     page_content_html = gr.HTML("")
    with gr.Tab(label="HTML (Source)"):
        page_content_html_source = gr.Code(
            "",
            label="HTML",
            language="html",
            interactive=False
        )
    with gr.Tab(label="Page Metadata"):
        page_metadata_json = gr.Code(
            "",
            label="JSON",
            language="json",
            interactive=False
        )

    example_buttons.select(
        fn=handle_example_select,
        outputs=[title, lang]
    )

    get_page_btn.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[
            page_content_markdown_source,
            page_content_html_source,
            page_metadata_json,
        ]
    ).then(
        api_name='get_page',
        fn=handle_get_page,
        inputs=[title, lang],
        outputs=[
            page_content_markdown_source,
            page_content_html_source,
            page_metadata_json,
        ]
    )  # .then(  # Syntax error in LaTeX will break the app (ParseFatalException).
    #     fn=render_content,
    #     inputs=[page_content_markdown_source, page_content_html_source],
    #     outputs=[page_content_markdown, page_content_html]
    # )

demo.queue(concurrency_count=1).launch()
