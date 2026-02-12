"""
08 - Gradio UI
QA Bot အတွက် Gradio web interface ကို တည်ဆောက်တယ်။
"""

import gradio as gr


def build_interface(process_fn, answer_fn) -> gr.Blocks:
    """
    Gradio Blocks interface ကို တည်ဆောက်ပြီး return ပြန်ပေးတယ်။

    Args:
        process_fn: Document ကို process လုပ်မယ့် function
        answer_fn: Question ကို answer ပေးမယ့် function

    Returns:
        gr.Blocks: Gradio app instance
    """
    with gr.Blocks() as demo:
        gr.Markdown("# 📄 QA Bot")
        gr.Markdown(
            "Upload a document and ask questions using Qwen LLM and LangChain.\n\n"
            "**Supported:** PDF, TXT, CSV, Markdown, Word, PowerPoint, Excel, Images (OCR)"
        )

        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Upload Document",
                    file_types=[
                        ".pdf", ".txt", ".csv", ".md",
                        ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
                        ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp",
                    ],
                )
                process_btn = gr.Button("Process Document")
                status_output = gr.Textbox(label="Status", interactive=False)

            with gr.Column():
                question_input = gr.Textbox(label="Your Question")
                submit_btn = gr.Button("Ask")
                answer_output = gr.Textbox(label="Answer")

        # Button actions
        process_btn.click(
            fn=process_fn,
            inputs=file_input,
            outputs=status_output,
        )

        submit_btn.click(
            fn=answer_fn,
            inputs=question_input,
            outputs=answer_output,
        )

    return demo
