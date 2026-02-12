# Qwen/Qwen2.5-72B-Instruct — size ကြီးလေ Myanmar handling ကောင်းလေ (ဒါပေမဲ့ Inference API မှာ free မရ)
# google/gemma-2-9b-it — Myanmar ကို training data ထဲ ပိုပါတယ်
# meta-llama/Llama-3.1-8B-Instruct — multilingual ပိုကောင်းတယ်
# 
# pip install langchain langchain-huggingface langchain-chroma langchain-community chromadb gradio pypdf sentence-transformers
# 
# pip install langchain-classic
# 
import os
import gradio as gr
import torch

# 1. Import the necessary libraries
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Setup environment (Replace 'hf_...' with your actual token if not set in env)
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_your_token_here"

# 2. Initialize the LLM (Qwen 2.5) from local cache
# local_files_only=True ensures we don't re-download if already cached.
model_id = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True,
)
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=False,
    temperature=0.1,
)
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# 3. Define the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="ibm-granite/granite-embedding-125m-english",
    model_kwargs={"local_files_only": True},
)

# Global variable to store the QA chain
qa_chain = None

def process_document(pdf_file):
    """
    Loads, splits, and indexes the PDF document.
    """
    global qa_chain
    
    if pdf_file is None:
        return "Please upload a PDF file first."

    # 4. Define the PDF document loader
    # Gradio 6.x File component returns a filepath string by default
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()

    # 5. Define the text splitter
    # Splitting text helps the model digest large documents in chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # 6. Define the vector store
    # ChromaDB stores the embeddings of the text chunks for fast retrieval.
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        collection_name="granite_collection" # Optional: name your collection
    )

    # 7. Define the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 8. Define a question-answering chain
    # We create a prompt template to guide Granite's behavior
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Combine the retriever and the LLM into a RAG chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    return f"Processed {len(splits)} chunks from the PDF. You can now ask questions!"

def answer_question(question):
    """
    Uses the QA chain to answer the user's question.
    """
    if qa_chain is None:
        return "Please upload and process a document first."
    
    response = qa_chain.invoke({"input": question})
    return response["answer"]

# 9. Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 📄 QA Bot")
    gr.Markdown("Upload a PDF and ask questions using Qwen LLM and LangChain.")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_btn = gr.Button("Process Document")
            status_output = gr.Textbox(label="Status", interactive=False)
        
        with gr.Column():
            question_input = gr.Textbox(label="Your Question")
            submit_btn = gr.Button("Ask")
            answer_output = gr.Textbox(label="Answer")

    # Button actions
    process_btn.click(
        fn=process_document,
        inputs=file_input,
        outputs=status_output
    )
    
    submit_btn.click(
        fn=answer_question,
        inputs=question_input,
        outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())