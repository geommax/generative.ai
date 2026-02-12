"""
06 - Retrieval
Vector store ထဲက relevant documents တွေကို retrieve လုပ်တယ်။
"""

from langchain_chroma import Chroma


def get_retriever(vectorstore: Chroma, top_k: int = 3):
    """
    Vector store ကနေ retriever object ကို တည်ဆောက်ပေးတယ်။

    Args:
        vectorstore: Chroma vector store instance
        top_k: retrieve လုပ်မယ့် document အရေအတွက်

    Returns:
        Retriever instance
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever
