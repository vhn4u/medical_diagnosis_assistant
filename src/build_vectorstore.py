"""Build a FAISS vector store over the medical knowledge base using the
domain fine-tuned sentence-transformer embeddings (LangChain-compatible).
"""

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .data_prep import load_knowledge_base, knowledge_base_to_documents
from .finetune_embeddings import FINETUNED_MODEL_PATH, BASE_MODEL

VECTORSTORE_DIR = Path(__file__).resolve().parent.parent / "models" / "faiss_index"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a LangChain embeddings wrapper around our (fine-tuned if available)
    sentence-transformer model."""
    model_path = str(FINETUNED_MODEL_PATH) if FINETUNED_MODEL_PATH.exists() else BASE_MODEL
    return HuggingFaceEmbeddings(model_name=model_path)


def build_vectorstore() -> FAISS:
    """Embed every knowledge-base entry and build a searchable FAISS index."""
    df = load_knowledge_base()
    docs_raw = knowledge_base_to_documents(df)
    documents = [Document(page_content=d["text"], metadata=d["metadata"]) for d in docs_raw]

    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(documents, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    return vectorstore


def load_vectorstore() -> FAISS:
    """Load a previously built FAISS index, building it if it doesn't exist yet."""
    embeddings = get_embedding_model()
    if VECTORSTORE_DIR.exists():
        return FAISS.load_local(
            str(VECTORSTORE_DIR), embeddings, allow_dangerous_deserialization=True
        )
    return build_vectorstore()


if __name__ == "__main__":
    build_vectorstore()
