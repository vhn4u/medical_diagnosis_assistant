"""Domain fine-tuning of a sentence-transformer embedding model on medical
symptom -> condition triplets, using Triplet Loss (parameter-efficient: the
base model is small, ~22M params, so this trains fully on CPU in minutes).
"""

from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from .data_prep import load_knowledge_base, build_finetune_triplets

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FINETUNED_MODEL_PATH = MODELS_DIR / "medical-minilm-finetuned"


def fine_tune_embedding_model(epochs: int = 4, batch_size: int = 8) -> SentenceTransformer:
    """Fine-tune all-MiniLM-L6-v2 on the medical knowledge base using triplet loss
    so that symptom phrasing embeds close to its true condition and far from
    unrelated conditions. Saves the result to MODELS_DIR.
    """
    df = load_knowledge_base()
    triplets = build_finetune_triplets(df)

    train_examples = [InputExample(texts=[a, p, n]) for a, p, n in triplets]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    model = SentenceTransformer(BASE_MODEL)
    train_loss = losses.TripletLoss(model=model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=10,
        show_progress_bar=True,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(FINETUNED_MODEL_PATH))
    return model


def load_finetuned_model() -> SentenceTransformer:
    """Load the fine-tuned model if present, otherwise fall back to the base model."""
    if FINETUNED_MODEL_PATH.exists():
        return SentenceTransformer(str(FINETUNED_MODEL_PATH))
    return SentenceTransformer(BASE_MODEL)


if __name__ == "__main__":
    fine_tune_embedding_model()
