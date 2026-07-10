"""Data loading and preparation utilities for the Medical Diagnosis Assistant."""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_knowledge_base() -> pd.DataFrame:
    """Load the structured medical knowledge base (condition -> symptoms/treatment)."""
    df = pd.read_csv(DATA_DIR / "medical_knowledge_base.csv")
    return df


def load_patient_histories() -> pd.DataFrame:
    """Load synthetic patient history records used for retrieval demos."""
    df = pd.read_csv(DATA_DIR / "patient_histories.csv")
    return df


def knowledge_base_to_documents(df: pd.DataFrame) -> list[dict]:
    """Flatten each knowledge-base row into a retrieval-ready text document + metadata."""
    documents = []
    for _, row in df.iterrows():
        text = (
            f"Condition: {row['condition']}\n"
            f"Specialty: {row['specialty']}\n"
            f"Symptoms: {row['symptoms']}\n"
            f"Description: {row['description']}\n"
            f"Recommended Action: {row['recommended_action']}\n"
            f"Severity: {row['severity']}"
        )
        documents.append(
            {
                "text": text,
                "metadata": {
                    "condition": row["condition"],
                    "specialty": row["specialty"],
                    "severity": row["severity"],
                },
            }
        )
    return documents


def build_finetune_triplets(df: pd.DataFrame, n_negatives_per_anchor: int = 2) -> list[tuple[str, str, str]]:
    """Build (anchor, positive, negative) triplets for domain fine-tuning of the
    sentence-transformer embedding model.

    Anchor: the symptom list for a condition (as a patient might phrase it).
    Positive: the condition's own clinical description (semantically matching).
    Negative: another condition's description (semantically mismatched).
    """
    triplets = []
    records = df.to_dict("records")
    for i, row in enumerate(records):
        anchor = f"Patient reports: {row['symptoms']}"
        positive = f"{row['condition']}: {row['description']}"
        negative_pool = [r for j, r in enumerate(records) if j != i]
        for neg in negative_pool[:n_negatives_per_anchor]:
            negative = f"{neg['condition']}: {neg['description']}"
            triplets.append((anchor, positive, negative))
    return triplets
