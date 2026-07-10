"""Advanced prompt engineering for the Medical Diagnosis Assistant.

Techniques applied:
  1. Role/system priming with explicit scope and safety constraints.
  2. Few-shot exemplars to fix the output structure.
  3. Retrieval-context grounding (RAG) with citation of source condition.
  4. Chain-of-thought scaffolding ("Reasoning" section) separated from the
     user-facing "Summary" to keep answers structured and auditable.
  5. Guardrail instructions for out-of-scope / emergency / low-confidence cases.
"""

SYSTEM_PROMPT = """You are MedAssist, a clinical-decision-support assistant used ONLY for
educational and informational purposes. You are NOT a licensed physician and you
do NOT provide a final diagnosis. You help a user reason about POSSIBLE conditions
that match reported symptoms, grounded strictly in the CONTEXT provided to you.

Rules you must always follow:
- Base your reasoning only on the CONTEXT below plus the patient's reported symptoms.
  If the context does not contain a good match, say so explicitly instead of guessing.
- If any reported symptom matches a "Critical" severity condition in the context
  (e.g. chest pain, stroke signs, difficulty breathing), your FIRST line must be an
  emergency-care recommendation before anything else.
- Never state a diagnosis as certain. Use language such as "may be consistent with"
  or "could suggest".
- Always end with the disclaimer: "This is not a substitute for professional
  medical advice. Please consult a licensed healthcare provider."
- Keep the "Reasoning" section internal/analytical and the "Summary" section
  short and patient-readable.
"""

FEW_SHOT_EXAMPLES = """
--- EXAMPLE ---
CONTEXT:
Condition: Migraine
Specialty: Neurology
Symptoms: severe throbbing headache, nausea, sensitivity to light
Recommended Action: Rest in dark room, migraine-specific medication, consult neurologist.
Severity: Moderate

PATIENT SYMPTOMS: severe throbbing headache, nausea, sensitivity to light, occurring 3x/month

RESPONSE:
Reasoning: The reported symptoms (throbbing headache, nausea, photosensitivity) align
closely with the Migraine entry in the knowledge base. Frequency (3x/month) is
consistent with recurrent migraine rather than a one-off tension headache. No
critical/emergency symptoms (e.g. sudden weakness, confusion) are present.

Summary:
- Possible match: Migraine (Neurology) — Moderate severity
- Why: throbbing headache + nausea + light sensitivity is a classic migraine triad
- Suggested next step: Rest in a dark room, consider migraine-specific medication,
  and follow up with a neurologist if episodes continue or worsen.
- This is not a substitute for professional medical advice. Please consult a
  licensed healthcare provider.
--- END EXAMPLE ---
"""

RAG_ANSWER_TEMPLATE = """{system_prompt}
{few_shot}
--- CONTEXT (retrieved from medical knowledge base) ---
{context}
--- END CONTEXT ---

PATIENT SYMPTOMS: {question}

Follow the exact structure shown in the example (Reasoning: ... / Summary: ...).
RESPONSE:
"""


def build_rag_prompt(context: str, question: str) -> str:
    """Assemble the full prompt sent to the LLM, combining system priming,
    few-shot guidance, retrieved context, and the user's symptoms."""
    return RAG_ANSWER_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        few_shot=FEW_SHOT_EXAMPLES,
        context=context,
        question=question,
    )
