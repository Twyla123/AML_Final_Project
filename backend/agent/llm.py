tokenizer = AutoTokenizer.from_pretrained("./model_final")
model = AutoModelForCausalLM.from_pretrained(
    "./model_final",
    device_map="auto",
    dtype="auto"
)

def llm_complete(
    prompt: str,
    max_new_tokens: int = 60
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.encode("\n")[0],
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# --------------------------------------------------
# Intent prediction (fine-tuned epoch_5)
# --------------------------------------------------
def predict_intent_with_epoch5(
    query: str,
    model,
    tokenizer,
    intent_labels=INTENT_LABELS,
    max_new_tokens: int = 4
) -> str:
    """
    Deterministic intent classifier using fine-tuned epoch_5.

    - NO free generation
    - HARD stop
    - One label only
    """

    prompt = (
        "You are a financial intent classifier.\n\n"
        "Choose exactly ONE intent from this list:\n"
        + ", ".join(intent_labels)
        + "\n\n"
        f"User query:\n{query}\n\n"
        "Intent:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,               
            temperature=0.0,                
            repetition_penalty=1.0,
            eos_token_id=tokenizer.encode("\n")[0], 
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ----------------------------
    # Parse the LAST token chunk
    # ----------------------------
    tail = text.split("Intent:", 1)[-1].strip()
    pred = tail.split()[0].lower()

    if pred in intent_labels:
        return pred

    # -------- HARD fallback (never crash agent) --------
    return "analysis"

