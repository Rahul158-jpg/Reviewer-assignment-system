import requests


def call_llm(prompt, models=None, timeout=10, retries=1, temperature=None, options=None):
    if models is None:
        models = ["phi3", "mistral"]

    for model in models:
        for attempt in range(max(1, int(retries))):
            try:
                payload = {"model": model, "prompt": prompt, "stream": False}
                merged_options = dict(options or {})
                if temperature is not None:
                    merged_options["temperature"] = temperature
                if merged_options:
                    payload["options"] = merged_options

                response = requests.post(
                    "http://127.0.0.1:11434/api/generate",
                    json=payload,
                    timeout=timeout,
                )

                if response.status_code != 200:
                    print("LLM HTTP ERROR:", response.status_code, "model=", model)
                    continue

                data = response.json()
                text = data.get("response", "").strip()

                if text:
                    return text

            except requests.exceptions.Timeout:
                print(f"LLM TIMEOUT (model={model} attempt={attempt+1})")

            except Exception as e:
                print("LLM ERROR:", e)

    return None


def generate_reviewer_reason(paper, reviewer, sim, auth, final):
    overlap_terms = reviewer.get("overlap_terms", [])
    overlap_text = ", ".join(overlap_terms[:4])

    comparisons = reviewer.get("comparison", [])
    comp_ids = [str(c.get("reviewer_id")) for c in comparisons]
    comp_text = ", ".join(comp_ids)

    prompt = (
        f"One short sentence: why is this reviewer best for '{paper.get('title','')}'? "
        f"Mention top topics: {overlap_text}. Mention outranking: {comp_text}."
    )

    response = call_llm(prompt, models=["phi3", "mistral"], timeout=10, retries=1)

    if response and len(response) >= 15:
        return response

    overlap_part = (
        f"overlapping topics ({overlap_text})" if overlap_text else "relevant domain alignment"
    )

    comp_part = (f" and higher ranking than reviewers {comp_text}" if comp_text else "")

    return (
        f"Reviewer selected due to {overlap_part} with authority {round(auth,2)}{comp_part}."
    )
