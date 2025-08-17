def format_groq_prompt(prompt: str) -> dict:
    """
    Prepare the payload for Groq API (OpenAI-style format).
    """
    return {
        "model": "llama3-70b-8192",   # Default Groq model
        "messages": [
            {"role": "system", "content": "You are Mehwish AI powered by Groq LLaMA."},
            {"role": "user", "content": prompt}
        ]
    }
