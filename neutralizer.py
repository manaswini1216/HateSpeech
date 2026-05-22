import streamlit as st
from google import genai

# ---------------- GEMINI CLIENT ---------------- #

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# ---------------- SIMPLE CACHE ---------------- #

cache = {}

# ---------------- NEUTRALIZER FUNCTION ---------------- #

def neutralize_text(text):

    # return cached result if already generated
    if text in cache:
        return cache[text]

    try:

        prompt = f"""
Rewrite this text in a respectful and non-toxic way
without changing the original meaning.

Text:
{text}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        neutralized = response.text.strip()

        # fallback if empty response
        if not neutralized:
            neutralized = "Neutralized version unavailable."

        # save in cache
        cache[text] = neutralized

        return neutralized

    except Exception as e:

        print("Gemini Error:", e)

        return "Neutralization service unavailable."

# ---------------- TEST ---------------- #

if __name__ == "__main__":

    sample_text = "You are a stupid useless person"

    result = neutralize_text(sample_text)

    print(result)
