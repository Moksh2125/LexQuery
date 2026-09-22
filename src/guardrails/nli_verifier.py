import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def verify_faithfulness(premise: str, hypothesis: str) -> str:
    """
    Uses Gemini API as a judge to determine if the hypothesis is entailed by the premise.
    Returns 'FAITHFUL' or 'UNSUPPORTED'.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-flash-lite-latest")
    
    prompt = f"""
    You are a strict legal hallucination detector. 
    Given the source text (Premise) and the generated claim (Hypothesis), does the source text completely entail and support the claim? 
    Reply strictly with 'FAITHFUL' if supported, or 'UNSUPPORTED' if it contains hallucinated facts, altered outcomes, or contradictions.
    Do not output any other text or explanation.
    
    Premise: {premise}
    
    Hypothesis: {hypothesis}
    """
    
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            result = response.text.strip().upper()
            if "FAITHFUL" in result:
                return "FAITHFUL"
            else:
                return "UNSUPPORTED"
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"Rate limit exceeded during NLI verification. Retrying in 35s... (Attempt {attempt+1}/3)")
                time.sleep(35)
            else:
                raise e
    
    # Fallback to UNSUPPORTED if API fails consistently
    return "UNSUPPORTED"
