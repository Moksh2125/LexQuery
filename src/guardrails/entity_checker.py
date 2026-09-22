import re

def verify_hard_entities(source_text: str, generated_text: str) -> list[str]:
    """
    Extracts strict legal entities (Sections, Acts, Dates, CNR numbers) from the generated_text
    using pure regex and checks if they exist in the source_text.
    Returns a list of hallucinated entities (entities present in generated but missing in source).
    """
    hallucinated = []
    
    # Simple regex patterns for legal entities
    patterns = {
        "Section": r"(?i)\b(?:Section|Sec\.?)\s+\d+[a-zA-Z]?\b",
        "Act": r"(?i)\b[A-Z][a-zA-Z\s]+(?:Act|Code)\b",
        "Date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "CNR": r"\b[a-zA-Z0-9]{16}\b"
    }
    
    for entity_type, pattern in patterns.items():
        extracted = re.findall(pattern, generated_text)
        for entity in extracted:
            # Simple string matching to check presence in source
            if entity.lower() not in source_text.lower():
                hallucinated.append(entity)
                
    # Remove duplicates
    return list(set(hallucinated))
