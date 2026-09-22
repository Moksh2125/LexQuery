import os
import sys
import time
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.guardrails.nli_verifier import verify_faithfulness
from src.guardrails.entity_checker import verify_hard_entities

class TestLLMNLIVerifier:
    def test_factual_claim(self):
        premise = "The Supreme Court granted bail to the appellant on medical grounds."
        hypothesis = "The appellant was given bail by the Supreme Court because of health issues."
        
        result = verify_faithfulness(premise, hypothesis)
        assert result == "FAITHFUL"
        
        # 10-second sleep to avoid rate limits on free tier
        time.sleep(10)
        
    def test_hallucinated_claim(self):
        premise = "The High Court rejected the bail application of the accused."
        hypothesis = "The High Court granted bail to the accused."
        
        result = verify_faithfulness(premise, hypothesis)
        assert result == "UNSUPPORTED"
        
        # 10-second sleep to avoid rate limits on free tier
        time.sleep(10)

class TestEntityChecker:
    def test_entity_checker_clean(self):
        source = "The accused was charged under Section 302 of the IPC."
        generated = "The accused was charged under Section 302."
        
        hallucinated = verify_hard_entities(source, generated)
        assert len(hallucinated) == 0
        
    def test_entity_checker_hallucination(self):
        source = "The accused was charged under Section 302 of the IPC."
        generated = "The accused was charged under Section 302 and Section 420."
        
        hallucinated = verify_hard_entities(source, generated)
        assert len(hallucinated) == 1
        assert "Section 420" in hallucinated
