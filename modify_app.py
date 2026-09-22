import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Imports
content = content.replace(
    "from src.guardrails.entity_checker import check_entities",
    "from src.guardrails.entity_checker import verify_hard_entities"
)

# 2. Logic change
old_logic = """                        verification = verify_faithfulness(source_text, result.answer)
                        entity_check = check_entities(
                            result.answer,
                            source_text,
                        )
                    else:
                        verification = None
                        entity_check = None

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result.answer,
                        "source_chunks": [c.model_dump() for c in result.source_chunks],
                        "confidence": result.confidence,
                        "verification": verification.model_dump() if verification else None,
                        "entity_check": entity_check.model_dump() if entity_check else None,
                    })"""
new_logic = """                        verification = verify_faithfulness(source_text, result.answer)
                        hallucinated_entities = verify_hard_entities(source_text, result.answer)
                    else:
                        verification = None
                        hallucinated_entities = []

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result.answer,
                        "source_chunks": [c.model_dump() for c in result.source_chunks],
                        "confidence": result.confidence,
                        "verification": verification,
                        "hallucinated_entities": hallucinated_entities,
                    })"""
content = content.replace(old_logic, new_logic)

# 3. UI Rendering change
old_ui = """                verification = msg.get("verification")
                entity_check = msg.get("entity_check")

                faith_color = get_faith_color(
                    verification["overall_score"] if verification else confidence
                )
                faith_score = verification["overall_score"] if verification else confidence
                faith_verdict = verification["overall_verdict"] if verification else "N/A"

                st.markdown(f\"\"\"
                <div class="chat-msg chat-bot">
                    <strong>⚖️ LexQuery</strong><br>
                    {answer}
                    <div style="margin-top: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 0.78rem; color: #666;">
                            Faithfulness: <strong style="color: {faith_color};">{faith_score:.0%}</strong>
                            ({faith_verdict})
                        </span>
                    </div>
                    <div class="faith-meter">
                        <div class="faith-fill" style="width: {faith_score*100}%; background: {faith_color};"></div>
                    </div>
                </div>
                \"\"\", unsafe_allow_html=True)"""
new_ui = """                verification = msg.get("verification")
                hallucinated_entities = msg.get("hallucinated_entities", [])

                if verification == "FAITHFUL":
                    faith_color = "#2e7d32"
                    faith_verdict = "FAITHFUL"
                elif verification == "UNSUPPORTED":
                    faith_color = "#c62828"
                    faith_verdict = "UNSUPPORTED"
                else:
                    faith_color = "#ff9800"
                    faith_verdict = "UNKNOWN"

                st.markdown(f\"\"\"
                <div class="chat-msg chat-bot">
                    <strong>⚖️ LexQuery</strong><br>
                    {answer}
                    <div style="margin-top: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
                        <span style="font-size: 0.78rem; color: #666;">
                            Faithfulness: <strong style="color: {faith_color};">{faith_verdict}</strong>
                        </span>
                    </div>
                </div>
                \"\"\", unsafe_allow_html=True)"""
content = content.replace(old_ui, new_ui)

# 4. Entity Checks & Per sentence verifications (Remove per-sentence and update entities)
old_entities_and_sentences = """                # Entity check results
                if entity_check and entity_check.get("total_entities", 0) > 0:
                    verdict = entity_check.get("verdict", "CLEAN")
                    if verdict == "HALLUCINATION_DETECTED":
                        st.warning(f"⚠️ Entity check found {entity_check['hallucinated_count']} "
                                   f"potentially hallucinated legal references")
                    else:
                        st.success(f"✅ All {entity_check['verified_count']} legal references verified")

                # Per-sentence verification
                if verification and verification.get("sentence_verdicts"):
                    with st.expander("🔬 Per-Sentence Faithfulness Analysis"):
                        for sv in verification["sentence_verdicts"]:
                            verdict = sv["verdict"]
                            icon = "✅" if verdict == "SUPPORTED" else "⚠️" if verdict == "UNSUPPORTED" else "❌"
                            st.markdown(f"{icon} **{verdict}** (entailment: {sv['entailment_score']:.2f}) — {sv['sentence']}")"""

new_entities = """                # Entity check results
                if hallucinated_entities:
                    st.warning(f"⚠️ Entity check found {len(hallucinated_entities)} "
                               f"potentially hallucinated legal references: {', '.join(hallucinated_entities)}")
                elif verification is not None:
                    st.success(f"✅ All legal references verified against source")"""
content = content.replace(old_entities_and_sentences, new_entities)

content = content.replace("Powered by Google Gemini, bge-m3, DeBERTa NLI", "Powered by Google Gemini, bge-m3 (Local)")

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Modifications done!")
