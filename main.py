from modulefinder import test
from engine import OASISRuleEngine

def main():
    print("Hello from oasis-rule-engine!")
    engine = OASISRuleEngine()
    qna = {"question": "How often do you need to have someone help you when you read instructions, pamphlets, or other written material from your doctor or pharmacy?", "answer": 3, "context": {"patient_uses": "Speaker 0: How often do you need to have someone help you read instructions from your doctor?, Speaker 1: Well, I don't know. As often as I can get. Okay., Speaker 0: Do you usually have one of your children go with you to your doctor's appointments?, Speaker 1: Recently. Yes."}}

    result = engine.validate_and_modify(
        question=qna["question"],
        answer=qna["answer"],
        context=qna["context"]
    )
    print(f"  Question: {qna['question']}")
    if result.detected_code:
        print(f"  Detected Code: {result.detected_code} ({result.detected_question})")
    print(f"  Original Answer: {result.original_answer}")
    print(f"  Modified Answer: {result.modified_answer}")
    print(f"  Rule Applied: {result.rule_applied}")
    print(f"  Confidence: {result.confidence:.0%}")
    if result.warnings:
        print(f"  Warnings: {', '.join(result.warnings)}")

if __name__ == "__main__":
    main()
