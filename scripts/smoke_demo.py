# scripts/smoke_demo.py
import sys, json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.integrated_system import ShoppingAssistant

def main():
    assistant = ShoppingAssistant()
    tests = [
        "Looking for a sporty jacket under $100",
        "Show me formal shoes around 150 dollars",
        "I want vintage style dresses",
        "Casual boots between $50 and $100",
    ]
    for q in tests:
        r = assistant.process_query(q, top_n=5)
        print("Q>", q)
        print("R>", json.dumps(r, ensure_ascii=False, indent=2))
        print("T>", assistant.generate_response(r))
        print("="*60)

if __name__ == "__main__":
    main()
