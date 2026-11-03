"""Simple Phase 3 evaluation: retrieval Hit@3, abstention and latency."""
from pathlib import Path
import csv
import json

from rag import ROOT, answer_question


def main():
    cases = json.loads((ROOT / 'eval_cases.json').read_text(encoding='utf-8'))
    rows = []
    for case in cases:
        result = answer_question(case['question'])
        retrieved = {s['source'] for s in result['sources']}
        hit = case['expected_source'] in retrieved if case.get('expected_source') else None
        abstained = "i don't know" in result['answer'].lower() or 'insufficient' in result['answer'].lower()
        row = {
            'question': case['question'],
            'expected_source': case.get('expected_source') or '',
            'hit_at_3': hit if hit is not None else '',
            'expected_abstention': case.get('should_abstain', False),
            'abstained': abstained,
            'latency_seconds': result['latency_seconds'],
            'answer': result['answer'],
        }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'answer'}))
    output = ROOT / 'evaluation_results.csv'
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    known = [r for r in rows if isinstance(r['hit_at_3'], bool)]
    if known:
        print(f"Retrieval Hit@3: {sum(r['hit_at_3'] for r in known)}/{len(known)}")
    print(f'Saved {output.name} (review answer correctness manually)')


if __name__ == '__main__':
    main()
