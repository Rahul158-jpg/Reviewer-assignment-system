import json
import copy

from assignment import assigner


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def to_list(data):
    if isinstance(data, dict):
        if 'manuscripts' in data:
            return data['manuscripts']
        return list(data.values())
    if isinstance(data, list):
        return data
    return []


def main():
    manuscripts_raw = load_json('data/manuscripts.json')
    reviewers_raw = load_json('data/reviewers.json')

    manuscripts = to_list(manuscripts_raw)
    reviewers = to_list(reviewers_raw)

    # sample up to 30 papers for broader validation
    sample_n = min(30, max(1, len(manuscripts)))
    step = max(1, len(manuscripts)//sample_n)
    indices = list(range(0, len(manuscripts), step))[:sample_n]

    total_reviewers = 0
    total_empty = 0
    paper_reports = []

    for i, idx in enumerate(indices):
        paper = manuscripts[idx]
        title = paper.get('title', '')
        paper_text = (paper.get('title', '') + ' ' + paper.get('abstract', ''))

        # paper keywords
        try:
            paper_kw = assigner.extract_keyphrases(paper_text, top_k=10)
        except Exception:
            paper_kw = []

        print('\n' + '='*60)
        print(f"Paper #{i+1} idx={idx} title={title}")
        print("Paper Keywords:", paper_kw)

        sample_reviewers = reviewers[:min(10, len(reviewers))]
        empty_count = 0
        per_review = []

        for rv in sample_reviewers:
            # include titles + short abstract snippets to improve semantic signals
            rv_text = ''
            for p in rv.get('publications', [])[:5]:
                rv_text += ' ' + (p.get('title', '') or '')
                rv_text += ' ' + (p.get('abstract', '') or '')[:200]
            try:
                reviewer_kw = assigner.extract_keyphrases(rv_text, top_k=10)
            except Exception:
                reviewer_kw = []

            try:
                overlap = assigner.compute_overlap(paper_kw, reviewer_kw)
            except Exception:
                # fallback to compute_topic_overlap if needed
                try:
                    _, overlap = assigner.compute_topic_overlap(paper, rv)
                except Exception:
                    overlap = []

            if not overlap:
                empty_count += 1

            print(f"- Reviewer {rv.get('reviewer_id')} Keywords: {reviewer_kw} | Overlap: {overlap}")
            per_review.append({'id': rv.get('reviewer_id'), 'reviewer_kw': reviewer_kw, 'overlap': overlap})

        fallback_rate = empty_count / len(sample_reviewers) if sample_reviewers else 0
        print(f"Fallback rate (empty overlaps among sample reviewers): {empty_count}/{len(sample_reviewers)} = {fallback_rate:.2%}")

        # run assign_reviewers to inspect reasons
        rv_copy = copy.deepcopy(sample_reviewers)
        try:
            sel = assigner.assign_reviewers(rv_copy, paper)
        except Exception as e:
            sel = []
            print("assign_reviewers() failed:", e)

        print('\nSelected reviewers and reasons:')
        for s in sel:
            print(f"  - {s.get('role')} {s.get('reviewer_id')}: {s.get('reason')}")

        paper_reports.append({'idx': idx, 'paper_kw': paper_kw, 'fallback_rate': fallback_rate, 'per_review': per_review, 'selected': sel})

        total_reviewers += len(sample_reviewers)
        total_empty += empty_count

    overall_fallback = total_empty / total_reviewers if total_reviewers else 0
    print('\n' + '='*60)
    print(f"Overall fallback rate across sampled papers/reviewers: {overall_fallback:.2%}")

    bad = [p for p in paper_reports if p['fallback_rate'] >= 0.5]
    print(f"Papers with >=50% fallback: {len(bad)}")
    for p in bad:
        print(' - idx', p['idx'], 'paper_kw', p['paper_kw'], 'fallback', p['fallback_rate'])


if __name__ == '__main__':
    main()
