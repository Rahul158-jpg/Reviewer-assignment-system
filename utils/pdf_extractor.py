# ==========================================
# Reviewer Recommendation — PDF Extractor (Robust)
# ==========================================

import fitz  # PyMuPDF
import re


# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    """
    Clean text while preserving paragraph/newline structure.

    - Normalize CRLF to LF
    - Collapse multiple spaces/tabs but keep newlines
    - Collapse many newlines into at most two
    """
    if not text:
        return ""

    # normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # collapse repeated spaces/tabs but preserve newlines
    text = re.sub(r'[ \t]+', ' ', text)

    # collapse multiple blank lines to at most two
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    return text.strip()


# -------------------------------
# EXTRACT METADATA
# -------------------------------
def extract_pdf_metadata(pdf_path):
    """
    Extract:
    ✔ Title
    ✔ Authors
    ✔ Abstract

    Strategy:
    - Read first 2 pages
    - Use heuristics (not perfect, but stable)
    """

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ Failed to open PDF: {e}")
        return None

    # read first 3 pages (metadata usually in first 2-3 pages)
    parts = []
    for page_num in range(min(3, len(doc))):
        page = doc[page_num]
        parts.append(page.get_text("text") or "")

    full_text = "\n\n".join(parts)
    full_text = clean_text(full_text)

    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]

    # -------------------------------
    # Use page layout + font sizes to detect title/authors/abstract
    # -------------------------------
    def _page_blocks(page):
        data = page.get_text("dict")
        blocks = []
        for b in data.get("blocks", []):
            if b.get("type") != 0:
                continue
            block_text_parts = []
            sizes = []
            bbox = b.get("bbox", [0, 0, 0, 0])
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = span.get("size", 0)
                    if text:
                        block_text_parts.append(text)
                        sizes.append(float(size))
            if block_text_parts:
                block_text = " ".join(block_text_parts).strip()
                max_size = max(sizes) if sizes else 0
                avg_size = sum(sizes) / len(sizes) if sizes else 0
                blocks.append({
                    "text": block_text,
                    "max_size": max_size,
                    "avg_size": avg_size,
                    "bbox": bbox,
                    "y0": bbox[1],
                    "y1": bbox[3]
                })
        return blocks

    page0 = doc[0]
    page_blocks = _page_blocks(page0)
    page_blocks = sorted(page_blocks, key=lambda b: b["y0"])

    # skip keywords for non-title blocks
    skip_keywords = ['received', 'revised', 'accepted', 'published', 'doi', 'copyright', '©', 'journal', 'research', 'published online', 'preprint']

    title = "Unknown Title"
    authors = []
    abstract = ""

    title_block = None
    if page_blocks:
        # prefer the block with largest font size that doesn't look like metadata
        cand_blocks = sorted(page_blocks, key=lambda b: b['max_size'], reverse=True)
        for b in cand_blocks:
            low = b['text'].lower()
            if any(k in low for k in skip_keywords):
                continue
            if len(b['text'].split()) >= 3 and len(b['text']) > 20 and not low.startswith('abstract'):
                title_block = b
                break

        if not title_block:
            title_block = cand_blocks[0]

        title = title_block['text']

        # find abstract block (if any) by searching for a block that contains 'abstract'
        abstract_block = None
        for b in page_blocks:
            low = b['text'].lower()
            if low.startswith('abstract') or low.startswith('summary') or re.match(r'^abstract[:.\-\s]', low):
                abstract_block = b
                break

        # authors are blocks between title_block and abstract_block (or the next few blocks)
        authors_blocks = []
        # find index of title_block in page_blocks
        idx = None
        for i, b in enumerate(page_blocks):
            if b is title_block or (abs(b['y0'] - title_block['y0']) < 1 and abs(b['y1'] - title_block['y1']) < 1):
                idx = i
                break

        if idx is not None:
            j = idx + 1
            while j < len(page_blocks):
                b = page_blocks[j]
                if abstract_block and b['y0'] >= abstract_block['y0']:
                    break
                # stop collecting if we hit a clear section header
                if re.match(r'^(introduction|keywords|index terms)\b', b['text'].strip().lower()):
                    break
                authors_blocks.append(b)
                # limit to first 4 blocks
                if len(authors_blocks) >= 4:
                    break
                j += 1

        author_block_text = ' '.join([b['text'] for b in authors_blocks]).strip()

        # clean author block
        author_block_text = re.sub(r'\S+@\S+', '', author_block_text)
        author_block_text = re.sub(r'\[\d+\]', '', author_block_text)
        author_block_text = re.sub(r'\s{2,}', ' ', author_block_text).strip()

        # attempt to extract names using capitalization patterns
        name_pattern = r"\b[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿA-Z'\-]+(?:\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿA-Z'\-]+){0,4}\b"
        candidates = re.findall(name_pattern, author_block_text)
        bad_keywords = ['university', 'department', 'laboratory', 'institute', 'journal', 'doi', 'http', 'via', 'published', 'author', 'cagliari', 'paris', 'italy', 'france', 'email']
        filtered = []
        for c in candidates:
            lowc = c.lower()
            if any(k in lowc for k in bad_keywords):
                continue
            if len(c.split()) == 1:
                continue
            if c not in filtered:
                filtered.append(c.strip())

        if filtered:
            authors = filtered
        else:
            # fallback to splitting heuristics
            parts = re.split(r',|;|\band\b| & |\||/|\\', author_block_text)
            cleaned = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                low = p.lower()
                if any(k in low for k in bad_keywords):
                    continue
                if len(p) < 3 or re.search(r'\d', p):
                    continue
                cleaned.append(re.sub(r'\d+', '', p).strip())

            if cleaned:
                authors = cleaned

        # extract abstract using block-level detection if available
        if abstract_block:
            abs_parts = []
            # collect blocks after abstract_block until next section header
            start_idx = None
            for i, b in enumerate(page_blocks):
                if b is abstract_block:
                    start_idx = i
                    break
            if start_idx is not None:
                for k in range(start_idx + 1, len(page_blocks)):
                    tb = page_blocks[k]
                    low = tb['text'].lower()
                    if re.match(r'^(keywords|index terms|introduction)\b', low):
                        break
                    abs_parts.append(tb['text'])
            abstract = ' '.join(abs_parts).strip()

    # fallback behaviors if block-level extraction failed
    if not abstract:
        # search in raw lines for abstract heading
        abstract_idx = None
        for i, line in enumerate(lines):
            low = line.lower()
            if low.startswith('abstract') or low.startswith('summary') or re.match(r'^abstract[:.\-\s]', low):
                abstract_idx = i
                break

        if abstract_idx is not None:
            abs_lines = []
            for j in range(abstract_idx + 1, min(len(lines), abstract_idx + 60)):
                ln = lines[j]
                low = ln.lower()
                if low.startswith('keywords') or low.startswith('index terms') or low.startswith('introduction') or re.match(r'^\d+\.|^1\.', low):
                    break
                abs_lines.append(ln)
            abstract = ' '.join(abs_lines).strip()

    # last resort fallback: chunk after first title-like line
    if not abstract and lines:
        # try to take lines after the detected title text
        for i, ln in enumerate(lines[:12]):
            if title and title in ln:
                cand = []
                for j in range(i + 1, min(len(lines), i + 20)):
                    if '@' in lines[j] or 'university' in lines[j].lower():
                        continue
                    cand.append(lines[j])
                    if len(' '.join(cand)) > 200:
                        break
                abstract = ' '.join(cand).strip()
                break

    # final cleaning
    title = clean_text(title)
    abstract = clean_text(abstract)

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract
    }


# -------------------------------
# DEBUG RUN
# -------------------------------
if __name__ == "__main__":

    test_pdf = "test.pdf"  # change this

    result = extract_pdf_metadata(test_pdf)

    print("\n=== PDF METADATA ===")
    print("Title:", result["title"])
    print("Authors:", result["authors"])
    print("Abstract:", result["abstract"][:300])