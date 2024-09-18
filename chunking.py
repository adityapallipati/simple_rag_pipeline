# chunking.py

import fitz  # PyMuPDF for loading PDF
from spacy.lang.en import English  # Sentence-based chunking

def load_pdf(pdf_path):
    """Load the PDF and return text from each page."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page_num": page_num, "text": text})
    return pages

def chunk_text(pages):
    """Chunk the text from the PDF pages using a sentence-based splitter."""
    nlp = English()
    nlp.add_pipe("sentencizer")
    
    chunks = []
    for page in pages:
        doc = nlp(page['text'])
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Group sentences into chunks of roughly 5 sentences to avoid cutting off context
        chunk_size = 5
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i:i + chunk_size])
            chunks.append(chunk)

    return chunks