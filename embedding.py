# embedding.py

import numpy as np
import random
from sentence_transformers import SentenceTransformer
import faiss
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher  # For deduplication

def embed_text(chunks):
    """Embed the text chunks using a sentence transformer model."""
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Filter chunks that have fewer than 30 tokens (words)
    filtered_chunks = [chunk for chunk in chunks if len(chunk.split()) >= 30]
    
    # Deduplicate chunks based on similarity threshold
    unique_chunks = []
    for chunk in filtered_chunks:
        if not any(SequenceMatcher(None, chunk, existing).ratio() > 0.85 for existing in unique_chunks):
            unique_chunks.append(chunk)
    
    embeddings = [embedding_model.encode(chunk) for chunk in unique_chunks]
    
    return np.array(embeddings), embedding_model, unique_chunks

def text_rank_embeddings(embeddings, damping=0.85, max_iter=100):
    """
    Compute TextRank scores for chunk embeddings based on cosine similarity.

    Parameters:
    - embeddings: A list of chunk embeddings (2D numpy array).
    - damping: Damping factor for PageRank (default=0.85).
    - max_iter: Maximum number of iterations for PageRank (default=100).

    Returns:
    - ranked_chunks: List of chunk indices sorted by importance.
    """
    # Compute cosine similarity matrix between all chunk embeddings
    similarity_matrix = cosine_similarity(embeddings)

    # Build a graph where each node is a chunk, and edges represent cosine similarity
    graph = nx.from_numpy_array(similarity_matrix)

    # Run PageRank on the graph with cosine similarity as the edge weight
    scores = nx.pagerank(graph, alpha=damping, max_iter=max_iter)

    # Sort chunks by their PageRank scores
    ranked_chunks = sorted(scores, key=scores.get, reverse=True)
    
    return ranked_chunks

def search_text_rank_with_query(embeddings, query_embedding, filtered_chunks, top_k=5):
    """
    Perform retrieval using TextRank and query re-ranking to return the top_k most relevant chunks.
    
    Parameters:
    - embeddings: 2D numpy array of chunk embeddings
    - query_embedding: Embedding of the query
    - filtered_chunks: List of filtered text chunks
    - top_k: Number of top chunks to retrieve
    
    Returns:
    - top_chunks: List of top_k relevant chunks
    """
    # Step 1: Use TextRank to get an initial ranking of chunks based on their importance
    ranked_chunk_indices = text_rank_embeddings(embeddings)
    
    # Retrieve the top-ranked chunks from TextRank
    initial_top_chunks = [filtered_chunks[idx] for idx in ranked_chunk_indices[:top_k]]
    initial_top_embeddings = np.array([embeddings[idx] for idx in ranked_chunk_indices[:top_k]])
    
    # Step 2: Re-rank the top chunks by their cosine similarity to the query
    query_similarities = cosine_similarity([query_embedding], initial_top_embeddings)[0]
    ranked_by_similarity = np.argsort(query_similarities)[::-1]  # Sort in descending order
    
    # Retrieve the final top chunks after re-ranking by query similarity
    top_chunks = [initial_top_chunks[idx] for idx in ranked_by_similarity[:top_k]]
    
    return top_chunks