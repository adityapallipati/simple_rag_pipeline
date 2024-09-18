# app.py

import streamlit as st
import openai
import random
import tempfile
from chunking import load_pdf, chunk_text
from embedding import embed_text, search_text_rank_with_query

# 1. User API Key Input
st.title("RAG-Powered Chatbot")
api_key = st.text_input("Enter your OpenAI API Key:", type="password")
openai.api_key = api_key  # Use the user's API key

# 2. PDF Upload and Processing
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
if uploaded_files:
    pdf_texts = []
    for pdf_file in uploaded_files:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_file.getbuffer())  # Write the file content to temp file
            temp_pdf_path = temp_pdf.name  # Get the temporary file path

        # Load and chunk PDF using the `chunking.py` module
        pages = load_pdf(temp_pdf_path)
        chunks = chunk_text(pages)

        # Debug: Show a random sample of the chunks (limit to 3)
        random_chunk_sample = random.sample(chunks, min(3, len(chunks)))
        st.write("Random sample of chunks:", random_chunk_sample)

    # 4. Embed the chunks using the `embedding.py` module
    embeddings, embedding_model, filtered_chunks = embed_text(chunks)

    st.success("PDFs uploaded and processed. Ready for query!")

# 5. User Query and Retrieval Using TextRank and Query Similarity
query = st.text_input("Enter your query:")
if query:
    # Embed the query
    query_embedding = embedding_model.encode(query)
    
    # Retrieve relevant chunks using TextRank and query similarity re-ranking
    top_chunks = search_text_rank_with_query(embeddings, query_embedding, filtered_chunks, top_k=5)

    # Debugging: Show a random sample of relevant chunks selected (limit to 2)
    random_relevant_chunks = random.sample(top_chunks, min(2, len(top_chunks)))
    st.write("Random sample of relevant chunks selected:", random_relevant_chunks)

    # 6. Construct final prompt with relevant chunks and format it
    separator = "\n\n---\n\n"  # Separator to delineate different chunks
    final_prompt = f"User Query: {query}\n\nContext from Document:\n\n"
    final_prompt += separator.join(top_chunks)

    # Display the final prompt with markdown for readability
    st.markdown(f"### Final Prompt\n```\n{final_prompt}\n```")

    # 7. Optionally, use the OpenAI API to generate a response (commented out for now)
    # if api_key:
    #     response = openai.Completion.create(
    #         engine="text-davinci-003",
    #         prompt=final_prompt,
    #         max_tokens=150
    #     )
    #     st.write(response.choices[0].text.strip())