import os
import time
import psutil
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


def run_rag_benchmark(embedding_model_name: str, document_path: str):
    """Runs a RAG benchmark for a given embedding model.

    Args:
        embedding_model_name: The name of the Hugging Face embedding model to use.
        document_path: The path to the document to load.
    """
    print("-" * 80)
    print(f"[*] Testing model: {embedding_model_name}")
    
    process = psutil.Process(os.getpid())

    # --- 1. Load and Split Document ---
    loader = TextLoader(document_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # --- 2. Initialize Embedding Model (CPU) ---
    # Using device='cpu' is crucial for CPU-only performance measurement.
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # --- 3. Measure Embedding and Indexing Time ---
    print("[*] Generating embeddings...")
    mem_before_embedding = process.memory_info().rss
    cpu_before_embedding = process.cpu_times()
    start_time = time.perf_counter()
    
    doc_embeddings = embeddings.embed_documents([d.page_content for d in docs])
    
    end_time = time.perf_counter()
    mem_after_embedding = process.memory_info().rss
    cpu_after_embedding = process.cpu_times()

    embedding_time = end_time - start_time
    embedding_mem_usage = (mem_after_embedding - mem_before_embedding) / (1024 * 1024)  # RSS in MB
    embedding_cpu_time = (cpu_after_embedding.user - cpu_before_embedding.user) + \
                         (cpu_after_embedding.system - cpu_before_embedding.system)
    print(f"[*] Done in {embedding_time:.4f} seconds.")

    print("[*] Building FAISS index...")
    text_embedding_pairs = zip([d.page_content for d in docs], doc_embeddings)
    
    mem_before_indexing = process.memory_info().rss
    cpu_before_indexing = process.cpu_times()
    start_time = time.perf_counter()

    vectorstore = FAISS.from_embeddings(text_embedding_pairs, embeddings)
    
    end_time = time.perf_counter()
    mem_after_indexing = process.memory_info().rss
    cpu_after_indexing = process.cpu_times()

    indexing_time = end_time - start_time
    indexing_mem_usage = (mem_after_indexing - mem_before_indexing) / (1024 * 1024)
    indexing_cpu_time = (cpu_after_indexing.user - cpu_before_indexing.user) + \
                        (cpu_after_indexing.system - cpu_before_indexing.system)
    print(f"[*] Done in {indexing_time:.4f} seconds.")

    # --- 4. Initialize LLM and RAG Chain ---
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    retriever = vectorstore.as_retriever()

    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- 5. Measure Query and Retrieval Time ---
    query = "How does QuantumOS handle Inter-Process Communication?"
    print(f"[*] Performing query: '{query}'")
    
    mem_before_query = process.memory_info().rss
    cpu_before_query = process.cpu_times()
    start_time = time.perf_counter()
    
    result = qa_chain.invoke(query)
    
    end_time = time.perf_counter()
    mem_after_query = process.memory_info().rss
    cpu_after_query = process.cpu_times()

    query_time = end_time - start_time
    query_mem_usage = (mem_after_query - mem_before_query) / (1024 * 1024)
    query_cpu_time = (cpu_after_query.user - cpu_before_query.user) + \
                     (cpu_after_query.system - cpu_before_query.system)
    print(f"[*] Done in {query_time:.4f} seconds.")

    # --- 6. Print Results ---
    print("\n--- Performance ---")
    print(f"Embedding Wall Time:     {embedding_time:.4f} seconds")
    print(f"Embedding CPU Time:      {embedding_cpu_time:.4f} seconds")
    print(f"Embedding Memory Usage:  {embedding_mem_usage:.2f} MB")
    print("-" * 20)
    print(f"Indexing Wall Time:      {indexing_time:.4f} seconds")
    print(f"Indexing CPU Time:       {indexing_cpu_time:.4f} seconds")
    print(f"Indexing Memory Usage:   {indexing_mem_usage:.2f} MB")
    print("-" * 20)
    print(f"Query Wall Time:         {query_time:.4f} seconds")
    print(f"Query CPU Time:          {query_cpu_time:.4f} seconds")
    print(f"Query Memory Usage:      {query_mem_usage:.2f} MB")
    print("\n--- Answer ---")
    print(result)
    print("-" * 80)


def main():
    """Main function to run the benchmark."""
    # Load Google API Key from .env file
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print(
            "Please create a .env file and add your key, e.g., GOOGLE_API_KEY='your_key_here'"
        )
        return

    document_path = "very_large_document.txt"
    if not os.path.exists(document_path):
        print(f"Error: Document not found at {document_path}")
        return

    models_to_test = [
        "MongoDB/mdbr-leaf-ir",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]

    for model in models_to_test:
        run_rag_benchmark(model, document_path)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
