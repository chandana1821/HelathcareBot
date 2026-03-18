import os
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai
import time
import sys

# Suppress warnings and non-error output
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# Initialize Gemini client
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Gemini API key not found")

client = genai.Client(api_key=api_key)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

def silent_print(*args, **kwargs):
    """Override print to suppress non-essential messages"""
    if kwargs.get('force', False):
        original_print(*args, **kwargs)

# Save original print function
original_print = print

def get_embedding(text, retry_count=3):
    """Generate embedding with retry logic - silent mode"""
    for attempt in range(retry_count):
        try:
            response = client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            if "429" in str(e) and attempt < retry_count - 1:
                time.sleep(60)  # Wait silently
            elif attempt < retry_count - 1:
                time.sleep(5)  # Wait silently
            else:
                return None
    return None

def search_documents(query, top_k=4):
    """Search for relevant documents - silent mode"""
    query_embedding = get_embedding(query)
    
    if not query_embedding:
        return []
    
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results.matches
    except Exception:
        return []

def generate_answer(query, context):
    """Generate answer using Gemini - silent mode with fallbacks"""
    if not context:
        return "I couldn't find any relevant information in the documents."
    
    prompt = f"""You are a helpful healthcare assistant. Answer the question using ONLY the information provided in the context below.

Context:
{context}

Question: {query}

Instructions:
- Be concise and accurate
- Only use information from the context
- If the context doesn't contain the answer, say "I cannot find this information in the available documents"
- Include the source of information when possible

Answer:"""
    
    # Try models in order of preference (silently)
    models_to_try = [
        "models/gemini-2.5-flash",           # This worked in your test
        "models/gemini-2.0-flash-lite",       # Higher quota
        "models/gemini-2.0-flash-001",        # Fallback
        "models/gemini-2.0-flash"             # Last resort
    ]
    
    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text
        except Exception:
            # Silently try next model
            continue
    
    return "I couldn't generate an answer at this time. Please try again later."

def format_results(matches):
    """Format search results for display"""
    if not matches:
        return ""
    
    contexts = []
    for i, match in enumerate(matches, 1):
        source = match.metadata.get('source', 'Unknown')
        text = match.metadata.get('text', '')
        score = match.score
        
        if len(text) > 500:
            text = text[:500] + "..."
        
        contexts.append(f"[{i}] Source: {source} (Relevance: {score:.2f})\n{text}\n")
    
    return "\n---\n".join(contexts)

def check_index_status():
    """Check if index has vectors - silent mode"""
    try:
        stats = index.describe_index_stats()
        if stats.total_vector_count == 0:
            return False
        return True
    except Exception:
        return False

def main():
    print("\n" + "=" * 60)
    print("🏥 Healthcare Knowledge Chatbot")
    print("=" * 60)
    
    if not check_index_status():
        print("\n⚠️ Index is empty. Please run 'python ingest.py' first.")
        return
    
    print("\n📚 Available documents:")
    print("   • Diabetes Management")
    print("   • Hypertension Guidelines")
    print("   • Fever in Children")
    
    print("\nType 'exit' to quit\n")
    
    while True:
        query = input("\n❓ Your question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'bye']:
            print("\n👋 Goodbye! Stay healthy!")
            break
        
        if not query:
            continue
        
        # Show searching indicator
        print("\n🔍 Searching...", end="", flush=True)
        
        matches = search_documents(query, top_k=int(os.getenv("TOP_K_RESULTS", 4)))
        
        if not matches:
            print("\r❌ No relevant information found.\n")
            continue
        
        context = format_results(matches)
        
        print("\r💭 Generating answer...", end="", flush=True)
        answer = generate_answer(query, context)
        
        # Clear the status line
        print("\r" + " " * 50 + "\r", end="", flush=True)
        
        print("\n" + "-" * 50)
        print(answer)
        print("-" * 50)
        
        print("\n📚 Sources:")
        sources = set()
        for match in matches[:3]:
            source = match.metadata.get('source', 'Unknown')
            if source not in sources:
                print(f"  • {source}")
                sources.add(source)
        
        # Add a small delay between queries to avoid rate limits (silent)
        time.sleep(3)

if __name__ == "__main__":
    main()
