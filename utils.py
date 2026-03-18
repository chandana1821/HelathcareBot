import os
import re
from typing import List, Dict
from pinecone import Pinecone

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove multiple newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Remove extra spaces
    text = re.sub(r' +', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\-\(\)]', '', text)
    return text.strip()

def extract_section(text: str, section_keywords: List[str]) -> Dict[str, str]:
    """Extract sections like symptoms, treatments from text"""
    sections = {}
    
    for keyword in section_keywords:
        # Look for sections with common patterns
        patterns = [
            rf"{keyword}\s*:?\s*([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\Z)",
            rf"{keyword}\s*\n([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\Z)",
            rf"{keyword}\s+([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\Z)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                sections[keyword.lower()] = match.group(1).strip()
                break
    
    return sections

def check_pinecone_index():
    """Check if Pinecone index exists and has data"""
    from dotenv import load_dotenv
    load_dotenv()
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    try:
        if index_name not in pc.list_indexes().names():
            print(f"❌ Index '{index_name}' does not exist")
            print("   Run 'python ingest.py' to create and populate the index")
            return False
        
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        
        print(f"\n📊 Index Statistics:")
        print(f"  • Name: {index_name}")
        print(f"  • Total vectors: {stats.total_vector_count}")
        print(f"  • Dimension: {stats.dimension}")
        
        if stats.total_vector_count == 0:
            print("  ⚠️ Index is empty. Run 'python ingest.py' to add documents")
            return False
        
        print("  ✅ Index is ready for queries")
        return True
        
    except Exception as e:
        print(f"❌ Error checking index: {e}")
        return False

def test_sample_queries():
    """Test some sample queries"""
    from query import search_similar_chunks
    
    test_questions = [
        "What are the symptoms of diabetes?",
        "How to treat high blood pressure?",
        "What should I do if my child has fever?",
        "What is the target blood pressure for hypertension?",
        "How to manage diabetes during pregnancy?",
        "What causes fever in children?",
        "What medications are used for hypertension?"
    ]
    
    print("\n🔬 Testing Sample Queries")
    print("=" * 60)
    
    for q in test_questions:
        print(f"\n❓ Query: {q}")
        matches = search_similar_chunks(q, top_k=3)
        
        if matches:
            print(f"✅ Found {len(matches)} relevant chunks")
            for i, match in enumerate(matches, 1):
                source = match.metadata.get('source', 'Unknown')
                page = match.metadata.get('page', 'Unknown')
                score = match.score
                print(f"   {i}. {source} (Page {page}) - Score: {score:.3f}")
        else:
            print("❌ No results found")
        
        time.sleep(1)  # Small delay between queries

if __name__ == "__main__":
    import time
    
    if check_pinecone_index():
        test_sample_queries()
    else:
        print("\nPlease run 'python ingest.py' first to populate the database.")