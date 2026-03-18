import os
import hashlib
import time
from dotenv import load_dotenv
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from google import genai

# Load environment variables
load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")

def create_pinecone_index():
    """Create Pinecone index with correct dimensions (768 for gemini-embedding-001)"""
    try:
        if index_name not in pc.list_indexes().names():
            print(f"Creating index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=768,  # Correct dimension for gemini-embedding-001
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("Waiting for index to initialize...")
            time.sleep(30)
        return True
    except Exception as e:
        print(f"Error creating index: {e}")
        return False

def read_pdf(filepath):
    """Read PDF file and extract text"""
    try:
        reader = PdfReader(filepath)
        text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n\n[Page {page_num + 1}]\n{page_text}"
        return text
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
        return ""

def chunk_text(text, chunk_size=800):
    """Split text into chunks"""
    chunks = []
    pages = text.split("\n\n[Page ")
    
    for page in pages:
        if not page.strip():
            continue
            
        if not page.startswith("[Page") and pages.index(page) > 0:
            page = "[Page " + page
            
        words = page.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                overlap_words_count = max(1, int(len(current_chunk) * 0.2))
                current_chunk = current_chunk[-overlap_words_count:]
                current_size = sum(len(w) + 1 for w in current_chunk)
        
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
    
    return chunks

def get_embedding(text, retry_count=3):
    """Generate embedding with retry logic for rate limits"""
    for attempt in range(retry_count):
        try:
            response = client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 60  # Wait 60 seconds for quota to reset
                print(f"\n⚠️ Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"\n⚠️ Error generating embedding (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(5)
                else:
                    return None
    return None

def process_pdfs():
    """Process all PDFs in data folder"""
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Created {data_folder} folder. Please add your PDF files.")
        return [], []
    
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {data_folder} folder.")
        return [], []
    
    all_chunks = []
    chunk_metadata = []
    
    for pdf_file in pdf_files:
        filepath = os.path.join(data_folder, pdf_file)
        print(f"\nProcessing: {pdf_file}")
        
        text = read_pdf(filepath)
        if not text:
            continue
        
        source_name = pdf_file.replace('.pdf', '').replace('_', ' ').title()
        chunks = chunk_text(text, chunk_size=int(os.getenv("CHUNK_SIZE", 800)))
        
        print(f"  Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{source_name}_{i}_{chunk[:50]}".encode()).hexdigest()[:16]
            all_chunks.append(chunk)
            chunk_metadata.append({
                "id": chunk_id,
                "source": source_name,
                "filename": pdf_file,
                "chunk_index": i
            })
    
    return all_chunks, chunk_metadata

def main():
    print("=" * 60)
    print("Healthcare Knowledge Base Ingestion")
    print("=" * 60)
    
    print("\n1. Setting up Pinecone with 768 dimensions...")
    if not create_pinecone_index():
        print("Failed to setup Pinecone. Exiting.")
        return
    
    index = pc.Index(index_name)
    
    print("\n2. Processing PDF files...")
    chunks, metadata = process_pdfs()
    
    if not chunks:
        print("No content to process. Please add PDF files to the 'data' folder.")
        return
    
    print(f"\n3. Generating embeddings (this may take a while due to rate limits)...")
    print(f"   Free tier allows 100 requests per minute")
    print(f"   Total chunks: {len(chunks)}")
    
    batch_size = int(os.getenv("BATCH_SIZE", 10))  # Smaller batches
    delay = int(os.getenv("DELAY_BETWEEN_BATCHES", 5))
    successful = 0
    
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch_end = min(i + batch_size, len(chunks))
        batch_chunks = chunks[i:batch_end]
        batch_metadata = metadata[i:batch_end]
        
        vectors = []
        
        for j, (chunk, meta) in enumerate(zip(batch_chunks, batch_metadata)):
            print(f"\n   Generating embedding {i + j + 1}/{len(chunks)}...")
            embedding = get_embedding(chunk)
            
            if embedding:
                vectors.append({
                    "id": meta["id"],
                    "values": embedding,
                    "metadata": {
                        "text": chunk[:1500],
                        "source": meta["source"],
                        "filename": meta["filename"]
                    }
                })
                successful += 1
        
        if vectors:
            try:
                index.upsert(vectors=vectors)
                print(f"\n✅ Uploaded batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            except Exception as e:
                print(f"\n❌ Error uploading batch: {e}")
        
        # Delay between batches to respect rate limits
        if i + batch_size < len(chunks):
            print(f"   Waiting {delay} seconds before next batch...")
            time.sleep(delay)
    
    print(f"\n✅ Ingestion complete!")
    print(f"   Successfully processed: {successful}/{len(chunks)} chunks")
    
    if successful > 0:
        stats = index.describe_index_stats()
        print(f"\n📊 Index Statistics:")
        print(f"   Total vectors: {stats.total_vector_count}")
        print(f"   Dimension: {stats.dimension}")
    else:
        print("\n❌ No vectors were successfully uploaded. Check your API keys and model access.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()