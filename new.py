# Run this in Python console
from pinecone import Pinecone
pc = Pinecone(api_key="pcsk_yiHTz_Tycrcn1NQXghbsK4z9ascPQpx9wwoeFkAyJ79vM745vnGrs7NMJdVRnGU6w3b1r")
pc.delete_index("healthcare-index")