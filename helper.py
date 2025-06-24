from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "rag-local"  # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)
embedding_model = OllamaEmbeddings(
    model="mxbai-embed-large:335m"
)
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)


def load_document(files):
    document = []
    for file in files:
        print("files being uploaded", file)
        loader = PyPDFLoader(file)
        print("Phase 1")
        document.extend(loader.load())
    
    return chunk_documents(document)
    
def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 400, chunk_overlap = 80)
    split_documents = text_splitter.split_documents(docs)
    print("Phase 2")
    return embed_document(split_documents)


def embed_document(docs):
    print("Phase 2.5")
    vector_store.add_documents(docs)
    print("Phase 3")
    return "Docs Embedding Done"


def retrieve_documents(query):
    # Get raw Pinecone results with scores
    results = index.query(
        vector=embedding_model.embed_query(query),
        top_k=10,
        include_values=False,
        include_metadata=True,
        include_score=True  # <<< THIS IS CRUCIAL
    )
    
  
    docs = []
    for match in results.matches:
        doc = Document(
            page_content=match.metadata.get("text", ""),
            metadata={
                **match.metadata,
                "_score": match.score  # Actual cosine similarity score
            }
        )
        docs.append(doc)
    
    # Filter by score threshold (now works!)
    filtered_docs = [doc for doc in docs if doc.metadata["_score"] >= 0.7]
    return filtered_docs