from langchain_google_community import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import os

# Step 1: Load documents from Google Drive
loader = GoogleDriveLoader(
    folder_id="1XpoDIsfnMa88mnwOYumFZD_pSckpg4dx",
    token_path="C:/Users/solan/OneDrive/Documents/Desktop/Rag/token.json",
    file_types=["pdf"],
    recursive=False,
)
docs = loader.load()
print(f"✅ Loaded {len(docs)} documents from Google Drive.")

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)
texts = text_splitter.split_documents(docs)
print(f"✅ Split into {len(texts)} text chunks.")

# Step 3: Prefix text for E5 model
for doc in texts:
    doc.page_content = "passage: " + doc.page_content

# Step 4: Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2",
    encode_kwargs={"normalize_embeddings": True}
)
print("✅ Embedding model loaded.")

# Step 5: Initialize Pinecone and create index if needed
api_key = "pcsk_7LgAGW_4vTXc3oC1jetH9MqftRUYzsJmyCRxhZeMaWJ4SFEpozABRxLzuk3MiRk4e5pP6W"  # 🔐 Replace with your actual key
index_name = "my-rag-index"

pc = Pinecone(api_key=api_key)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-central1"
        )
    )
    print(f"✅ Created index '{index_name}'")
else:
    print(f"ℹ️ Index '{index_name}' already exists")

# Step 6: Connect to the index
index = pc.Index(index_name)

# Step 7: Use LangChain PineconeVectorStore
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_model,
    text_key="text"
)

# Step 8: Threaded upload function
def upload_doc(doc, i):
    try:
        vectorstore.add_documents([doc])
        return f"✅ Inserted chunk {i+1}/{len(texts)}"
    except Exception as e:
        return f"⚠️ Skipped chunk {i+1}: {str(e)}"

# Step 9: Use ThreadPoolExecutor for parallel uploads
print("🚀 Uploading documents using threading...")
with ThreadPoolExecutor(max_workers=10) as executor:  # You can adjust the number of threads
    futures = [executor.submit(upload_doc, doc, i) for i, doc in enumerate(texts)]
    for future in as_completed(futures):
        print(future.result())     

print("✅ Vector store created and populated successfully (threaded).")

vectorstore.add_documents(texts)
print("✅ Documents uploaded to Pinecone.")


# Ensure your Gemini API key is configured
os.environ["GOOGLE_API_KEY"] = "AIzaSyAhepG8ma4JPVxxC0hQ8rmiULek8ifXl0Q"


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)


result = qa_chain.invoke("what is sensor system?")
print(result["result"])






