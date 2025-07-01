import os
import gradio as gr
from langchain_google_community import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
import uuid

# Pinecone & Google Drive configs
PINECONE_API_KEY = "pcsk_7LgAGW_4vTXc3oC1jetH9MqftRUYzsJmyCRxhZeMaWJ4SFEpozABRxLzuk3MiRk4e5pP6W"
GOOGLE_DRIVE_TOKEN_PATH = "C:/Users/solan/OneDrive/Documents/Desktop/Rag/token.json"

# Chat history in memory
chat_histories = {}

# 🔄 RAG pipeline
def rag_pipeline(user_id, folder_id, llm_choice, ollama_url, gemini_api_key, query):
    try:
        # 1. Load Google Drive PDFs
        loader = GoogleDriveLoader(
            folder_id=folder_id,
            token_path=GOOGLE_DRIVE_TOKEN_PATH,
            file_types=["pdf"],
            recursive=False,
        )
        docs = loader.load()
        if not docs:
            return "⚠️ No documents found.", ""

        # 2. Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        texts = splitter.split_documents(docs)
        for doc in texts:
            doc.page_content = "passage: " + doc.page_content

        # 3. Embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2",
            encode_kwargs={"normalize_embeddings": True}
        )

        # 4. Vector DB - Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = user_id.strip().lower()

        if len(index_name) > 45:
            return "❌ Error: Index name (user_id) too long (max 45 characters).", ""

        existing_indexes = pc.list_indexes().names()
        if index_name not in existing_indexes:
            if len(existing_indexes) >= 5:
                return "❌ Pinecone index limit reached (5). Delete old indexes or upgrade your plan.", ""
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        index = pc.Index(index_name)

        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        vectorstore.add_documents(texts)

        # 5. Choose LLM
        if llm_choice == "gemini-1.5-flash":
            if not gemini_api_key:
                return "❌ Gemini API Key is required.", ""
            os.environ["GOOGLE_API_KEY"] = gemini_api_key.strip()
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=gemini_api_key.strip()
            )
        elif llm_choice == "ollama":
            ollama_url = ollama_url.strip() or "http://localhost:11434"
            llm = ChatOllama(model="llama3", base_url=ollama_url)
        else:
            return "❌ Invalid LLM selected.", ""

        # 6. QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        result = qa_chain.invoke(query)
        answer = result["result"]

        # 7. Update history
        if user_id not in chat_histories:
            chat_histories[user_id] = []
        chat_histories[user_id].append({"question": query, "answer": answer})
        chat_histories[user_id] = chat_histories[user_id][-5:]

        history_str = "\n\n".join([
            f"🗾 Q: {item['question']}\n🗱 A: {item['answer']}"
            for item in chat_histories[user_id]
        ])

        return answer, history_str

    except Exception as e:
        return f"❌ Error: {str(e)}", ""

# 🎛️ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("🔍 **RAG Q&A with Gemini/Ollama + Pinecone + Google Drive**")

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="💬 Your Question", 
                placeholder="Ask a question from your PDF files..."
            )
            answer_output = gr.Textbox(
                label="🧠 Answer",
                lines=4,
                interactive=False
            )
            chat_history_output = gr.Textbox(
                label="🕘 Chat History (Last 5)",
                lines=10,
                interactive=False
            )
            submit_btn = gr.Button("🚀 Submit")
            clear_btn = gr.Button("🗑️ Clear")

        default_user_id = str(uuid.uuid4())
        with gr.Column(scale=1):
            user_id_input = gr.Textbox(
                label="👤 User ID",
                value=default_user_id,
                placeholder="Enter your ID"
            )
            folder_id_input = gr.Textbox(
                label="📁 Google Drive Folder ID",
                placeholder="Paste your folder ID here"
            )
            llm_selector = gr.Radio(
                choices=["ollama", "gemini-1.5-flash"],
                label="🤖 Choose LLM Model",
                value="gemini-1.5-pro"
            )
            ollama_url_input = gr.Textbox(
                label="🌐 Ollama Server URL",
                placeholder="http://localhost:11434",
                visible=False
            )
            gemini_api_key_input = gr.Textbox(
                label="🔑 Gemini API Key",
                placeholder="Enter your Gemini API key here",
                visible=True
            )

    def toggle_llm_fields(llm_choice):
        return (
            gr.update(visible=llm_choice == "ollama"),
            gr.update(visible=llm_choice == "gemini-1.5-flash")
        )

    llm_selector.change(
        toggle_llm_fields,
        inputs=llm_selector,
        outputs=[ollama_url_input, gemini_api_key_input]
    )

    submit_btn.click(
        fn=rag_pipeline,
        inputs=[
            user_id_input,
            folder_id_input,
            llm_selector,
            ollama_url_input,
            gemini_api_key_input,
            query_input
        ],
        outputs=[answer_output, chat_history_output]
    )

    def clear_all():
        return "", "", "", "", "", ""

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[user_id_input, folder_id_input, query_input, ollama_url_input, answer_output, chat_history_output]
    )

demo.launch()