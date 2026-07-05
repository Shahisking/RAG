from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("RAG.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)
len(chunks)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever()

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens=256,
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=pipe)
ques=input("Ask your question about RAG: ")
def rag_chain(question):
   
    docs = retriever.invoke(question)


    context = "\n\n".join([d.page_content for d in docs])

    prompt_text = f"""
You are an AI assistant. Use ONLY the provided context to answer.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:
"""

    # 4. Call the model
    raw = llm.invoke(prompt_text)
    answer = raw.replace(prompt_text, "").strip()
    return answer

response = rag_chain(ques)
print(response)
