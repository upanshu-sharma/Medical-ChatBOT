from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore as PC
from pinecone import Pinecone, ServerlessSpec
import pinecone
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')

extracted_data=load_pdf("C:\VS Workspace\Medical ChatBOT\data")
text_chunks=text_split(extracted_data)
embeddings=download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"  
index = pc.Index(index_name)
docsearch=PC.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)


