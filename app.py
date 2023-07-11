from flask import Flask, request, render_template
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import requests
from PyPDF2 import PdfFileReader
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import openai
#import pickle
#from openai.embeddings_utils import get_embedding
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
#import dotenv
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')




pinecone.init(
    api_key=PINECONE_API_KEY,  
    environment=PINECONE_API_ENV  
)
index_name = "semantic-search-book"


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
docsearch = Pinecone.from_existing_index("semantic-search-book", embeddings)

@app.route('/', methods=['GET', 'POST'])
def search():
    response = None
    if request.method == 'POST':
        query = request.form['query']
        docs = docsearch.similarity_search(query)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)

    return render_template('search.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)
