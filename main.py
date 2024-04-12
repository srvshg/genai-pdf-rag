import os
import fitz  # PyMuPDF
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
# Open the PDF file
with fitz.open('input-files/Stand_out_of_our_Light.pdf') as doc:
    text_in_pdf = ""
    for page in doc:
        text_in_pdf += page.get_text()

    with open("text_in_pdf.txt", "w") as file:
        file.write(text_in_pdf)


loader = TextLoader("text_in_pdf.txt")
text_documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(text_documents)


model = ChatOpenAI(openai_api_key=os.environ.get(
    "OPENAI_API_KEY"), model="gpt-3.5-turbo")


parser = StrOutputParser()


template = """
Answer the question based on the context below. If you can't
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


embeddings = OpenAIEmbeddings()


vectorstore = DocArrayInMemorySearch.from_documents(documents, embeddings)


chain = (
    {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)

print(chain.invoke("what is the book about?"))
