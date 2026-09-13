from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from rich import print
import re
import os
path=os.path.join(os.path.dirname(os.path.dirname(__file__)),"ASSETS","firstaid1.pdf")
loader = PyMuPDFLoader(path)
docs = loader.load()
docs=docs[4:]
for doc in docs:
    doc.metadata['page']-=4
print("TOTAL DOCS:", len(docs))
for i, doc in enumerate(docs[40:80]):
    text = doc.page_content
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = text.strip()
    doc.page_content = text
    print("=" * 20, i, "=" * 20)
    print(doc)
    print("-" * 41)


