import dotenv
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.document_loaders import (PagedPDFSplitter,
                                                  PyPDFium2Loader, PyPDFLoader)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

dotenv.load_dotenv()
settings = dotenv.dotenv_values(dotenv_path=dotenv.find_dotenv())

if (PHB_PATH := settings.get('PHB_PATH')) is None:
    raise ValueError('Set the PHB_PATH var in your .env file')

loader = PyPDFLoader(PHB_PATH)

pages = loader.load()

vectorstore = Chroma.from_documents(
    documents=pages, embedding=OpenAIEmbeddings(), persist_directory=settings.get('DB_PATH', './phb_db')
)
