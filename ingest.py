from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='test.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50,
                                                    separators=["\n\n","\n\uf06e", " ", "","。"])
    texts = text_splitter.split_documents(documents)

    # You can find chinese embedding model here:
    # https://huggingface.co/spaces/mteb/leaderboard

    # embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
    #                                    model_kwargs={'device': 'cuda'})
    # embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese",
    #                                    model_kwargs={'device': 'cuda'})
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large-zh",
                                       model_kwargs={'device': 'cuda'})
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH, index_name = "test")

if __name__ == "__main__":
    create_vector_db()

