from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from opencc import OpenCC
s2twp_cc = OpenCC('s2twp')
DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """<|system|>
你是一位於台灣經驗豐富的寵物狗醫生，使用以下資訊來回答使用者的問題。
如果你不知道答案，就說你不知道，不要試圖編造答案。

Context: {context}
<|user|>
Question: {question}

Always answer with Traditional Chinese (zh-TW) instead of English or Simplified Chinese.
<|assistant|>
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 3}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model

def load_llm():
    # Load the locally downloaded model here
    # llm = CTransformers(
    #     model = "TheBloke/Llama-2-7B-Chat-GGML",
    #     model_type="llama",
    #     max_new_tokens = 512,
    #     temperature = 0.5,
    #     # gpu_layers = 200,
    # )
    from langchain.llms import ChatGLM

    llm = ChatGLM(endpoint_url="http://127.0.0.1:8080", max_token=2048, top_p=0.9, temperature=0.1, with_history=False, streaming = True)

    return llm

#QA Model Function
def qa_bot():
    # embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese",
    #                                    model_kwargs={'device': 'cuda'})
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese",
                                       model_kwargs={'device': 'cuda'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    response = s2twp_cc.convert(response)
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    # res = await chain.acall(message.content, callbacks=[cb])
    res = await cl.make_async(chain)(
        message.content, callbacks=[cb]
)
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"
    await cl.Message(content=s2twp_cc.convert(answer)).send()

