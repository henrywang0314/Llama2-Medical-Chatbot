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
你是一位于台湾经验丰富的宠物狗医生，你将会判断并於Context中挑選与使用者问题(Question)高度相关的资讯片段，最后使用高度相关的资讯回答使用者的问题。
高度相关的资讯定義為: 可在该资讯下直接获得使用者问题的答案。
如果你不知道答案，就说你不知道，不要试图编造答案。

Context: {context}
(Only use extremely high relevent pieces of information)
<|user|>
Question: {question}

Always answer with Traditional Chinese (zh-TW) instead of English or Simplified Chinese.
<|assistant|>
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    from langchain.output_parsers.regex import RegexParser
    from langchain_core.prompts import PromptTemplate

    output_parser = RegexParser(
        regex=r"(.*?)\nScore: (\d*)",
        output_keys=["answer", "score"],
    )

    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:

    Question: [question here]
    Helpful Answer: [answer here]
    Score: [score between 0 and 100]

    How to determine the score:
    - Higher is a better answer
    - Better responds fully to the asked question, with sufficient level of detail
    - If you do not know the answer based on the context, that should be a score of 0
    - Don't be overconfident!

    Example #1

    Context:
    ---------
    Apples are red
    ---------
    Question: what color are apples?
    Helpful Answer: red
    Score: 100

    Example #2

    Context:
    ---------
    it was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv
    ---------
    Question: what type was the car?
    Helpful Answer: a sports car or an suv
    Score: 60

    Example #3

    Context:
    ---------
    Pears are either red or orange
    ---------
    Question: what color are apples?
    Helpful Answer: This document does not answer the question
    Score: 0

    Begin!

    Context:
    ---------
    {context}
    ---------
    Question: {question}
    Helpful Answer:"""
    # prompt = PromptTemplate(
    #     template=prompt_template,
    #     input_variables=["context", "question"],
    #     output_parser=output_parser,
    # )
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    # stuff
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt},
                                       verbose = True,
                                       )

    # map_rerank
    # map_template = "Write a summary of the following text:\n\n{text}"
    # map_prompt_template = PromptTemplate(template=map_template, input_variables=["text"])
    # score_template = "Rate the relevance of this summary to the question {question} on a scale of 1 to 5:\n\n{text}"
    # score_prompt_template = PromptTemplate(template=score_template, input_variables=["question", "text"])
    # qa_chain = RetrievalQA.from_chain_type(llm=llm, 
    #                                        chain_type="map_rerank", 
    #                                        retriever=db.as_retriever(search_kwargs={'k': 3}), 
    #                                        chain_type_kwargs={'prompt': PROMPT}
    #                                        )

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
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
#Loading the model
# def load_llm():
#     # Load the locally downloaded model here
#     model_config = AutoConfig.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
#     tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
#     model = AutoModel.from_pretrained("THUDM/chatglm3-6b", config=model_config, trust_remote_code=True).quantize(4).cuda()
#     model = model.eval()
    
#     pipe = transformers.pipeline(
#         model=model,
#         task='text-generation',
#         tokenizer=tokenizer,
#         temperature=0.2,
#         trust_remote_code=True,
#         max_new_tokens=2048,
#         device=0
#     )
#     # llm = ChatGLM3()
#     llm = HuggingFacePipeline(pipeline=pipe)
#     return llm
#QA Model Function
def qa_bot():
    # embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese",
    #                                    model_kwargs={'device': 'cuda'})
    # db = FAISS.load_local(DB_FAISS_PATH, embeddings, index_name = "thenlper_gte-large-zh")
    # embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese",
    #                                    model_kwargs={'device': 'cuda'})
    # db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large-zh",
                                       model_kwargs={'device': 'cuda'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, index_name = "thenlper_gte-large-zh")
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

