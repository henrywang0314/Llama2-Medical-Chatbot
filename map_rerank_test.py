from langchain.chains import StuffDocumentsChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers.regex import RegexParser
from langchain.llms import ChatGLM
from langchain.chains.combine_documents.map_rerank import MapRerankDocumentsChain

document_variable_name = "context"
llm = ChatGLM(endpoint_url="http://127.0.0.1:8080", max_token=2048, top_p=0.9, temperature=0.1, with_history=False, streaming = True)
# The prompt here should take as an input variable the
# `document_variable_name`
# The actual prompt will need to be a lot more complex, this is just
# an example.
prompt_template = (
    "Use the following context to tell me the chemical formula "
    "for water. Output both your answer and a score of how confident "
    "you are. Context: {content}"
)
output_parser = RegexParser(
    regex=r"(.*?)\nScore: (.*)",
    output_keys=["answer", "score"],
)
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context"],
    output_parser=output_parser,
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
chain = MapRerankDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name=document_variable_name,
    rank_key="score",
    answer_key="answer",
)