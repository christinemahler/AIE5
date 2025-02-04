import os
import getpass
os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key:")

import nest_asyncio
nest_asyncio.apply()

from langchain.document_loaders import SitemapLoader
documents = SitemapLoader(web_path="https://blog.langchain.dev/sitemap-posts.xml").load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 0,
    length_function = len,
)
split_chunks = text_splitter.split_documents(documents)

len(split_chunks)

max_chunk_length = 0
for chunk in split_chunks:
  max_chunk_length = max(max_chunk_length, len(chunk.page_content))
print(max_chunk_length)

from langchain_community.vectorstores import Qdrant
from langchain_openai.embeddings import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_vectorstore = Qdrant.from_documents(
    documents=split_chunks,
    embedding=embedding_model,
    location=":memory:"
)

qdrant_retriever = qdrant_vectorstore.as_retriever()

from langchain.prompts import ChatPromptTemplate
base_rag_prompt_template = """\
<<COMPLETE YOUR RAG PROMPT HERE>>

Context:
{context}

Question:
{question}
"""
base_rag_prompt = ChatPromptTemplate.from_template(base_rag_prompt_template)

from langchain_openai.chat_models import ChatOpenAI
base_llm = ChatOpenAI(model="gpt-4o-mini", tags=["base_llm"])

from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

retrieval_augmented_qa_chain = (
    {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | base_llm, "context": itemgetter("context")}
)

print(retrieval_augmented_qa_chain.get_graph().draw_ascii())

response = retrieval_augmented_qa_chain.invoke({"question" : "What's new in LangChain v0.2?"})
response["response"].content

for context in response["context"]:
  print("Context:")
  print(context)
  print("----")

response = retrieval_augmented_qa_chain.invoke({"question" : "What is the airspeed velocity of an unladen swallow?"})
response["response"].content

from uuid import uuid4
unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"LangSmith - {unique_id}"

os.environ["LANGCHAIN_API_KEY"] = getpass.getpass('Enter your LangSmith API key: ')

retrieval_augmented_qa_chain.invoke({"question" : "What is LangSmith?"}, {"tags" : ["Demo Run"]})['response']

import pandas as pd
test_df = pd.read_csv("DataRepository/langchain_blog_test_data.csv")

from langsmith import Client
client = Client()
dataset_name = "langsmith-demo-dataset-aie4-triples-v3"
dataset = client.create_dataset(
    dataset_name=dataset_name, description="LangChain Blog Test Questions"
)

for triplet in test_df.iterrows():
  triplet = triplet[1]
  client.create_example(
      inputs={"question" : triplet["question"], "context": triplet["context"]},
      outputs={"answer" : triplet["answer"]},
      dataset_id=dataset.id
  )

def prepare_data_ref(run, example):
  return {
      "prediction" : run.outputs["response"],
      "reference" : example.outputs["answer"],
      "input" : example.inputs["question"]
  }

def prepare_data_noref(run, example):
  return {
      "prediction" : run.outputs["response"],
      "input" : example.inputs["question"]
  }

def prepare_context_ref(run, example):
  return {
      "prediction" : run.outputs["response"],
      "reference" : example.inputs["context"],
      "input" : example.inputs["question"]
  }

from langsmith.evaluation import LangChainStringEvaluator, evaluate
eval_llm = ChatOpenAI(model="gpt-4o-mini", tags=["eval_llm"])
cot_qa_evaluator = LangChainStringEvaluator("cot_qa",  config={"llm":eval_llm}, prepare_data=prepare_context_ref)

unlabeled_dopeness_evaluator = LangChainStringEvaluator(
    "criteria",
    config={
        "criteria" : {
            "dopeness" : "Is the answer to the question dope, meaning cool - awesome - and legit?"
        },
        "llm" : eval_llm,
    },
    prepare_data=prepare_data_noref
)

labeled_score_evaluator = LangChainStringEvaluator(
    "labeled_score_string",
    config={
        "criteria": {
            "accuracy": "Is the generated answer the same as the reference answer?"
        },
    },
    prepare_data=prepare_data_ref
)

base_rag_results = evaluate(
    retrieval_augmented_qa_chain.invoke,
    data=dataset_name,
    evaluators=[
        cot_qa_evaluator,
        unlabeled_dopeness_evaluator,
        labeled_score_evaluator,
        ],
    experiment_prefix="Base RAG Evaluation"
)