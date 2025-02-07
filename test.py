from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


from lib.eli import CustomEmbeddings, CustomLLM

from lib.config import config_get_item

embedding_model = CustomEmbeddings()

custom_llm = CustomLLM(api_generate_url=config_get_item("eli", "eli_generate"), api_key=config_get_item("eli", "eli_api_key"))


documents1 = [
    "丹尼尔跟唐博是同事",
    "丹尼尔跟赵铜锌是同事",
    "丹尼尔每天居家办公",
    "赵铜锌擅长SAPC的产品",
    "丹尼尔不会SAPC",
    "唐博不会SAPC"
]

db = FAISS.from_texts(documents1, embedding_model)
db.save_local("faiss_index")


db = FAISS.load_local(
        "faiss_index", embedding_model, allow_dangerous_deserialization=True
    )

qa_chain = RetrievalQA.from_chain_type(llm=custom_llm, retriever=db.as_retriever())

query = "唐博跟赵铜锌是什么关系？"
result = qa_chain.run(query)
print("回答：", result)

query = "赵铜锌可以去哪找丹尼尔？"
result = qa_chain.run(query)
print("回答：", result)

query = "唐博有SAPC的问题，该向谁求助？"
result = qa_chain.run(query)
print("回答：", result)