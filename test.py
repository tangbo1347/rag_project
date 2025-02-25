from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


from lib.eli import CustomEmbeddings, CustomLLM
from lib.embedding import TransformersEmbeddings
from lib.deepseek import DeepSeekLLM, DeepSeekLLM_Agent

from lib.config import config_get_item

embedding_model = TransformersEmbeddings()

custom_llm = CustomLLM(api_generate_url=config_get_item("eli", "eli_generate"), api_key=config_get_item("eli", "eli_api_key"))
custom_llm_deep = DeepSeekLLM(api_generate_url=config_get_item("deepseek", "base_url"), deepseek_api_key=config_get_item("deepseek", "deepseek_api_key"))


documents1 = [
    "2024年1月1日，芯片板块上涨，白酒板块回调",
    "2024年1月2日铜线缆板块上涨",
    "2024年1月3日数据板块上涨",
    "2024年1月4日光通信板块上涨",
    "2024年1月5日芯片板块回调，白酒和猪肉板块上涨",
    "2024年1月6日白酒板块上涨后，指数容易出现冲高回落"
]

# import os
# data_path = os.path.join(os.getcwd(), "data", "output_002304.csv")
# with open(data_path, "r") as file:
#     content = file.read()



db = FAISS.from_texts(documents1, embedding_model)
db.save_local("faiss_index")


db = FAISS.load_local(
        "faiss_index", embedding_model, allow_dangerous_deserialization=True
    )

qa_chain = RetrievalQA.from_chain_type(llm=custom_llm_deep, retriever=db.as_retriever())

query = "如果日期代表板块轮动的先后关系，能否分析出板块之间的轮动关系"
result = qa_chain.invoke(query)
print("回答：", result)
