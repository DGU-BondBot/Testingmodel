from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI()
#data_loader = UnstructuredWebPageLoader("https://namu.wiki/w/%EC%A7%91%EB%8B%A8%EC%A3%BC%EC%9D%98")
data_loader = UnstructuredFileLoader ("../files/wiki.txt")
cache_dir = LocalFileStore("./.cache/")

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50
)

docs = data_loader.load_and_split(text_splitter=splitter)
embeddings = OpenAIEmbeddings()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

vectorstore = Chroma.from_documents(docs, cached_embeddings)
retriever = vectorstore.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="map_reduce",
    retriever=retriever,
)

chain.run("한국의 집단주의에 대해 설명해줘")
# AIMessage(content='한국의 집단주의는 개인보다는 집단의 이익과 조화를 중시하는 사고방식을 말합니다. 한국 사회에서는 집단의 일원으로서의 역할과 책임을 강조하며, 개인의 욕구나 성취보다는 집단의 안정과 조화를 추구하는 경향이 있습니다. 이로 인해 한국인들은 식당에서도 한 가지 메뉴로 통일하려고 하며, 음식을 나눠먹는 일이 많습니다. 또한 혼자 음식을 먹는 것이 타인에게 나누어 주지 않는 것으로 간주되어 좋지 않게 여겨집니다. 이러한 집단주의는 가족, 친구, 동료와의 관계에서도 나타날 수 있으며, 개인의 선택이나 다른 음식을 먹는 것에 대해 신기해하는 경향이 있습니다. 이러한 문화적 배경으로 인해 한국 사회에서는 집단의 일원으로서의 역할과 집단의 조화를 중시하는 특징을 볼 수 있습니다.')

# ChatOpenAI() 기본 모델 답변과 비교
# '한국의 집단주의는 한국 사회에서 중요한 가치 중 하나로 여겨지는 개념입니다. 이는 개인의 이익보다는 집단의 이익을 우선시하는 사고방식을 의미합니다.\n\n한국 사회에서는 집단이 개인보다 더 큰 가치를 갖는다고 여깁니다. 따라서 개인의 목표나 행동은 종종 집단의 이익과 조화를 이루도록 조절되며, 개인의 권리와 의무는 집단의 안정과 번영을 위해 제한될 수 있습니다. 이러한 이유로 한국 사회에서는 개인의 자유보다는 집단의 안정과 조화를 중시하는 경향이 있습니다.\n\n한국의 집단주의는 한국인들이 다른 사람들과의 관계를 중요하게 여기고, 공동체 의식을 강조하는데 영향을 받았습니다. 가족, 동료, 직장 동료 등과의 관계는 매우 중요하며, 이들과의 상호작용을 통해 개인의 정체성이 형성됩니다. 따라서 한국 사회에서는 개인의 행동이 집단의 평판에 큰 영향을 미치기도 합니다.\n\n한국의 집단주의는 일부로는 안정적이고 화합적인 사회를 형성하는 데 도움이 되는 장점이 있습니다. 그러나 때로는 개인의 창의성과 독립적인 사고를 억압하거나 집단의 압력에 의해 개인의 권리와 자유가 제한될 수도 있습니다. 또한, 집단의 이익을 우선시하는 태도가 개인의 성장과 사회적 변화를 제한할 수도 있습니다.\n\n요약하자면, 한국의 집단주의는 집단의 이익을 우선시하는 사고방식으로, 한국 사회에서 중요한 가치 중 하나입니다. 이는 개인의 이익보다는 집단의 안정과 조화를 중시하며, 개인의 자유와 권리는 집단의 이익을 위해 제한될 수 있습니다.'