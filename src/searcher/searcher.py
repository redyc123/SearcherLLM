from typing import List
from langchain_core.documents import Document

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.llamacpp import LlamaCpp

from langchain.document_loaders.excel import UnstructuredExcelLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores.chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

from googletrans import Translator

from template_store import SEARCH_TEMPLATE


class Searcher:
    trans = Translator()
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    def __init__(self, data: str) -> None:
        self.loader = UnstructuredExcelLoader(
            file_path=f'./data/{data}',
            mode='elements'
        )
        self.head_titles: List[str] = []
        self.db: Chroma
        self.llm: LlamaCpp 

    def make_head_titles(self, texts: List[Document]) -> List[str]:
        head = texts.pop(0).page_content
        self.head_titles = head.split('\n')
        return self.head_titles

    def embend(self) -> Chroma:
        docs = self.loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)

        self.make_head_titles(texts)

        for i in range(len(texts)):
            line = texts[i].page_content.split('\n')
            text = []
            for j in range(len(line)):
                text.append(self.head_titles[j] + ': ' + line[j] + ' ||')
            texts[i].page_content = ' '.join(text)
        embedings = SentenceTransformerEmbeddings(
            model_name="all-mpnet-base-v2",
            model_kwargs={'device': 'cuda'}
        )
        return Chroma.from_documents(filter_complex_metadata(texts), embedings)

    def llama_init(self) -> LlamaCpp:
        llm = LlamaCpp(
            use_mlock=True,
            n_batch = 2048,
            # n_gpu_layers=100,
            model_path="./data/llama-2-7b-chat.Q2_K.gguf",
            temperature=0.75,
            repeat_penalty=1,
            max_tokens=2000,
            n_ctx=5000,
            top_p=1,
            # callback_manager=callback_manager,
            verbose=False,  # Verbose is required to pass to the callback manager
            stop=['\n'],
        )
        return llm

    async def request(self, query: str) -> str:
        db = self.db
        llm = self.llm
        documents = db.similarity_search(query, k=1)
        prompt = PromptTemplate.from_template(SEARCH_TEMPLATE)
        context = documents[0].page_content
        query_llm = LLMChain(llm=llm, prompt=prompt)
        resp = query_llm.run({'context': context, 'question': query})
        ru_resp = self.trans.translate(resp, source='en', dest='ru')
        return ru_resp.text
