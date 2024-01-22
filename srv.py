from fastapi import FastAPI

from src.searcher.searcher import Searcher

app = FastAPI()

searcher = Searcher('data3.xlsx')

searcher.embend()
searcher.llama_init()

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/question/{query}")
async def read_item(query):
    resp = await searcher.request(query)
    return {"text": resp}
