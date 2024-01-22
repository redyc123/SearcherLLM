from fastapi import FastAPI

from src.searcher.searcher import Searcher
from query import Query

app = FastAPI()

searcher = Searcher('data3.xlsx')

searcher.embend()
searcher.llama_init()

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/question")
async def read_item(query: Query):
    resp = await searcher.request(query.text)
    return {"text": resp}
