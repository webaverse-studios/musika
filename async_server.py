import asyncio
import json
import os
import sys
from dataclasses import dataclass
from fastapi import FastAPI, Request, BackgroundTasks
import logging
import os
from time import time
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

import numpy as np
from generator import Generator
from scipy.io.wavfile import write
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from parse.parse_test import parse_args
from models import Models_functions


logging.basicConfig(level=logging.INFO, format="%(levelname)-9s %(asctime)s - %(name)s - %(message)s")
LOGGER = logging.getLogger(__name__)

EXPERIMENTS_BASE_DIR = "/experiments/"
QUERY_BUFFER = {}

args = None
models_ls = None
generator = None
path = "./result"

def load_model():  
    global args, models_ls, generator
    # parse args
    args = parse_args()

    # initialize networks
    model_functions = Models_functions(args)
    models_ls = model_functions.get_networks()
    generator = Generator(args)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)
loop = asyncio.get_event_loop()

@dataclass
class Query():
    query_name: str
    query_sequence: str
    duration: int
    style_variation: int
    result: str = ""
    extention: str = ".glb"
    experiment_id: str = None
    status: str = "pending"

    def __post_init__(self):
        self.experiment_id = str(time())
        self.experiment_dir = os.path.join(EXPERIMENTS_BASE_DIR, self.experiment_id)

@app.get("/generate")
async def root(request: Request, background_tasks: BackgroundTasks, duration: int, style_variation: int):
    print(duration, style_variation)
    query = Query(query_name="test", query_sequence=5, duration=duration, style_variation=style_variation)
    QUERY_BUFFER[query.experiment_id] = query
    background_tasks.add_task(process, query)
    LOGGER.info(f'root - added task')
    return {"id": query.experiment_id}

@app.get("/generate_result")
async def result(request: Request, query_id: str):
    if query_id in QUERY_BUFFER:
        if QUERY_BUFFER[query_id].status == "done":
            resp = FileResponse(QUERY_BUFFER[query_id].result, filename="output.wav")
            resp.headers["Content-Type"] = "audio/wav"
            del QUERY_BUFFER[query_id]
            return resp
        return {"status": QUERY_BUFFER[query_id].status}
    else:
        return {"status": "finished"}

def process(query):
    LOGGER.info(f"process - {query.experiment_id} - Submitted query job. Now run non-IO work for 10 seconds...")
    res = generator.generate(models_ls, query.duration, query.style_variation)
    _path = path + "/output" + str(time()) + ".wav"
    write(_path, args.sr, np.squeeze(res[1][1]))
    
    QUERY_BUFFER[query.experiment_id].status = "done"
    QUERY_BUFFER[query.experiment_id].result = _path
    LOGGER.info(f'process - {query.experiment_id} - done!')

@app.get("/backlog/")
def return_backlog():
    return {f"return_backlog - Currently {len(QUERY_BUFFER)} jobs in the backlog."}


if __name__ == "__main__":
    if not os.path.exists(path):
        os.mkdir(path)
    load_model()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7777)