import sys
import os
import json
import glob
from typing import List, Dict

import datasets
import tqdm
import pandas as pd
from pyserini.search import LuceneSearcher
from kilt.kilt_utils import load_data

WIKI_INDEX_PATH = "/dev/shm/lucane_indexes/kilt_w100_sharded"
KILT_DATA_PREFIX = "/dev/shm/kilt_data/data"

def decode_doc(doc):
    return json.loads(doc.raw())["contents"]

def search_question(batch: Dict, searcher: LuceneSearcher) -> Dict:
    qids = batch["qid"]
    hits = searcher.batch_search(batch["question"], qids, threads=300, k=100)
    ctxs_per_doc = [[hit.docid for hit in hits[qid]] for qid in qids]
    ctxs = sum(ctxs_per_doc, [])
    doc_res = searcher.batch_doc(ctxs, threads=300)
    docs_raw = [[decode_doc(doc_res[x]) for x in doc_hits] for doc_hits in ctxs_per_doc]
    batch["ctxs"] = docs_raw
    return batch

def format_example(x: Dict) -> Dict:
    result = {"question": x["input"], "qid": x["id"]}
    if "output" in x:
        result["output"] = x["output"]
    return result

def get_paths(prefix: str) -> List[Dict]:
    dataset_paths = []
    for p in glob.glob(f"{prefix}/*"):
        name, split, _ = p.split("/")[-1].split(".jsonl")[0].split("-")
        dataset_paths.append((name, split, p))
    df = pd.DataFrame(dataset_paths, columns=["name", "split", "path"])
    df = df.pivot(index="name", columns="split", values="path")
    df.fillna("", inplace=True)
    df.reset_index(inplace=True)
    return df.to_dict("records")


def process_dataset(file_path: str, searcher: LuceneSearcher, name:str,split:str) -> datasets.Dataset:
    data = load_data(file_path)
    input_list = [format_example(x) for x in data]
    dataset = datasets.Dataset.from_list(input_list)
    
    mapped_dataset = dataset.map(
        search_question,
        batch_size=300,
        batched=True,
        fn_kwargs={"searcher": searcher},
        cache_file_name=f"/dev/shm/tmp/{name}_{split}.arrow"
    )
    
    return mapped_dataset

def generate_dataset(record: Dict, searcher: LuceneSearcher, name:str) -> datasets.DatasetDict:
    dataset_dict = {}
    for split in record.keys():
        if record.get(split):
            dataset = process_dataset(record[split], searcher, name, split)
            dataset_dict[split] = datasets.Dataset.from_list(dataset)
    
    return datasets.DatasetDict(dataset_dict)

def main():
    searcher = LuceneSearcher(WIKI_INDEX_PATH)
    records = get_paths(KILT_DATA_PREFIX)
    
    for record in records:
        dataset_name = record['name']
        print(f"Processing dataset: {dataset_name}")
        dataset_dict = generate_dataset(record, searcher, dataset_name)
        
        # Save to disk or push to HuggingFace Hub
        cache_dir = f"/dev/shm/datasets/{dataset_name}_bm25_top100"
        # dataset_dict.save_to_disk(cache_dir)
        
        # Uncomment the following line to push to HuggingFace Hub
        dataset_dict.push_to_hub(f"{dataset_name}_bm25_top100_kilt", token=os.environ["HF_TOKEN"])

if __name__ == "__main__":
    main()