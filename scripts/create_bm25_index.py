import os
from tqdm import tqdm
import json
import csv
from typing import Set, Iterator, Tuple

def load_passages(
    path: str, 
    restricted_ids: Set[str] = None, 
    use_csv_reader: bool = None, 
    topk: int = None
) -> Iterator[Tuple[str, str, str]]:
    if use_csv_reader is None:
        use_csv_reader = 'psgs_w100.tsv' in path
    if not os.path.exists(path):
        print(f'{path} does not exist')
        return iter([])
    
    print(f'Loading passages from: {path}')
    with open(path) as fin:
        if use_csv_reader:
            reader = csv.reader(fin, delimiter='\t')
            header = next(reader)
        else:
            header = fin.readline().strip().split('\t')
        assert len(header) == 3 and header[0] == 'id', 'header format error'
        textfirst = header[1] == 'text'
        
        for k, row in enumerate(reader if use_csv_reader else fin):
            if (k + 1) % 1000000 == 0:
                print(f'{(k + 1) // 1000000}M', end=' ', flush=True)
            try:
                if not use_csv_reader:
                    row = row.rstrip('\n').split('\t')
                if restricted_ids and row[0] not in restricted_ids:
                    continue
                if textfirst:
                    did, text, title = row[0], row[1], row[2]
                else:
                    did, text, title = row[0], row[2], row[1]
                yield did, text, title
                if topk is not None and k + 1 >= topk:
                    break
            except:
                print(f'The following input line has not been correctly loaded: {row}')
    print()

def to_content(passage: Tuple[str, str, str]) -> str:
    did, text, title = passage
    item = {
        "id": did,
        "contents": f"{title} {text}"
    }
    return json.dumps(item)

def split_passages_sharded(file_path: str, number_of_shards: int, output_directory: str):
    # Get file size using os.stat
    file_size = os.stat(file_path).st_size
    bytes_per_shard = file_size // number_of_shards + (file_size % number_of_shards > 0)
    
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Open the input file and start processing
    passages_iterator = load_passages(file_path)
    
    for shard_number in range(number_of_shards):
        # Prepare a shard file name
        output_file_path = os.path.join(output_directory, f'shard_{shard_number}.jsonl')
        # Open the shard file
        with open(output_file_path, 'w') as output_file:
            bytes_written = 0
            for passage in tqdm(passages_iterator, desc=f'Writing Shard {shard_number}'):
                content = to_content(passage)
                output_file.write(f"{content}\n")
                bytes_written += len(content) + 1  # +1 for newline
                if bytes_written >= bytes_per_shard:
                    break
        
        if bytes_written < bytes_per_shard:  # If we didn't write all expected bytes, we're done
            break

# Usage
split_passages_sharded('kilt_w100_title.tsv', 16, 'kilt_w100_sharded')