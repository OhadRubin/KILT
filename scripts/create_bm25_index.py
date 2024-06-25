import os
from tqdm import tqdm
import json
import csv
from typing import Set, Iterator, Tuple

import os
from tqdm import tqdm
import json
import csv
from typing import Set, Iterator, Tuple


def to_content(passage: Tuple[str, str, str]) -> str:
    did, text, title = passage
    item = {
        "id": did,
        "contents": f"{title}\n\n\n{text}"
    }
    return json.dumps(item)
import more_itertools
def get_line_count(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for _ in file)
    
def split_tsc_sharded(file_path, number_of_shards, output_directory):
    # Count total lines to determine lines per shard
    total_lines = get_line_count(file_path)
    
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    

    # Open the input file
    with open(file_path, 'r') as file:
        content_records = (f"{to_content(record)}\n" for record in tqdm(csv.reader(file, delimiter='\t'), desc='Processing Records'))
        # records = list(tqdm(, total=total_lines, desc='Reading Records'))
        
        for shard_number,shard_record in enumerate(more_itertools.distribute(number_of_shards, content_records)):
            shard_record = list(shard_record)
            # Prepare a shard file name
            output_file_path = os.path.join(output_directory, f'shard_{shard_number}.jsonl')
            # Open the shard file
            with open(output_file_path, 'w') as output_file:
                for line in tqdm(shard_record, desc=f'Writing Shard {shard_number}'):
                    output_file.write(line)

# Usage
# split_passages_sharded('kilt_w100_title.tsv', 16, 'kilt_w100_sharded')
# Usage
split_tsc_sharded('kilt_w100_title.tsv', 16, 'kilt_w100_sharded')