python3.10 scripts/execute_retrieval.py -m bm25 -o predictions/bm25 --test_config kilt/configs/all_data.json



mkdir -p /dev/shm/kilt_data/data
cd /dev/shm/kilt_data
export PYTHONPATH=/home/ohadr/KILT:$PYTHONPATH
python3.10 /home/ohadr/KILT/scripts/download_all_kilt_data.py
python3.10 /home/ohadr/KILT/scripts/get_triviaqa_input.py


wget http://dl.fbaipublicfiles.com/KILT/kilt_w100_title.tsv
python3.10 /home/ohadr/KILT/scripts/create_bm25_index.py

python3.10 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /dev/shm/kilt_data/kilt_w100_sharded \
  --index /dev/shm/lucane_indexes/kilt_w100_sharded \
  --generator DefaultLuceneDocumentGenerator \
  --threads 200 \
  --storePositions --storeDocvectors --storeRaw

ln -s /dev/shm/kilt_data/data /home/ohadr/KILT/data


python3.10 /home/ohadr/KILT/scripts/execute_retrieval.py -m bm25 -o /dev/shm/kilt_data/bm25_predictions --test_config kilt/configs/all_data.json
python3.10 /home/ohadr/KILT/scripts/upload_to_hf.py

