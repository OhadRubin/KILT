python scripts/execute_retrieval.py -m bm25 -o predictions/bm25 --test_config kilt/configs/all_data.json



mkdir -p /dev/shm/kilt_data/data
cd /dev/shm/kilt_data
export PYTHONPATH=/home/ohadr/KILT:$PYTHONPATH
python3.10 /home/ohadr/KILT/scripts/download_all_kilt_data.py
python3.10 /home/ohadr/KILT/scripts/get_triviaqa_input.py


wget http://dl.fbaipublicfiles.com/KILT/kilt_w100_title.tsv