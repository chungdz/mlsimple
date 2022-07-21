source ~/.bashrc

mkdir data cps
python -m preprocess.get_meta

python train.py
python train.py --with_id=0

python -m preprocess.get_meta --dpath=/data/yunfanhu/mannual --filep=sample_1M.tsv
python train.py --dpath=/data/yunfanhu/mannual --filep=sample_1M.tsv

