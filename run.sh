source ~/.bashrc

mkdir data cps

python -m preprocess.get_meta --dpath=/data/yunfanhu/mannual --filep=one_day_0.tsv --vfilep=sample_1M.tsv --chunk_size=50000
python train.py --dpath=/data/yunfanhu/mannual --filep=sample_1M.tsv --epoch=3
python train.py --dpath=/data/yunfanhu/mannual --filep=sample_1M.tsv --with_id=0  --epoch=3

python train_catboost.py  --dpath=/data/yunfanhu/mannual --filep=sample_1M.tsv

python -m preprocess.get_meta --dpath=/data/yunfanhu/mannual --filep=train_small.tsv --vfilep=valid_small.tsv --chunk_size=50000

