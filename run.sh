source ~/.bashrc

mkdir data cps

python -m preprocess.get_meta --dpath=/data/yunfanhu/mannual --filep=one_day_0.tsv --vfilep=sample_1M.tsv --chunk_size=50000
# 4 GPUs
python train.py --dpath=/data/yunfanhu/mannual --batch_size=2 --filep=one_day_0.tsv --vfilep=valid_1M.tsv --max_steps=1000000 --save_steps=100000
# 8 GPUs
python train.py --with_id=0 --dpath=/data/yunfanhu/mannual --batch_size=2 --chunk_size=16 --filep=one_day_0.tsv --vfilep=valid_1M.tsv --max_steps=1000000 --save_steps=100000


python train_catboost.py  --dpath=/data/yunfanhu/mannual --filep=sample_1M.tsv

python -m preprocess.get_meta --dpath=/data/yunfanhu/mannual --filep=train_small.tsv --vfilep=valid_small.tsv --chunk_size=50000

python -m preprocess.get_meta --dpath=/data/yunfanhu/samples --filep=train.tsv --vfilep=valid.tsv --chunk_size=50000
python train.py --dpath=/data/yunfanhu/samples --batch_size=8 --filep=train.tsv --vfilep=valid_3M.tsv --max_steps=800000 --save_steps=80000