source ~/.bashrc

mkdir data cps
python -m preprocess.get_meta

python train.py
python train.py --with_id=0
python train_catboost.py

python predict.py --load_from=cps/checkpoint-250/pytorch_model.bin

python -m preprocess.get_meta --dpath=/data/yunfanhu/mannual --filep=sample_1M.tsv
python train.py --dpath=/data/yunfanhu/mannual --filep=sample_1M.tsv --epoch=3
python train.py --dpath=/data/yunfanhu/mannual --filep=sample_1M.tsv --with_id=0  --epoch=3

python train_catboost.py  --dpath=/data/yunfanhu/mannual --filep=sample_1M.tsv
