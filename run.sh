source ~/.bashrc

mkdir data cps

# one day full data
python -m preprocess.get_meta --dpath=/data/yunfanhu/mannual --filep=one_day_0.tsv --vfilep=sample_1M.tsv --chunk_size=50000
# 4 GPUs
python train.py --dpath=/data/yunfanhu/mannual --batch_size=2 --filep=one_day_0.tsv --vfilep=valid_1M.tsv --max_steps=1000000 --save_steps=100000

# 14 train and 7 valid
python -m preprocess.get_meta --dpath=/data/yunfanhu/samples --filep=train.tsv --vfilep=valid.tsv --chunk_size=50000
python -m preprocess.build_no_id_dataset --dpath=/data/yunfanhu/samples --filep=train.tsv --vfilep=valid.tsv --chunk_size=50000
# origin
python train.py --dpath=/data/yunfanhu/samples \
                    --batch_size=8 \
                    --filep=train.tsv \
                    --vfilep=valid_3M.tsv \
                    --max_steps=800000 \
                    --save_path=cps \
                    --save_steps=80000

python train.py --dpath=/data/yunfanhu/samples \
                    --batch_size=8 \
                    --filep=train.tsv \
                    --vfilep=valid_3M.tsv \
                    --max_steps=800000 \
                    --save_steps=80000 \
                    --save_path=cps2
# no id
python train.py --dpath=/data/yunfanhu/samples \
                    --batch_size=8 \
                    --filep=train.tsv \
                    --vfilep=valid_3M.tsv \
                    --max_steps=800000 \
                    --save_steps=80000 \
                    --with_id=0 \
                    --save_path=cps_noid
# small train
python train_rand.py --dpath=/data/yunfanhu/samples \
                    --filep=train_5M.tsv \
                    --vfilep=valid_1M.tsv \
                    --epoch=10 \
                    --save_path=cps_small



