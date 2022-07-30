source ~/.bashrc

mkdir data cps
# 14 train and 7 valid
python -m preprocess.get_meta --dpath=/data/yunfanhu/samples --filep=train.tsv --vfilep=valid.tsv --chunk_size=50000
# origin
python train.py --dpath=/data/yunfanhu/samples_emb \
                    --batch_size=8 \
                    --filep=train.tsv \
                    --vfilep=valid_3M.tsv \
                    --max_steps=800000 \
                    --save_steps=80000 \
                    --save_path=cps_emb2

python train.py --dpath=/data/yunfanhu/samples_emb \
                    --batch_size=4 \
                    --filep=train.tsv \
                    --vfilep=valid_3M.tsv \
                    --max_steps=1600000 \
                    --save_steps=160000 \
                    --save_path=cps_emb

