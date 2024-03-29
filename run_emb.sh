# 14 train and 7 valid
python -m preprocess.get_meta_emb --dpath=/data/yunfanhu/samples_emb \
                                --filep=train.tsv \
                                --vfilep=valid.tsv \
                                --chunk_size=50000
python -m preprocess.process_meta --dpath=/data/yunfanhu/samples_emb \
                                    --drop_num=10
# origin
python train_emb.py --dpath=/data/yunfanhu/samples_emb \
                    --batch_size=2 \
                    --chunk_size=2048 \
                    --filep=train.tsv \
                    --vfilep=valid_5M.tsv \
                    --max_steps=300000 \
                    --save_path=cps_emb \
                    --plots=plots/m1_emb.jpg \
                    --save_steps=30000
