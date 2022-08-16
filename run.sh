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
                    --batch_size=2 \
                    --chunk_size=2048 \
                    --filep=train.tsv \
                    --vfilep=valid_5M.tsv \
                    --max_steps=75000 \
                    --save_path=cps \
                    --plots=plots/m1.jpg \
                    --save_steps=15000
# 20
python train.py --dpath=/data/yunfanhu/samples_20 \
                    --batch_size=2 \
                    --chunk_size=2048 \
                    --filep=train.tsv \
                    --vfilep=valid_5M.tsv \
                    --max_steps=300000 \
                    --save_path=cps_20 \
                    --plots=plots/m1_20.jpg \
                    --save_steps=30000
# downsample
python train.py --dpath=/data/yunfanhu/downsamples_20 \
                    --batch_size=2 \
                    --chunk_size=2048 \
                    --filep=train.tsv \
                    --vfilep=valid_5M.tsv \
                    --max_steps=300000 \
                    --save_path=cps_d30 \
                    --plots=plots/m1_d.jpg \
                    --save_steps=30000

# no id
python train.py --dpath=/data/yunfanhu/samples_20 \
                    --batch_size=2 \
                    --chunk_size=2048 \
                    --filep=train.tsv \
                    --vfilep=valid_5M.tsv \
                    --max_steps=300000 \
                    --save_path=cps_noid_w1 \
                    --with_id=0 \
                    --plots=plots/baseline_w1.jpg \
                    --save_steps=30000
# small train
python train_rand.py --dpath=/data/yunfanhu/samples \
                    --batch_size=512 \
                    --filep=train_5M.tsv \
                    --vfilep=valid_1M.tsv \
                    --epoch=5 \
                    --save_path=cps_small
# build plot
python -m utils.plot_two_cali --spath=plots/two.jpg \
                                --m0=cps_noid_w1/res.csv \
                                --m1=cps2/res.csv \
                                --points=500 \
                                --quantile

python -m utils.plot_one_cali --spath=plots/one.jpg \
                                --res=cps2/res.csv \
                                --points=500 \
                                --quantile

# train lightgbm for importance
python train_catboost.py  --dpath=/data/yunfanhu/samples \
                    --filep=train_5M.tsv \
                    --vfilep=valid_1M.tsv
# process bigger files            
python -m preprocess.get_meta --dpath=/data/yunfanhu/samples_20 \
                            --filep=train.tsv \
                            --vfilep=valid.tsv \
                            --chunk_size=50000

python -m preprocess.process_meta --dpath=/data/yunfanhu/samples_20 \
                                    --drop_num=10

