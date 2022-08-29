source ~/.bashrc

mkdir data cps
# small dataset
python -m preprocess.get_meta --dpath=data --filep=sample.tsv --vfilep=valid.tsv --chunk_size=50000
python train_rand.py --dpath=data \
                    --batch_size=512 \
                    --filep=sample.tsv \
                    --vfilep=valid.tsv \
                    --epoch=2 \
                    --plots=plots/samples.jpg \
                    --save_path=cps_samples \
                    --points=50
# one day full data
python -m preprocess.get_meta --dpath=/data/yunfanhu/mannual --filep=one_day_0.tsv --vfilep=sample_1M.tsv --chunk_size=50000
# 4 GPUs
python train.py --dpath=/data/yunfanhu/mannual --batch_size=2 --filep=one_day_0.tsv --vfilep=valid_1M.tsv --max_steps=1000000 --save_steps=100000

# 14 train and 7 valid
python -m preprocess.get_meta --dpath=/data/yunfanhu/samples_20 --filep=train.tsv --vfilep=valid.tsv --chunk_size=50000
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
                    --save_path=cps_noid \
                    --with_id=0 \
                    --plots=plots/baseline.jpg \
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
                                --m0=cps_noid/baseline_new/res.csv \
                                --m1=cps_20/m1_new/res.csv \
                                --points=500 \
                                --sample=quantile

python -m utils.plot_one_cali --spath=plots/one.jpg \
                                --res=cps2/res.csv \
                                --points=500 \
                                --sample=quantile

# train lightgbm for importance
python train_lightgmb.py  --dpath=/data/yunfanhu/samples \
                    --filep=train_5M.tsv \
                    --vfilep=valid_1M.tsv
# process bigger files            
python -m preprocess.get_meta --dpath=/data/yunfanhu/samples_20 \
                            --filep=train.tsv \
                            --vfilep=valid.tsv \
                            --chunk_size=50000

python -m preprocess.process_meta --dpath=/data/yunfanhu/samples_20 \
                                    --drop_num=10
# build files for tree
python nn_predict_single.py --dpath=/data/yunfanhu/samples_20 \
                            --save_path=cps_20 \
                            --checkpoint=cps_20/m1_0821_raw/pytorch_model.bin \
                            --resp=train_prob.tsv \
                            --filep=train.tsv \
                            --total_len=543886254

python nn_predict_single.py --dpath=/data/yunfanhu/samples_20 \
                            --save_path=cps_20 \
                            --checkpoint=cps_20/m1_0821_raw/pytorch_model.bin \
                            --resp=valid_prob.tsv \
                            --filep=valid_5M.tsv \
                            --total_len=5000000


