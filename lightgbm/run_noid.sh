# Get the prediction of neural network for both trainset and validset
# the train_prob.csv and valid_prob.csv should be moved to same folder
python nn_predict_single.py --dpath=/data/yunfanhu/samples_20 \
                            --save_path=cps_noid \
                            --checkpoint=cps_noid/baseline_new/pytorch_model.bin \
                            --resp=train.tsv.bin.init \
                            --filep=train.tsv \
                            --with_id=0 \
                            --total_len=543886254

python nn_predict_single.py --dpath=/data/yunfanhu/samples_20 \
                            --save_path=cps_noid \
                            --checkpoint=cps_noid/baseline_new/pytorch_model.bin \
                            --resp=valid.tsv.bin.init \
                            --filep=valid_5M.tsv \
                            --with_id=0 \
                            --total_len=5000000

"../../lightgbm" config=train_cls.conf \
                is_save_binary_file=false \
                data=/data/yunfanhu/gbm_cls_noid/train.tsv.bin \
                valid_data=/data/yunfanhu/gbm_cls_noid/valid.tsv.bin

"../../lightgbm" config=train.conf \
                is_save_binary_file=false \
                data=/work/tmp2/train.tsv \
                valid_data=/work/tmp2/valid.tsv

