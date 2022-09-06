# Get the prediction of neural network for both trainset and validset
# the train_prob.csv and valid_prob.csv should be moved to same folder
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

# Also manually make train.tsv.init and valid.tsv.init
# They can be generated based on train_prob.tsv and valid_prob.tsv
# If init score is needed. In this case, the init score is the raw output of the neural network
# The format is one result one line
# Check https://lightgbm.readthedocs.io/en/latest/Parameters.html#continued-training-with-input-score for instruction

# Get the output of neural network and transform
# train.tsv and valid_5M.tsv is in --dpath
# train_prob.csv and valid_prob.csv are in --prob_path
# outputs in --out_path are trainset and validset for lightgbm
python -m preprocess.transform --dpath=/data/yunfanhu/samples_20/ \
                        --prob_path=/data/yunfanhu/prob/ \
                        --out_path=/data/yunfanhu/gbm/

# install newest cmake in DLTS
# use pip install because no sodu access
# use hash -r to update path
pip install cmake
hash -r
# download and compile CLI version of LightGBM
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build
cd build
cmake ..
make -j4

# make new directory to save config files
cd ../examples/
mkdir fr
cd fr

# put the train.conf and predict.conf into fr folder
# first time train
"../../lightgbm" config=train.conf \
                is_save_binary_file=true \
                data=/data/yunfanhu/gbm/train.tsv \
                valid_data=/data/yunfanhu/gbm/valid.tsv

# secpnd time train can be faster by using binary files
# first time load data from text files
# label_column ignore_column categorical_feature is needed
# save binary will save smaller binary dataset with same_name.bin
# second time load data from binary files, three settings is no longer needed
"../../lightgbm" config=train.conf \
                is_save_binary_file=false \
                data=/data/yunfanhu/gbm/train.tsv.bin \
                valid_data=/data/yunfanhu/gbm/valid.tsv.bin

# predict
# can not use the binary file saved before
"../../lightgbm" config=predict.conf \
                data=/data/yunfanhu/gbm/valid.tsv \
                input_model=LightGBM_model.txt \
                output_result=/data/yunfanhu/gbm/LightGBM_predict_result.txt
# calculation metrics
# combine output of neural network and lightgbm and get performance
python -m lightgbm.cal_metric --prob_path=/data/yunfanhu/prob/ \
                            --gbm=/data/yunfanhu/gbm/LightGBM_predict_result.txt \
                            --label_file=/data/yunfanhu/gbm/valid.tsv \
                            --plots=plots/LightGBM.jpg 

# binary classification
"../../lightgbm" config=train_cls.conf \
                is_save_binary_file=true \
                data=/data/yunfanhu/gbm_cls/train.tsv \
                valid_data=/data/yunfanhu/gbm_cls/valid.tsv
# predict
"../../lightgbm" config=predict_cls.conf \
                data=/data/yunfanhu/gbm_cls/valid.tsv \
                input_model=LightGBM_model_cls.txt \
                output_result=/data/yunfanhu/gbm_cls/LightGBM_predict_result.txt
# begging and calculate metric
python -m lightgbm.begging --prob_path=/data/yunfanhu/prob/ \
                            --gbm=/data/yunfanhu/gbm_cls/LightGBM_predict_result.txt \
                            --label_file=/data/yunfanhu/gbm_cls/valid.tsv \
                            --plots=plots/emsemble.jpg 

# binary classification
"../../lightgbm" config=train_cls.conf \
                is_save_binary_file=false \
                data=/data/yunfanhu/gbm_cls_2/train.tsv.bin \
                valid_data=/data/yunfanhu/gbm_cls_2/valid.tsv.bin


