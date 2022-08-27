# first get the output of neural network
# and transform
python -m transform.py --dpath=/data/yunfanhu/samples_20/ \
                        --prob_path=/data/yunfanhu/prob/ \
                        --out_path=/data/yunfanhu/gbm/

# prepare LightGBM
pip install cmake
hash -r
# install CLI version of LightGBM
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build
cd build
cmake ..
make -j4

# make new directory
cd ../examples/
mkdir fr
cd fr

# put the train.conf and predict.conf into fr folder
# first time run
"../../lightgbm" config=train.conf \
                is_save_binary_file=true \
                data=/data/yunfanhu/gbm/train.tsv \
                valid_data=/data/yunfanhu/gbm/valid.tsv

# first time load data from text files
# label_column ignore_column categorical_feature is needed
# save binary will save smaller binary dataset with same_name.bin
# second time load data from binary files, three settings is no longer needed
"../../lightgbm" config=train.conf \
                is_save_binary_file=false \
                data=/data/yunfanhu/gbm/train.tsv.bin \
                valid_data=/data/yunfanhu/gbm/valid.tsv.bin

# predict
# use the binary file saved before
"../../lightgbm" config=predict.conf \
                data=/data/yunfanhu/gbm/valid.tsv \
                input_model=LightGBM_model.txt \
                output_result=/data/yunfanhu/gbm/LightGBM_predict_result.txt
# calculation metrics
python -m lightgbm.cal_metric --prob_path=/data/yunfanhu/prob/ \
                            --gbm=/data/yunfanhu/gbm/LightGBM_predict_result.txt \
                            --label_file=/data/yunfanhu/gbm/valid.tsv \ 
                            --plots=plots/LightGBM.jpg 

# binary classification
"../../lightgbm" config=train_cls.conf \
                is_save_binary_file=true \
                data=/data/yunfanhu/gbm_cls/train.tsv \
                valid_data=/data/yunfanhu/gbm_cls/valid.tsv

