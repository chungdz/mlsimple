# Description
The source code is for click prediction model experiments.

# Package Introduction
- gscope: the code for generate scope scripts to fetch dataset.
- nn_modules: the code for neural network in Pytorch.
- prepocess: the code for processing dataset.
- utils: the code for tools.

# Prepare Data
Data is stored in DLTS Azure-EastUS-P40-2 /data/yunfanhu/. They are first generated in Cosmos and then moved to DLTS. Since I generate the data day by day. The final files for training and validation are concatenated in DLTS. Training datasets are from 2022/03/24 to 2022/04/06. Validation datasets are from 2022/04/07/ to 2022/04/13.

Original data: Combine counting features and id features. It is stored in Cosmos and never moved to DLTS. The scope scripts to make the original data are generated by original_data.ipynb in gscope folder.

D1: Dataset uniformly sampled from original data and sample rate is 6%. It is stored in /data/yunfanhu/samples/ as train.tsv and valid.tsv. Use valid_3M.tsv to do validation. The scope scripts to make the original data are generated by D1.ipynb in gscope folder.

D2: Dataset that has 5 million rows for training and 1 million rows for validation. It is used for fast experiments. It is stored in /data/yunfanhu/samples/ as train_5M.tsv and valid_1M.tsv. It is sampled from D1.

D3: Dataset uniformly sampled from original data and sample rate is 20%. It is stored in /data/yunfanhu/samples_20/ as train.tsv and valid.tsv. Use valid_5M.tsv to do validation. The scope scripts to make the original data are generated by D3.ipynb in gscope folder.

D4: D1 adding UGE embeddings. It is stored in /data/yunfanhu/samples_emb/ as train.tsv and valid.tsv. Use valid_5M.tsv to do validation. The scope scripts to make the original data are generated by D4.ipynb in gscope folder.

D5: Dataset has all positive samples from original dataset and downsampled negative sample. The number of rows of D5 is 20% of total. It is stored in /data/yunfanhu/downsamples_20/ as train.tsv and valid.tsv. Use valid_5M.tsv to do validation. The scope scripts to make the original data are generated by D5.ipynb in gscope folder.

# Set Environments
Python version should be equal or larger than 3.8. Currently Python 3.9 is used.

Download Miniconda to deploy python environment on the [website](https://docs.conda.io/en/latest/miniconda.html).

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Change permission mode if no execution permission and run it to install Miniconda.

```shell
Chmod 700 Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

After the installation finish, it will ask whether to start conda from bash. Type yes.

To activate conda:

```shell
source ~/.bashrc
```

Current Python environment for base is 3.9.

Pytorch=1.12.1 is needed. For DLTS with CUDA version 11.0 (supported by DeepScale2.1-Regular framework), the instruction is:
```shell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu110
```

Instruction to install other packages.
```shell
pip install datasets transformers tensorboard sklearn lightgbm matplotlib ipython
``` 

# Get meta infomation of different features
Before train the model, meta infomation should be gathered to help transform the data. Dataset with UGE embeddings and without UGE embeddings are now save in different folder. The meta data for them should be collect separately.

Min and max of the counting features are collected. It is better than mean and standard diviation for normalizing the counting feature. First, min and max are easy to collect and do not need to calculate. Second, for features only have zero and one, the normalized featrue stays the same. Third, for features has only one unique number, it avoids number divided by zero.

The frequencies of each unique ID are gathered. The IDs appears less than threshold are map to *Unknown*.

The meta information is stored as Json format and in the same folder of the train.tsv.

The meta Json file is like:
metadict = {
    "all_features": list of counting feature names,
    "to_minus": list of number for counting feature normalization,
    "to_div": list of number for counting feature normalization,
    "all_ids": list of ID feature names,
    "dicts": list of vocabulary for each ID feature,
    "total_count": total row number of dataset
}

## For dataset without UGE embeddings

The example instructions are:

```shell
python -m preprocess.get_meta --dpath=/data/yunfanhu/samples_20 \
                            --filep=train.tsv \
                            --vfilep=valid.tsv \
                            --chunk_size=50000

python -m preprocess.process_meta --dpath=/data/yunfanhu/samples_20 \
                                    --drop_num=10
``` 

The meaning of each argument can be found in python code or use *-h*

## For dataset with UGE embeddings

The example instructions are:

```
python -m preprocess.get_meta_emb --dpath=/data/yunfanhu/samples_emb \
                                --filep=train.tsv \
                                --vfilep=valid.tsv \
                                --chunk_size=50000
python -m preprocess.process_meta --dpath=/data/yunfanhu/samples_emb \
                                    --drop_num=10

```

The meaning of each argument can be found in python code or use *-h*

# Training and testing

The framework of training use huggingface framework. It automatically uses all GPUs in environment. If no GPUs then CPU is used. Checkpoints and runtime information are saved automatically in the target path and are named pytorch_model.bin and trainer_state.json respectively.

## To train the model without UGE embeddings
Two codes can be run:

```shell
python train.py --dpath=/data/yunfanhu/samples_20 \
                    --batch_size=2 \
                    --chunk_size=2048 \
                    --filep=train.tsv \
                    --vfilep=valid_5M.tsv \
                    --max_steps=300000 \
                    --save_path=cps_20 \
                    --plots=plots/m1_20.jpg \
                    --with_id=1 \
                    --save_steps=30000
```

The meaning of each argument can be found in python code or use *-h*. This code iteratively fetch data chunk from disk to avoid memory problem when dataset is too large. The chunk size can not be too large or too small. Either case makes GPUs hungry. The arguments related to steps need to be calculate before training start. 

For example, the D3 dataset has 543886254 rows. With number of GPUs in environment is 4, batch size is 2, and chunk size is 2048, the rows consumed by one step is 4 * 2 * 2048 = 16384. Therefore the step for one epoch is 543886254 / 16384 = 33197. Therefore I set max_steps as 300000 and save_steps as 30000, so that the model runs about 10 epoch and save the checkpoints per epoch.

The second way to train is to fetch all data once into memory:

```shell
python train_rand.py --dpath=/data/yunfanhu/samples \
                    --batch_size=512 \
                    --filep=train_5M.tsv \
                    --vfilep=valid_1M.tsv \
                    --epoch=5 \
                    --with_id=1 \
                    --plots=plots/m1_samples.jpg \
                    --save_path=cps_small

```
The meaning of each argument can be found in python code or use *-h*. Here the rows consumed by one step is batch_size * GPU number. The model is trained and saved epoch by epoch. It is not suggested to use the second way if dataset is large.

If argument with_id is set to zero, then model with no ID features is used. 

## To train the model with UGE embeddings:

```shell
python train_emb.py --dpath=/data/yunfanhu/samples_emb \
                    --batch_size=2 \
                    --chunk_size=2048 \
                    --filep=train.tsv \
                    --vfilep=valid_5M.tsv \
                    --max_steps=300000 \
                    --save_path=cps_emb \
                    --plots=plots/m1_emb.jpg \
                    --save_steps=30000
```
The usage of arguments is similar to *train.py*.

# Evaluation

## Metrics
Five metrics are used during training process: ROC AUC, MRR, nDCG, Precison-Recall AUC, RIG. For details, check *utils/metrics.py*.

## Calibration Plots
After training process finished, parameters with best ROC AUC are loaded, and predictions are made for validation dataset. The result is saved as *res.csv* in the save_path, which can be used to plot calibration plot and do other analysis. The result has two columns, labels and predictions.

Calibration plot is saved in target path after training is finished. But if want to manually plot based on result file:

```shell
python -m utils.plot_one_cali --spath=plots/one.jpg \
                                --res=cps2/res.csv \
                                --points=500 \
                                --sample=quantile
```
The meaning of each argument can be found in python code or use *-h*. 

Also, the result of two models can be plot together:

```shell
python -m utils.plot_two_cali --spath=plots/two.jpg \
                                --m0=cps_noid/baseline_new/res.csv \
                                --m1=cps_20/m1_new/res.csv \
                                --points=500 \
                                --sample=quantile
```
The meaning of each argument can be found in python code or use *-h*. 

## lightgbm for find feature importance
LightGBM is used to calculate feature importance.

```shell
python train_lightgmb.py  --dpath=/data/yunfanhu/samples \
                    --filep=train_5M.tsv \
                    --vfilep=valid_1M.tsv \
                    --sfilep=para/fimp.tsv
```
The meaning of each argument can be found in python code or use *-h*. 

## Get prediction of training and validation dataset
Get the prediction result of training and validation dataset. The model checkpoints with best results should be set in the arguments. Use single CPU/GPU to do this.

```shell
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

```
