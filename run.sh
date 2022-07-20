source ~/.bashrc

mkdir data cps
python -m preprocess.get_meta

python train.py
python train.py --with_id=0

python -m preprocess.get_meta --dpath=/work/yunfanhu/mannual
python train.py --dpath=/work/yunfanhu/mannual

