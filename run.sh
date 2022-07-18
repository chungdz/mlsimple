mkdir data cps
python -m preprocess.get_meta
python train.py
python train.py --with_id=0