## Data
Data can be downloaded [here](https://drive.google.com/file/d/1uMuwUi-n-kklwemubHT_Im2curUUDW0w/view?usp=sharing)

## SetUp
```
conda create -n vlgrammar python=3.7 pytorch=1.7.1 torchvision -c pytorch
conda activate vlgrammar

pip install -r requirements.txt

git clone --branch infer_pos_tag https://github.com/zhaoyanpeng/pytorch-struct.git
cd pytorch-struct
pip install -e 
```

## Clustering
```
cd SCAN
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_partit_chair.yml
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_partit_chair.yml
```
or use our pretrained model

## Grammar Induction
```
cd VLGrammr
python train.py or python train.py --type chair
```

## Checkpoints
Model checkpoints can be downloaded [here](https://drive.google.com/file/d/1nA3Daqh-zH6Abmv8Q9QGE5Tl1VcyCBcr/view?usp=sharing)

## Citation
	@misc{hong2021vlgrammar,
	      title={VLGrammar: Grounded Grammar Induction of Vision and Language}, 
	      author={Yining Hong and Qing Li and Song-Chun Zhu and Siyuan Huang},
	      year={2021},
	      eprint={2103.12975},
	      archivePrefix={arXiv},
	      primaryClass={cs.CV}
	}
[paper](https://arxiv.org/abs/2103.12975)
