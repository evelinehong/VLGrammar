## Data
Data can be downloaded [here](https://drive.google.com/file/d/1DuGfZvV3RXAanR_2engzm_tZDb2LmrxK/view?usp=sharing)

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
Model checkpoints can be downloaded [here](https://drive.google.com/file/d/1giLbR2HRJHhhMcpujNVjaUdRTlug-Ptm/view?usp=sharing)

## Citation
	@misc{hong2021vlgrammar,
	      title={VLGrammar: Grounded Grammar Induction of Vision and Language}, 
	      author={Yining Hong and Qing Li and Song-Chun Zhu and Siyuan Huang},
	      year={2021},
	      journal={ICCV},
	}
[paper](https://arxiv.org/abs/2103.12975)

## Acknowledgements
Parts of the codes are based on [vpcfg](https://github.com/zhaoyanpeng/vpcfg) and [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification)
