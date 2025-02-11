# RobustMTSAD
Toward a robust approach to multivariate time series anomaly detection

## Usage
1. Install Python 3.6 or higher.
2. Install library.
```bash
pip install -r requirements.txt
```
3. Download dataset. This example is based on SMD dataset.
4. Run normalization.
```bash
python preprocess_AggZNorm_SMD.py
```
4. Train and evaluate. 
```bash
bash ./scripts/SMD.sh
```

## Main Result

We compare our model with 15 baselines, including THOC, InterFusion, etc. **Generally,  Anomaly-Transformer achieves SOTA.**

<p align="center">
<img src=".\pics\result.png" height = "450" alt="" align=center />
</p>

## Citation
If you find this repo useful, please cite our paper. 

```
@inproceedings{
xu2022anomaly,
title={Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy},
author={Jiehui Xu and Haixu Wu and Jianmin Wang and Mingsheng Long},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=LzQQ89U1qm_}
}
```

## Contact
If you have any question, please contact wuhx23@mails.tsinghua.edu.cn.
