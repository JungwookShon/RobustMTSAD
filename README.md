# RobustMTSAD
Toward a robust approach to multivariate time series anomaly detection

## Usage
1. Install Python 3.6 or higher.
2. Install library.
```bash
pip install -r requirements.txt
```
3. Download dataset. 
   SMD dataset is already in dataset directory. This example is based on SMD dataset.
4. Run normalization.
```bash
python preprocess_AggZNorm_SMD.py
```
4. Train and evaluate. 
```bash
bash ./scripts/SMD.sh
```
