# uLLSAM
Official Implement of uLLSAM

## 1. Installation guaidance
```bash
git clone git@github.com:ieellee/uLLSAM.git
cd ./uLLSAM
conda env create -f environment.yml
conda activate ullsam
```
If your encounter some unexpected errors, you can also refer to [InvernVL](https://github.com/OpenGVLab/InternVL/tree/main) and [SAM](https://github.com/facebookresearch/segment-anything) to install your own environment.
## 2. Launch app server
```bash
python app.py
```
You can visit the application at localhost:9996 in your browser, chrome is recommended。
## 3. Train and Finetune uLLSAM
