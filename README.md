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
## 2. Download checkpoints
Please follow [README.md](./checkpoints/README.md) in checkpoints folder.
## 3. Launch app server
```bash
python app.py
```
You can visit the application at localhost:9996 in your browser, chrome is recommended。
![demo](./figs/demo.gif)
## 4. Train and Finetune uLLSAM
If you want to reproduce uLLSAM, just use the ./data/train_seg_all.jsonl to train the model, you need to prepare 9 datasets.

You can refer to [torch_em](https://github.com/constantinpape/torch-em/tree/main/torch_em/data/datasets) to prepare and download datasets.
```bash
bash ./scripts/train_all_joint_v2.sh
```
If you want to finetune your custom data, follow the data structure in ./data/train_seg_all.jsonl

Specifically, each line in jsonl is structured as {"image_path": "...", "conversation": \[{"role": "user", "content": "Describe the image in detail\n<image>"}, {"role": "assistant", "content": ""}\]}
