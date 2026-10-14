## Project Status

HSQformer is a research project for ultrasound-based liver lesion classification and early hepatocellular carcinoma screening. The repository contains research code and is still being organized; paths, dependencies, and experiment settings may need to be adapted to your environment.

This project is intended for research and is not a medical device. Its outputs must not be used as the sole basis for clinical diagnosis or treatment.

## Latest News

- [2026/10/14] Updated the experiment reproduction logs. Logs for all five folds are available under [`logs/tensorboard/Replication experiment on October-14-2026/`](logs/tensorboard/Replication%20experiment%20on%20October-14-2026/).
- [2026/10/12] Model checkpoints are available in the [HSQformer Hugging Face model repository](https://huggingface.co/chaoyinshe/HSQformer), under `checkpoint/`.
- [2025/09/04] The US-Liver dataset was released on Hugging Face: [chaoyinshe/US-Liver](https://huggingface.co/datasets/chaoyinshe/US-Liver).
- [2025/08/03] The related study “AI enhanced diagnostic accuracy and workload reduction in hepatocellular carcinoma screening” was published in *npj Digital Medicine*: [read the paper](https://www.nature.com/articles/s41746-025-01892-9).

## Model Checkpoints

The October 12, 2026 checkpoints are hosted at:

- [Hugging Face model page](https://huggingface.co/chaoyinshe/HSQformer)
- [Checkpoint directory](https://huggingface.co/chaoyinshe/HSQformer/tree/main/checkpoint)

The directory contains five `.pth` checkpoint files named with the `val0` through `val4` suffixes. Select the checkpoint that matches your evaluation protocol and use it with the corresponding model configuration in this repository.

To download the checkpoints, install `huggingface_hub` and run:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="chaoyinshe/HSQformer",
    repo_type="model",
    allow_patterns="checkpoint/*.pth",
    local_dir="./weights",
)
```

The repository does not currently provide a standalone inference script. Loading a checkpoint requires constructing the matching model architecture and preprocessing inputs in the same way as the experiment that produced it.

## Reproduction Logs

The experiment reproduction logs were updated on **October 14, 2026**. TensorBoard logs are organized by fold (`fold_0` through `fold_4`) in [`logs/tensorboard/Replication experiment on October-14-2026/`](logs/tensorboard/Replication%20experiment%20on%20October-14-2026/). Open the parent directory with TensorBoard to inspect the recorded training and validation metrics:

```bash
tensorboard --logdir "logs/tensorboard/Replication experiment on October-14-2026"
```

## Dataset and Data Preparation

The US-Liver dataset is available from [Hugging Face Datasets](https://huggingface.co/datasets/chaoyinshe/US-Liver). The repository also contains CSV manifests under `dataset/`, but these manifests may refer to image paths that are specific to the original environment. Make sure the image files are available and update the paths in the manifests before training or evaluation.

Dataset usage is restricted to academic research and is not permitted for commercial use. Please follow all applicable laws, institutional ethics requirements, and the dataset's terms. Dataset terms are separate from the software license.

## Getting Started

### Requirements

Dependencies and tested version combinations are not yet pinned in a requirements file. The training code uses PyTorch, torchvision, timm, torchmetrics, pandas, scikit-learn, and other scientific Python packages. Set up compatible versions for your platform before running the experiments.

### Training

`train.py` is the training entry point. Inspect its available options with:

```bash
python train.py --help
```

The training script expects fold-specific CSV files (for example, `train_0.csv` and `val_0.csv`) and additional test manifests in the directory passed through `--data_dir`. Each manifest must point to image files that are accessible on your machine. The default `--data_dir` is an absolute path from the original Linux environment, so specify your local dataset directory.

For example:

```bash
python train.py --data_dir dataset --checkpoints checkpoints/my_experiment
```

The script defaults to five-fold training. Set `--checkpoints` to a new output directory, as in the example, to keep the released checkpoints in `checkpoint/` unchanged. Training logs are written to `logs/` by default.

## Project and Data Terms

The source code is distributed under the [Apache License 2.0](LICENSE). Dataset access and use are governed by the dataset's own terms; the academic-only and non-commercial restriction described above applies to the dataset.

## About the Study

Ethical certification date: May 15, 2023
Certifying institution: The Third Affiliated Hospital of Sun Yat-sen University

---

## Citation

If you use this project in your research, please cite the following papers:

```bibtex
@article{lu2025ai,
  title={AI enhanced diagnostic accuracy and workload reduction in hepatocellular carcinoma screening},
  author={Lu, Rui-Fang and She, Chao-Yin and He, Dan-Ni and Cheng, Mei-Qing and Wang, Ying and Huang, Hui and Lin, Ya-Dan and Lv, Jia-Yi and Qin, Si and Liu, Ze-Zhi and others},
  journal={npj Digital Medicine},
  volume={8},
  number={1},
  pages={500},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
@misc{she2025retrospectivesystematicstudyhierarchical,
      title={A Retrospective Systematic Study on Hierarchical Sparse Query Transformer-assisted Ultrasound Screening for Early Hepatocellular Carcinoma}, 
      author={Chaoyin She and Ruifang Lu and Danni He and Jiayi Lv and Yadan Lin and Meiqing Cheng and Hui Huang and Lida Chen and Wei Wang and Qinghua Huang},
      year={2025},
      eprint={2502.03772},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.03772}, 
}

```
