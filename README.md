<p align="center">
  <img src="docs/teaser_font_larger_01.png" width="700">
  <h1 align="center">Stairway to Success: An Online Floor-Aware Zero-Shot Object-Goal Navigation Framework via LLM-Driven Coarse-to-Fine Exploration</h1>
  <h3 align="center">
    <a href="https://zeying-gong.github.io/">Zeying Gong</a><sup>1</sup>, 
    <a href="https://rongli.tech/">Rong Li</a><sup>1</sup>, 
    <a href="https://hutslib.github.io/">Tianshuai Hu</a><sup>2</sup>, 
    <a href="https://openreview.net/profile?id=~Ronghe_Qiu2">Ronghe Qiu</a><sup>1</sup>, 
    <a href="https://ldkong.com/">Lingdong Kong</a><sup>3</sup>,
    <br>
    <a href="https://scholar.google.com/citations?user=nzfJ4mEAAAAJ&hl">Lingfeng Zhang</a><sup>4</sup>, 
    <a href="https://guoyangzhao.github.io/">Guoyang Zhao</a><sup>1</sup>, 
    <a href="https://www.linkedin.com/in/ding-justin-08a421266">Yiyi Ding</a><sup>1</sup>, 
    <a href="https://junweiliang.me/">Junwei Liang</a><sup>1,2,&#9993</sup>
    <br>
    <p>
        <h45>
            <sup>1</sup> The Hong Kong University of Science and Technology (Guangzhou). &nbsp;&nbsp;
            <br>
            <sup>2</sup> The Hong Kong University of Science and Technology &nbsp;&nbsp;
            <br>
            <sup>3</sup> National University of Singapore &nbsp;&nbsp;
            <br>
            <sup>4</sup> Tsinghua University &nbsp;&nbsp;
            <br>
        </h45>
    </p>

  </h3>

  <!-- Badges -->
  <p align="center">
    <a href="https://zeying-gong.github.io/projects/ascent/">
      <img src="https://img.shields.io/badge/Web-Ascent-deepgreen.svg" alt="Project Web Badge">
    </a>
    <a href="https://www.youtube.com/watch?v=1uqS-aMk-tE">
      <img src="https://img.shields.io/badge/Video-Youtube-red.svg" alt="YouTube Video Badge">
    </a>
    <a href="https://arxiv.org/abs/2505.23019">
      <img src="https://img.shields.io/badge/cs.ai-arxiv:2505.23019-42ba94.svg" alt="arXiv Paper Badge">
    </a>
    <a href="https://github.com/facebookresearch/habitat-sim">
      <img src="https://img.shields.io/static/v1?label=supports&message=Habitat%20Sim&color=informational" alt="Habitat Sim Badge">
    </a>
    <a href="https://github.com/Zeying-Gong/habitat-lab/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License Badge">
    </a>
  </p>
</p>

## 📋 TODO List

- ✅ Complete Installation and Usage documentation
- ✅ Add datasets download documentation
- ✅ Release the main algorithm of **ASCENT**
- ❌ Release the code of real-world deployment

## :hammer_and_wrench: Environment Setup

### 1. **Preparing Conda Environment**

Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda env:
```
conda_env_name=ascent_nav
conda create -n $conda_env_name python=3.9 cmake=3.14.0
conda activate $conda_env_name
```

Install proper version of torch:
```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### 2. **Install Habitat-Sim**

```
conda install habitat-sim=0.3.1 withbullet headless -c conda-forge -c aihabitat
```

> If you encounter network problems, you can manually download the Conda package from [this link](https://anaconda.org/aihabitat/habitat-sim/0.3.1/download/linux-64/habitat-sim-0.3.1-py3.9_headless_bullet_linux_3d6d67d6deae4ab2472cc84df7a3cef1503f606d.tar.bz2) to download the conda bag, and install it via: `conda install --use-local /path/to/xxx.tar.bz2` to download.

In theory, versions >= 0.2.4 are all compatible, but it is better to keep the same version between habitat-lab and habitat-sim. Here we use 0.3.1 version.

### 3. **Clone Repository**
```
git clone --recurse-submodules https://github.com/Zeying-Gong/ascent.git
```

### 4. **Install Habitat-Lab**
```
cd third_party/habitat-lab
git checkout v0.3.1
pip install -e habitat-lab
pip install -e habitat-baselines
cd ../..
```

### 4. **GroundingDINO**

Following [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)'s instruction:

```
export CUDA_HOME=/path/to/cuda-11.8 # replace with actual path

cd third_party/GroundingDINO
pip install -e . --no-build-isolation --no-dependencies
cd ../..
```

### 5. **MobileSAM**

Following [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)'s instruction:

```
cd third_party/MobileSAM
pip install -e .
cd ../..
```

### 6. **Others**
```
pip install -r requirements.txt
```

The following dependencies require special build flags:
```bash
pip install transformers==4.37.0
```


## :weight_lifting: Downloading Model Weights

Download the required model weights and save them to the `pretrained_weights/` directory:

| Model | Filename | Download Link |
|-------|----------|---------------|
| Places365 | `resnet50_places365.pth.tar` | [Download](http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar) |
| MobileSAM | `mobile_sam.pt` | [GitHub](https://github.com/ChaoningZhang/MobileSAM) |
| GroundingDINO | `groundingdino_swint_ogc.pth` | [GitHub](https://github.com/IDEA-Research/GroundingDINO) |
| D-FINE | `dfine_x_obj2coco.pth` | [GitHub](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj2coco.pth) |
| RedNet | `rednet_semmap_mp3d_40.pth` | [Google Drive](https://drive.google.com/file/d/1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv) |
| RAM++ | `ram_plus_swin_large_14m.pth` | [HuggingFace](https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth) |

### Qwen2.5-7B Weights
Through [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) or [ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct) download the checkpoints, and put them in `pretrained_weights/`

### PointNav Weights

The PointNav weight is directly from [VLFM](https://github.com/bdaiinstitute/vlfm), located in `third_party/vlfm/data/pointnav_weights.pth`. 

- Locate Datasets: The file structure should look like this:
```
pretrained_weights
├── mobile_sam.pt
├── groundingdino_swint_ogc.pth
├── dfine_x_obj2coco.pth
├── ram_plus_swin_large_14m.pth
├── rednet_semmap_mp3d_40.pth
├── resnet50_places365.pth.tar
└── Qwen2.5-7b
    ├── model-00001-of-00005.safetensors
    └── ...
```

## 📚 Datasets Setup

- Download Scene & Episode Datasets: Following the instructions for **HM3D** and **MP3D** in Habitat-lab's [Datasets.md](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md).

- Locate Datasets: The file structure should look like this:
```
data
└── datasets
    ├── objectnav
    │   ├── hm3d
    │   │   └── v1
    │   │        └── val
    │   │             ├── content
    │   │             └── val.json.gz
    │   └── mp3d
    │       └── v1
    │            └── val
    │                 ├── content
    │                 └── val.json.gz
    └── scene_datasets
        ├── hm3d
        │   └── ...
        └── mp3d
            └── ...
```

## 🚀 Evaluation

Run VLM servers
```
./scripts/launch_vlm_servers_ascent.sh
```

It will open a tmux windows in a separate terminal.

Open another terminal, run evaluation on HM3D dataset:
``` 
python -u -m ascent.run --config-name=eval_ascent_hm3d.yaml
```

Or run evaluation on MP3D dataset:
``` 
python -u -m ascent.run --config-name=eval_ascent_mp3d.yaml
```

## ⚠️ Notes

- This is a **refactored version** of the original codebase with improved code organization and structure.
- Due to the inherent randomness in **object detection** (GroundingDINO, D-FINE) and **LLM inference** (Qwen2.5), evaluation results may vary slightly from the paper's reported metrics.

## :black_nib: Citation

If you use **ASCENT** in your research, please use the following BibTeX entry.

```
@article{gong2025stairway,
  title={Stairway to Success: Zero-Shot Floor-Aware Object-Goal Navigation via LLM-Driven Coarse-to-Fine Exploration},
  author={Gong, Zeying and Li, Rong and Hu, Tianshuai and Qiu, Ronghe and Kong, Lingdong and Zhang, Lingfeng and Ding, Yiyi and Zhang, Leying and Liang, Junwei},
  journal={arXiv preprint arXiv:2505.23019},
  year={2025}
}
```

## 	:pray: Acknowledgments

We would like to thank the following repositories for their contributions:
- [ApexNav](https://github.com/Robotics-STAR-Lab/ApexNav)
- [VLFM](https://github.com/bdaiinstitute/vlfm)
- [L3MVN](https://github.com/ybgdgh/L3MVN)
- [SeeGround](https://github.com/iris0329/SeeGround)
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)