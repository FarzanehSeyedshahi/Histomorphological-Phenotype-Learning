# A histomorphological atlas of resected mesothelioma from 3446 whole-slide images discovered by self-supervised learning
*[Link](https://arxiv.org/abs/2205.01931)*

---

**Abstract:**

*Mesothelioma is a highly lethal and poorly biologically understood disease which presents diagnostic challenges due to its morphological complexity. This study uses self-supervised AI (Artificial Intelligence) to map the histomorphological landscape of the disease. The resulting atlas consists of recurrent patterns identified from 3446 Hematoxylin and Eosin (H\&E) stained images scanned from resected tumour slides. These patterns generate highly interpretable predictions, achieving state-of-the-art performance with 0.65 concordance index (c-index) for outcomes and 85% AUC in subtyping. Their clinical relevance is endorsed by comprehensive human pathological assessment. Furthermore, we characterise the molecular underpinnings of these diverse, meaningful, predictive patterns. Our approach both improves diagnosis and deepens our understanding of mesothelioma biology, highlighting the power of this self-learning method in clinical applications and scientific discovery.*

---

## Citation
```
@article{seyedshahi2024histomorphological,
  title={A histomorphological atlas of resected mesothelioma from 3446 whole-slide images discovered by self-supervised learning},
  author={Seyedshahi, Farzaneh and Rakovic, Kai and Poulain, Nicolas and Quiros, Adalberto Claudio and Powley, Ian R and Klebe, Sonja and Richards, Cathy and Uraiby, Hussein and Nakas, Apostolos and Wilson, Claire and others},
  journal={bioRxiv},
  pages={2024--11},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

---

## Repository overview

In this repository you will find the following sections: 
1. [WSI tiling process](#WSI-tiling-process): Instructions on how to create H5 files from WSI tiles.
2. [Workspace setup](#Workspace-setup): Details on H5 file content and directory structure.
3. [HPL instructions](./README_HPL.md): Step-by-step instructions on how to run the complete methodology.
   1. Self-supervised Barlow Twins training.
   2. Tile vector representations.
   3. Combination of all sets into one H5.
   4. Fold cross-validation files.
   5. Include metadata in the H5 file.
   6. Leiden clustering.
   7. Removing background tiles.
   8. HPC configuration selection.
   9. Logistic regression for mesothelioma type WSI classification.
   10. Cox proportional hazards for survival prediction.
   11. Correlation between annotations and HPCs.
   12. Get tiles and WSI samples for HPCs.

4. [LATTICe-M HPL files](#TCGA-HPL-files): HPL output files from our paper results.
5. [Python Environment](#Python-Environment): Python version and packages.

---


## Workspace setup 
*You can find the full details on HPL instructions in this [Readme_HPL file](README_HPL.md).*

## HPL Mesothelioma files
This section contains the following LATTICe-M files produced by HPL:
1. WSI tile image datasets.
2. Self-supervised trained weights.
3. LATTICe-M tile projections.
4. LATTICe-M HPC configurations.
5. LATTICe-M WSI & patient representations. 

### TCGA WSI tile image datasets
You can find the WSI tile images at:
2. [LATTICe-M training subsample]() for self-supervised model training.


### LATTICe-M Pretrained Models
[Self-supervised model weights - trained on LATTICe-M](https://figshare.com/articles/dataset/Phenotype_Representation_Learning_PRL_-_LUAD_LUSC_5x/19715020)

### LATTICe-M tile vector representations
You can find tile projections for LATTICe-M cohort at the following location. This is the projections used in the publication results: [LATTICe-M tile vector representations (background and artifact tiles unfiltered)]()


### LATTICe-M HPC files
You can find HPC configurations used in the publication results at:
1. [Background and artifact removal](https://drive.google.com/drive/folders/1K0F0rfKb2I_DJgmxYGl6skeQXWqFAGL4?usp=sharing)
2. [Sarc/Biph Vs. Epi type classification](https://drive.google.com/drive/folders/1TcwIJuSNGl4GC-rT3jh_5cqML7hGR0Ht?usp=sharing)
3. [Mesothelioma survival](https://drive.google.com/drive/folders/1CaB1UArfvkAUxGkR5hv9eD9CMDqJhIIO?usp=sharing)


### WSI & patient vector representations
You can find WSI and patient vector representations used in the publication results at:
1. [Subtype classification](https://drive.google.com/file/d/1K2Fteuv0UrTF856vnJMr4DSyrlqu_vop/view?usp=sharing)


## Python Environment
The code uses Python 3.8 and the necessary packages can be found at [requirements.txt](./requirements.txt)

The flow uses TensorFlow 1.15 and according to [TensorFlows Specs](https://www.tensorflow.org/install/source#gpu) the closest CUDA and cuDNN version are `cudatoolkits==10.0` and `cudnn=7.6.0`. 
However, depending on your GPU card you might need to use `cudatoolkits==11.7` and `cudnn=8.0` instead. 
Newer cards with Ampere architecture (Nvidia 30s or A100s) would only work with CUDA 11.X, Nvidia maintains this [repo](https://github.com/NVIDIA/tensorflow), so you can use TensorFlow 1.15 with the new version of CUDA.

These commands should get the right environment to run HPL:
```
conda create -n HPL python=3.8 \ 
conda activate HPL \
python3 -m pip install --user nvidia-pyindex \
python3 -m pip install --user nvidia-tensorflow \
python3 -m pip install -r requirements.txt \
```



