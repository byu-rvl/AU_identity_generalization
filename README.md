# Beyond 41 Identities: Generalizing Facial Action Unit Recognition for Downstream Success

Authors:
[Andrew Sumsion](https://www.linkedin.com/in/drew-sumsion/)
[Dah-Jye Lee](https://ece.byu.edu/directory/d-j-lee)

[Paper Link]()

## Environment Setup

To recreate this environment, ensure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed, then run:

```bash
# Create the environment from the .yml file
conda env create -f environment.yml

# Activate the environment
conda activate fsrt
```

2. Install PyTorch (GPU Support)
To ensure hardware acceleration, we install the exact versions of Torch 2.8.0 and Torchvision 0.23.0 specifically compiled for CUDA 12.6.

```Bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
```
**Different CUDA Versions:** > If you have a different version of CUDA, find the appropriate command on the Official PyTorch Get Started Page. For older versions of PyTorch, refer to the Previous PyTorch Versions list.

## Download Model Weights

TODO: Add download for model weights.

## Run Prediction On An Image

TODO: Add in ability to crop the image.

```Bash
python run.py --checkpoint path/to/downloaded/model/weights.pth --img path/to/image
```

## Method Overview:

### Contribution Overview:
![Contribution overview](images/Fig1.jpg)

### Training vs. Evaluation:
![Training vs. Evaluation](images/Fig2.jpg)

### Identity Transfer Trainings:
![Identity Transfer trainings](images/Fig3.jpg)

### Overall Training
![Overall Training](images/Fig4.jpg)

Contributions:
1. We present an identity transfer method to AU recognition that increases generalization for downstream tasks. Specifically, we increase the identities trained on from a max of 41 identities in a lab setting to an additional 5,556 identities taken from the wild, an over 125 times increase in number of identities. 

2. We provide a denser prediction for AU recognition. We increase the number of predicted AUs to 15 over previous models that predict only 8 or 12 AUs. 

3. We surpass other available open source AU recognition models on the combined average performance on the BP4D \cite{zhang2014bp4d, zhang2013high} and DISFA \cite{mavadati2013disfa, mavadati2012automatic} datasets on the overlapping AUs between the datasets. We also achieve the highest average performance between all 12 AUs from the BP4D dataset and all 8 from the DISFA dataset.

4. We emphasize our model's increased generalization and usefulness by comparing our model's performance with other open source models on three downstream tasks. We outperform all other available open source AU recognition models for all three downstream tasks, while previous methods only perform well on one or a few of the downstream tasks. In addition, we increase the accuracy by up to 6.3 percentage points.

## BibTeX

```bibtex
@misc{yourproject2024,
  author = {Your Name},
  title = {My Awesome Project},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/my-awesome-project}},
}
