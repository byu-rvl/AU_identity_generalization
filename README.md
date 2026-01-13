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

Following previous manners, we did a 3 fold cross validation for the cross-corpus evaluation. As such, we release all three folds. However, for downstream applications that need the best generalized model, we suggest using fold 3 for predicting AUs.

Fold | Recommendation | Link
--- | --- | ---
1 | Good, but not the best one. | [Huggingface Link](https://huggingface.co/dwsum/GenBGen_weights/tree/main/propWithFsrt_folds_smoothing0_3_proportion1_0_--limitFsrtLoss_numEncoderLayers3_batchsize256/fold1)
2 | Good, but not the best one. | [Huggingface Link](https://huggingface.co/dwsum/GenBGen_weights/tree/main/propWithFsrt_folds_smoothing0_3_proportion1_0_--limitFsrtLoss_numEncoderLayers3_batchsize256/fold2)
3 | Recommendation for best downstream task performance. | [Huggingface Link](https://huggingface.co/dwsum/GenBGen_weights/tree/main/propWithFsrt_folds_smoothing0_3_proportion1_0_--limitFsrtLoss_numEncoderLayers3_batchsize256/fold3)

## Run Prediction On An Image

To run the network exactly as trained crop the image using the code in preprocessing/Face_crop_align_mtcnn_mtcnn_detector.py. Once the face is cropped run:
```Bash
python run.py --checkpoint path/to/downloaded/model/weights.pth --img path/to/image
```

To run on an uncropped image you can instead run this code. (Note, the cropping is slightly different, but should be very similar).
```Bash
python run.py --checkpoint path/to/downloaded/model/weights.pth --img path/to/image --crop
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
