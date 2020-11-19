# Pragmatic Issue Sensitive Image Captioning (ISIC)

This is the companion repo for EMNLP 2020 Findings [paper](https://arxiv.org/abs/2004.14451): 

`Pragmatic Issue-Sensitive Image Captioning
Nie, A. Cohn-Gordon, R., and Potts, C. (2020). arXiv preprint arXiv:2004.14451.` 

## Introducing Issue Sensitive Image Captioning (ISIC)

Issue Sensitive Image Captioning (ISIC) is a task where we specify an issue for a caption (generative) model,
and the model is required to generate a caption that discusses the specified issue.

We define issue broadly, as any concept that can generate a partition of images. Issues are domain-specific.
For example, in the Caltech-UCSD Birds dataset, issue is defined as a body part of the bird, because difference in body part can 
give rise to the partition (birds with similar body parts, and birds without similar body parts).

In the MSCOCO dataset, we define issue as a VQA question, because the answer to the VQA question `"Red" = VQA(Image, "What is the color of the wall?"`
can produce a partition of images (images with red walls and images without red walls).

We extend a popular probabilistic model (Rational Speech Act) that is widely used to model various
linguistic pragmatic behaviors (vagueness, generics, presupposition, question under discussion).
We make the vanilla RSA model issue-sensitive by imposing equivalence structure (cell structure) into the partition.
We further introduce a novel entropy penalty to the RSA model to penalize spurious generation.

Our partition generation method and decoding method can be extended to other generative models including language modeling, dialogue, machine translation, etc.
You can find our implementation and evaluation methods in this repo and our paper [here](https://arxiv.org/abs/2004.14451). 

## Installation

The CUB captioning model is modified from https://github.com/salaniz/pytorch-gve-lrcn

The installation guide comes from Salaniz repo. The data downloading link provided from the original repo is broken. We host a separate data downloading
 source from AWS.
 
1.Clone the repository
```shell
git clone https://github.com/windweller/Pragmatic-ISIC.git
cd Pragmatic-ISIC
```
2.Create conda environment
```shell
conda env create -f environment.yml
```
3.Activate environment
```shell
conda activate gve-lrcn
```

4.Download pre-trained model and data
```bash
sh rsa-file-setup.sh 
```

5. Install other packages

```bash
pip install -r requirements.txt
```

## CUB

Our main experiment's code is under the `cub` folder, referring to the Caltech-UCSD Birds dataset.
The training code of S0 model (base model) is adapted from [repo](https://github.com/salaniz/pytorch-gve-lrcn). 

We have complete evaluation pipeline and interactive jupyter notebook (to be released soon).

## MSCOCO

We have implemented our pragmatic decoder on a very popular state-of-the-art image captioning repo. Even though we do not have
quantitative experiment, we made the code and notebook available so that our Pragmatic caption decoder can be used by other
researchers.

This will be made available very shortly (if you need the code now, please email).

## RSA Re-ranking Visualization

During the process of developing Pragmatic ISIC model, we  developed companion tools
that help us debug our implementation and visualize the Bayesian re-ranking process.

The RSA computation can be thought of as a series of probabilistic re-weighting of each word's generation probability.
We care about the relative rank of each word compared to other words in the vocabulary.

 

## FAQ

The PyTorch version used in this code is not the latest version. In fact, if you use the latest version,
some "type error" might occur during sentence decoding. Be aware. It is recommended to create
a unique conda environment to run this code.

## Contact

Please contact anie@stanford.edu if you have problem using these scripts! Thank you!
