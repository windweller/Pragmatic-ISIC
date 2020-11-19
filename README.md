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

<img src="https://github.com/windweller/Pragmatic-ISIC/raw/master/misc/isic_fig4.png" width="600" height="684" alt="bird caption"/>

In the MSCOCO dataset, we define issue as a VQA question, because the answer to the VQA question `"Red" = VQA(Image, "What is the color of the wall?"`
can produce a partition of images (images with red walls and images without red walls).

![MSCOCO](https://github.com/windweller/Pragmatic-ISIC/raw/master/misc/isic_fig1.png)

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

## CUB Dataset

Our main experiment's code is under the `cub` folder, referring to the Caltech-UCSD Birds dataset.
The training code of S0 model (base model) is adapted from [repo](https://github.com/salaniz/pytorch-gve-lrcn). 

You can run an interactive version of our code in `interactive.ipynb`, where you can specify an issue and generate the caption.

The complete evaluation pipeline will be released soon.

## MSCOCO Dataset

We have implemented our pragmatic decoder on a very popular state-of-the-art image captioning [repo](https://github.com/ruotianluo/self-critical.pytorch). 
Even though we do not have
quantitative experiment, we made the code and notebook available so that our Pragmatic caption decoder can be used by other
researchers.

This version of code integrated incremental RSA decoding with the beam search. We are happy to share our RSA beam search decoder in this repo (available soon).

## RSA Re-ranking Visualization

During the process of developing Pragmatic ISIC model, we  developed companion tools
that help us debug our implementation and visualize the Bayesian re-ranking process.

The RSA computation can be thought of as a series of probabilistic re-weighting of each word's generation probability.
We care about the relative rank of each word compared to other words in the vocabulary.

We built a tool to visualize how each step of computation in our model is affecting the relative ranking of words.
In this example, we are visualizing a list of words `['eye', 'superciliary',
  'stripe',
  'yellow-silver',
  'stripes',
  'streak',
  'beack']` at position 11 of our generated caption. We can see that although at first, "superciliary" is ranked higher than the rest,
  eventually after re-weighting, the probability distribution of S1 has "streak" ranked higher. 
  
```python
from rsa import IncRSADebugger
debugger = IncRSADebugger(model, rsa_dataset)

debugger.visualize_words_decision_paths_at_timestep(11, ['eye',
  'superciliary',
  'stripe',
  'yellow-silver',
  'stripes',
  'streak',
  'beack'])
```

![word_comparison_ranking2](https://github.com/windweller/Pragmatic-ISIC/raw/master/misc/word_comparison_ranking2.png)

We can also visualize the ranking of one word over many positions of the generated caption. This allows you to see how the probability
of generating a single word increases or decreases along the generation process at each time step.

```python
debugger.visualize_word_decision_path_at_timesteps("eye")
```

![word_across_time](https://github.com/windweller/Pragmatic-ISIC/raw/master/misc/word_across_time.png)

At last, this debugger will check the if the implementation of model is correct or not:

```python
debugger.run_full_checks()
```
```
S0 - The following value should be 1: tensor(1.0000, device='cuda:0')
L1 - The following value should be 1: tensor(1., device='cuda:0')
U1 - The following value should be less than 1: tensor(0.6973, device='cuda:0')
L1 QuD - The following value should be 1: tensor(1., device='cuda:0')
U2 - The following two values should equal 3.1263315677642822 == 3.126331329345703
S0 - The following value should be 1: tensor(1., device='cuda:0')
```

## FAQ

The PyTorch version used in this code is not the latest version. In fact, if you use the latest version,
some "type error" might occur during sentence decoding. Be aware. It is recommended to create
a unique conda environment to run this code.

## Contact

Please contact anie@stanford.edu if you have problem using these scripts! Thank you!
