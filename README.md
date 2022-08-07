# DeepPhageFinder
DeepPhageFinder is a tool for identifying (pro)phage nucleotides from massive metagenomic assembly data. It is developed based on [DeepVirFinder](https://github.com/jessieren/DeepVirFinder).
## Description  
DeepPhageFinder is a tool developed to identify (pro)phage nucleotides from massive dataset. Its deep learning framework is extremely similar with DeepVirFinder. DeepVirFinder utilize an ingenious and concise CNN to identify virus nucleotides by using k-mers info as feature. Such alignment-free method provides the capability to outperform other virus identification tools in speed. However, its model is relative outdated and its speed could be accelerated by optimizing code. So we developed DeepPhageFinder, a faster and more precise tool which is more applicable to train and predict on large datasets. Unlike DeepVirFinder, DeePhageFinder outputs the score of every 3000 bp nucleotides rather than the average score of whole sequence. Phage domain were then predicted according to the scores to detect (pro)phage and cope with metagenomic assembly data more efficiently. 
## Dependencies
We recommend users to create a virtual environment for DeepPhageFinder.  
`conda create --name dpf python=3.7 numpy keras=2.3.1 tensorflow-gpu=2.2.0 scikit-learn Biopython`  
`source activate dpf`
## Installation
`git clone https://github.com/YuCheng1217/DeepPhageFinder`  
`cd DeepPhageFinder`
## Usage  
The usage of DeepPhageFinder is not so different from [usage of DeepVirFinder](https://github.com/jessieren/DeepVirFinder#usage). And the good news is that multiprocessing([-t] in dpf_encode.py) and multi-GPU training([-g] in dpf_train.py) is available when using DeepPhageFinder to encode and train customized dataset. What's more, options of seed_score, seed_number and extend_score are appended in dpf.py to provide users a modifiable standard to predict pro(phage) area based on score calculated by DeepPhageFinder.
