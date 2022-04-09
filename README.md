# CL_NER
Continual Learning with Structured Regularization in Named Entity Recognition

## Requirements
- PyTorch
- numpy
- tqdm
- seqeval
- gensim
- Jupyter Notebook

## Experiments
To replicate the experiments:
1. Clone the repository. 

2. Under the folder `experiments`, locate the specific experiments to replicate (NA - New Addition, Seq - Sequential).

3. [Download](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g) a copy of Google word2vec pre-trained embedding, put under "NER/checkpoints/GoogleNews-vectors-negative300.bin" or specify the path in configuration of each train script.

4. Three scripts are enlisted in each directory (Baseline vs. EWC vs. SI). Inside, "multiple_allowed" controls whether to purge sentences with multiple entity tags. Run `python3 ewc_train.py` for example, for training and evaluation.

## Acknowledgement
The copy of CONLL 2003 dataset is referred [here](https://huggingface.co/datasets/conll2003) from huggingface.
