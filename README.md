# Zero Shot Learning with Semantics Information
The repository demonstrates a few established zero shot learning methods in image and text classification tasks.

## Image Classification on AwA2 dataset
The code experiments different label embeddings, from manually extracted binary and continous attributes, to semantic attribute embeddings. A large part of the implementation follows the model DeVise [1]. We use AwA2 dataset for the experiment. You can find the detail about the dataset in https://cvml.ist.ac.at/AwA2/ [2].

### Image Feature
Download from http://cvml.ist.ac.at/AwA2/AwA2-features.zip and unzip it under <code> ./image/datasets/awa/Features </code>.

### Run
```
python ./image/train_awa.py -e bin -d cos
```
The code computes the top1, top3, top5 accuracy on the test classes. <code>-e</code> specifies the label embedding schemes. <code>-d</code> specifies the distance metric used in the prediction phase. It can be cosine distance (<code>cos</code>) or L2 distance (<code>l2</code>). 

The different embedding schemes are listed below:
- <code>bin</code>: Binary attributes
- <code>cont</code>: Continous attributes
- <code>text</code>: Summation of GloVe embedding of attribute labels [3]
- <code>weighted_text</code>: Weighted sum of GloVe embedding of attribute labels
- <code>label</code>: GloVe embedding of class labels
- <code>bert_text</code>: Sentence BERT embedding of attribute labels [4]
- <code>bert_wiki</code>: Sentence BERT embedding of text desciption of the class label

## Citation
```
@incollection{NIPS2013_5204,
title = {DeViSE: A Deep Visual-Semantic Embedding Model},
author = {Frome, Andrea and Corrado, Greg S and Shlens, Jon and Bengio, Samy and Dean, Jeff and Ranzato, Marc\textquotesingle Aurelio and Mikolov, Tomas},
booktitle = {Advances in Neural Information Processing Systems 26},
editor = {C. J. C. Burges and L. Bottou and M. Welling and Z. Ghahramani and K. Q. Weinberger},
pages = {2121--2129},
year = {2013},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf}
}

@article{Xian_2019,
   title={Zero-Shot Learning—A Comprehensive Evaluation of the Good, the Bad and the Ugly},
   volume={41},
   ISSN={1939-3539},
   url={http://dx.doi.org/10.1109/TPAMI.2018.2857768},
   DOI={10.1109/tpami.2018.2857768},
   number={9},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Xian, Yongqin and Lampert, Christoph H. and Schiele, Bernt and Akata, Zeynep},
   year={2019},
   month={Sep},
   pages={2251–2265}
}

@inproceedings{pennington2014glove,
  author = {Jeffrey Pennington and Richard Socher and Christopher D. Manning},
  booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},
  title = {GloVe: Global Vectors for Word Representation},
  year = {2014},
  pages = {1532--1543},
  url = {http://www.aclweb.org/anthology/D14-1162},
}

@article{Reimers_2019,
   title={Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
   url={http://dx.doi.org/10.18653/v1/d19-1410},
   DOI={10.18653/v1/d19-1410},
   journal={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
   publisher={Association for Computational Linguistics},
   author={Reimers, Nils and Gurevych, Iryna},
   year={2019}
}
```
