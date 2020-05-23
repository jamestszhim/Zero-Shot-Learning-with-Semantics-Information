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


## Text Classification on Emotion dataset
The code experiments on two types of text zero-shot classification, partial unseen case (Text_DeVise) and fully unseen case (GLMC) [6], where for partial unseen case, 5 out of 9 classes were used to trained the text-DeVise model, for fully unseen cases, the pre-trained GPT-2 [7] was used without any fine-tuning. To verify the performance of zero-shot classification, we have also added a LSTM classifier trained supervisedly on 9 classes for comparison.

We used the emotion dataset [5]. You can find the splited dataset for this project in the <code>./nlp/Dataset/emotion</code>, or you can download the original dataset of the paper from https://github.com/yinwenpeng/BenchmarkingZeroShot


### Sentence Feature
For text-DeVise, Sentence-BERT [4] was used to extract sentence's features. The feature pickle files are too large to included. To regenerate the feature pickle files, please run the ***LAST TWO*** parts (Transform sentence to sentence vector by s-bert) of <code>./nlp/Dataset_Preprocessing.ipynb</code>

### Code

#### Dataset Preprocessing 
<code>./nlp/Dataset_Preprocessing.ipynb</code>
The jupyter notebook is a self-explainary, step-by-step data preprocessing guide. By running it sequentially, it cleaned and splited the raw emotion dataset, create label vectors by GloVe [3] and transform sentence from text to sentence vector by Sentence-BERT [4]

#### GRU Classifier
<code>./nlp/GRU_CLS.ipynb</code>
The jupyter notebook is a self-explainary, step-by-step experiment of LSTM Classifier. By running it sequentially, it would train a supevised LSTM Classifier on full classes, seen classes, and unseen classes respectively. And then it would claculated the top1, top2, top3 accuracies and f1 scores of the three models with comfusion matrix plotted.

#### Text DeVise
<code>./nlp/Text_DeVise.ipynb</code>
The jupyter notebook is a self-explainary, step-by-step experiment of Text DeVise, which is a variation of DeVise [1] on text. By running it sequentially, it would train Text DeVise unseen classes under various settings, and plot the accuracy and loss curves. By mannul inspecting and selecting the best model, it would claculated the top1, top2, top3 accuracies and f1 scores with comfusion matrix plotted. At last, it also plot the t-SNE of the sentences after mappting to the common semantic space. 


#### Generative Language Model Classifier (GLMC)
<code>./nlp/GLMC.ipynb</code>
The jupyter notebook is a self-explainary, step-by-step experiment of GLMC. By running it sequentially, it first defined the input format of GPT-2 model, and then applied the input to the pre-trained GPT-2, and selected the class label with highest score during the next token prediction stage as the predicted class. After iterate through the whole dataset, it would return top1, top2, top3 accuracies and f1 scores with confusion matrix plotted.




## Citation
```
[1]
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

[2]
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

[3]
@inproceedings{pennington2014glove,
  author = {Jeffrey Pennington and Richard Socher and Christopher D. Manning},
  booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},
  title = {GloVe: Global Vectors for Word Representation},
  year = {2014},
  pages = {1532--1543},
  url = {http://www.aclweb.org/anthology/D14-1162},
}

[4]
@article{Reimers_2019,
   title={Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
   url={http://dx.doi.org/10.18653/v1/d19-1410},
   DOI={10.18653/v1/d19-1410},
   journal={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
   publisher={Association for Computational Linguistics},
   author={Reimers, Nils and Gurevych, Iryna},
   year={2019}
}

[5]
@misc{yin2019benchmarking,
    title={Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach},
    author={Wenpeng Yin and Jamaal Hay and Dan Roth},
    year={2019},
    eprint={1909.00161},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

[6]
@misc{puri2019zeroshot,
    title={Zero-shot Text Classification With Generative Language Models},
    author={Raul Puri and Bryan Catanzaro},
    year={2019},
    eprint={1912.10165},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

[7]
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```
