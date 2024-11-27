# Thai Universal Dependency Treebank (TUD)
**Thai Universal Dependency Treebank (TUD)** is a Thai treebank consisting of 3,627 trees annotated using the [Universal Dependencies (UD)](https://universaldependencies.org/guidelines.html) framework. TUD includes 15 part-of-speech tags and 35 dependency relations, as shown in the distribution table below. The documents used for annotation were randomly sampled from the *Thai National Corpus (TNC)* and *the November 2020 dump of Thai Wikipedia*, covering a wide range of document types and topics. The process of constructing the treebank and benchmarks for 92 dependency parsing models' detail are in the paper *"The Thai Universal Dependency Treebank"*, published in Transactions of the Association for Computational Linguistics. 

<center>
<div style="display: grid; grid-template-columns: 30% 40% ; gap: 0px 50px ;justify-content: center;">

|UPOS|Train|Dev|Test|
|:----|-----:|---:|----:|
|NOUN|18777|2270|2310|
|VERB|14881|1802|1867|
|ADP|4517|530|560|
|ADV|4498|557|521|
|AUX|3424|401|421|
|PRON|2796|322|350|
|SCONJ|2438|321|335|
|PROPN|2488|293|295|
|CCONJ|2063|239|270|
|ADJ|1575|223|197|
|PART|1366|156|169|
|NUM|1161|165|118|
|DET|1140|137|144|
|PUNCT|871|104|125|
|SYM|16|1|1|



|DEPREL|Train|Dev|Test||DEPREL|Train|Dev|Test|
|:------------------|----:|---:|---:|-|:------------------|----:|---:|---:|
|nmod|6268|781|810||punct|865|104|122|
|obj|5474|655|663||cop|834|83|94|
|advmod|5366|692|644||flat|709|72|88|
|compound|5272|656|666||clf|539|79|74|
|nsubj|4529|548|568||fixed|442|69|60|
|acl|4539|485|563||xcomp|349|43|46|
|case|4442|522|548||list|348|29|21|
|root|2902|362|363||dep|74|5|7|
|obl|2811|328|322||discourse|67|10|8|
|mark|2720|326|360||dislocated|64|11|8|
|aux|2548|311|319||orphan|71|3|6|
|conj|1992|208|249||csubj|59|11|9|
|cc|1898|221|254||appos|63|5|9|
|advcl|1784|225|211||iobj|50|5|6|
|amod|1449|215|168||parataxis|27|2|3|
|ccomp|1304|168|174||expl|14|2|2|
|det|1117|135|145||vocative|1|1|1|
|nummod|1020|149|92|||||
</div>

<div style="display: grid; grid-template-columns: 30% 40% ; gap: 0px 50px ;justify-content: center;">

*Table 1 : UPOS (Universal Part-Of-Speech) distribution.*

*Table 2 : DEPREL (Dependency relationships) distribution.*
</div>
</center>

## Content
This repository consisting of 3 parts.
1. **TUD**: contains the treebank itself, including both the original full treebank and its train-dev-test splits, and various statistics about the treebank.

```{figure} img/dependency_length.png
:align: center
:width: 40%

```
<center>

 *Figure 1 : Dependency length distribution in TUD.*
</center><br>

2. **Experiment**: contains the code used in our experiment, including the specific train-dev-test splits of [Thai-PUD](https://github.com/UniversalDependencies/UD_Thai-PUD) used in our experiment.
3. **Prediction**: contains the full test split prediction of the 92 models trained in our experiment and their confusion matrices.


## Experiment Results
Thai dependency parsing models evaluated in the experiments can be categorized into two categories: (1) baseline models and (2) open-source models. All models used treebank gold-standard tokenization. Two types of parsers were tested: (1) Transition-based parsers and (2) Graph-based parsers. The evaluation results are in Figure 2. 

```{figure} img/model_eval.png
:align: center
:width: 70%
```

<center>

*Figure 2 : Evaluation results of each model on each treebank’s test split. T = Transition-based, G = Graph-based, S = Arc-standard, E = Arc-eager, A = Augmented with sentence and super token embeddings, W = Wangchan-
BERTa, P = PhayaThaiBERT.* <br> *Open-source models are all graph-based.*

</center>

<div style="display: grid; grid-template-columns: 70% 30% ; gap: 30% 0px ;justify-content: center;">

```{figure} img/linear_regression.png
:align: center
:width: 100%
```

```{figure} img/UPOS_tag.png
:align: center
:width: 90%
```
</div>
<div style="display: grid; grid-template-columns: 70% 30% ; gap: 30% 0px ;justify-content: center;">

*Figure 3 : Table5: Linear regression results for UAS(R2=0.54) and LAS(R2=0.507). 
The reference categories are baseline models graph-based architecture, WangchanBERTa as encoder, non-augmented, and agnostic UPOS.*

*Figure 4 : F1 scores of our UPOS taggers on each label and treebank. W=WangchanBERTa. P=PhayathaiBERT. ADP tag is used instead for SCONJ in Thai-PUD*

</div>
<br>

The results of the experiments can answer the questions below (1-4) and the challenges unique to Thai dependency parsing are also addressed (5).

### 1. Which parsing architecture is better for Thai?
`Transition-based model`
1. Transition-based models perform significantly better than the graph-based models in UAS (unlabeled attachment score) but perform similarly in LAS (label attachment score).
2. Even though Stanza is a graph-based model, having additionally unique techniques such as using text's static pretrained word embeddings to augment the token's representation, and including terms that model the probability of each link between a head and a dependent based on their distance and linear order, Stanza is the overall best model.

### 2. Is PhayaThaiBERT a better Thai encoder than WangchanBERTa?
`Yes`
1. Results show that PhayaThaiBERT performs better than WangchanBERTa in dependency parsing. (p < 0.05)
2. Large, contextualized, language-specific, pretrained encoders are important. Compared to previous Thai dependency parsers, which do not use pretrained language models, models utilizing pretrained language models in this experiment perform better.

### 3. Do sentence embeddings and super-token embeddings help?
`No`  Significant improvement in using sentence embeddings and super-token embeddings in token embeddings augmentation is not found in this work. Even though some improvements were spotted, the method does not improve models consistently enough to be statistically significant, **the condition in which the method works best should be investigated further**.

### 4. Do gold-standard UPOS tags play an important role?
`Yes` While Gold-standard UPOS tags lead to significantly superior performance (p < 0.05), automatically tagged UPOS does not show a significant improvement and even shows a slight degradation. The results show the **need to improve the POS taggers for dependency parsing improvement**.

### 5. Challenges in Thai dependency parsing. 

```{figure} img/common_confusions.png
:align: center
:width: 70%
```
<center>

*Figure : 4 Top ten of common confusions made by the taggers for UPOS and the parsers for DEPREL along with their most frequentlt associated tokens.*
</center>

Six challanges unique to Thai dependency parsing are identified in this work. 
1. **Polyfunctional words**:Most UPOS confusions were caused by the polyfunctional words. 
2. **Common nouns and Proper nouns**:The distinction between common nouns and proper nouns in Thai is unclear causing the confusion between the NOUN and PRON tags.
3. **Absence of SUBJ-VERB agreement & Pro-drop nature**: Thai relative pronouns are difficult to determine without world knowledge due to the pro-drop nature and absence of subject-verb agreement in Thai.
4. **Compounds and Syntactic phrases**: Parsers cannot distinguish between compounds and syntactic phrases easily as the distinction is very subtle.
5. **Shared word forms**: Verbal dependents (xcomp, compound, ccomp, advcl) and verbal-like dependents (advmod, aux) were often confused with one another as verbal-like dependents often share word forms with verbs.
6. **Chain dependencies**: Many mistakes were correlated with *chain dependencies* as some relations allow the tokens of the same UPOS to be chained in one structure, leading to ambiguity when subsequent dependencies need to be attached to one of the tokens in the chain. 



## Citations
If you use TUD in your project or publication, please cite as follows:

BibTex

```
@article{Sriwirote-etal-2024-TUD,
  title={The Thai Universal Dependency Treebank},
  author={Panyut Sriwirote and Wei Qi Leong and 
  Charin Polpanumas and Santhawat Thanyawong  and 
  William Chandra Tjhi and Wirote Aroonmanakun and 
  Attapol T. Rutherford},
  journal={Transactions of the Association for Computational Linguistics},
  year={in press},
  publisher={MIT Press Direct}
}
```