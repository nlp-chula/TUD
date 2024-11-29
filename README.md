# Thai Universal Dependency Treebank (TUD)
**Thai Universal Dependency Treebank (TUD)** is a Thai treebank consisting of 3,627 trees annotated using the [Universal Dependencies (UD)](https://universaldependencies.org/guidelines.html) framework. TUD includes 15 part-of-speech tags and 35 dependency relations, as shown in the distribution table below. The documents used for annotation were randomly sampled from the *Thai National Corpus (TNC)* and *the November 2020 dump of Thai Wikipedia*, covering a wide range of document types and topics. The process of constructing the treebank and benchmarks for 92 dependency parsing models' detail are in the paper *"The Thai Universal Dependency Treebank"*, published in Transactions of the Association for Computational Linguistics. 


<div align="center">
 
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

*Table 1 : UPOS (Universal Part-Of-Speech) distribution.*

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


*Table 2 : DEPREL (Dependency relationships) distribution.*
</div>

## Content
This repository consisting of 3 parts.
1. **TUD**: contains the treebank itself, including both the original full treebank and its train-dev-test splits, and various statistics about the treebank.
2. **Experiment**: contains the code used in our experiment, including the specific train-dev-test splits of [Thai-PUD](https://github.com/UniversalDependencies/UD_Thai-PUD) used in our experiment.
3. **Prediction**: contains the full test split prediction of the 92 models trained in our experiment and their confusion matrices.


## Experiment Results
Thai dependency parsing models evaluated in the experiments can be categorized into two categories: (1) baseline models and (2) open-source models. All models used treebank gold-standard tokenization. Two types of parsers were tested: (1) Transition-based parsers and (2) Graph-based parsers. The evaluation results are in Table 3. 


<div align="center">
<table><tr><th></th><th>Thai-PUD</th><th>TUD</th></tr>
<tr>
<td>

| <br>Model|
|-----------|
| T S ∅ W   | 
| T S A W   |
| T E ∅ W   |
| T E A W   |
| T S ∅ P   |
| T S A P   |
| T E ∅ P   |
| T E A P   |
| G-∅ W     |
| G-A W     |
| G-∅ P     |
| G-A P     |
| UDPipe* W    |
| UDPipe* P    |
| Stanza* W    |
| Stanza* P    |
| Trankit* W   |
| Trankit* P   |

</td>
<td>

| Gold POS<br>UAS  | Gold POS<br>LAS |Auto POS<br>UAS  | Auto POS<br>LAS | No POS<br>UAS  | No POS<br>LAS |
|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| 88.14      | 80.39      | 85.28      | 76.65      | 85.60      | 75.45      | 
| 88.83      | 82.23      | 88.14      | 80.20      | 86.25      | 76.60      |
| 87.40      | 80.53      | 88.00      | 79.60      | 84.54      | 75.03      |
| 88.42      | 81.91      | 87.77      | 80.39      | 86.39      | 78.08      |
| 89.57      | 82.33      | 87.91      | 79.51      | 84.73      | 75.27      |
| 89.43      | 83.48      | 88.28      | 80.94      | 85.65      | 76.70      |
| 89.11      | 82.60      | 88.92      | 80.48      | 86.48      | 78.17      |
| 89.39      | 83.76      | 88.37      | 81.17      | 87.45      | 79.51      |
| 85.97      | 80.43      | 83.43      | 76.60      | 84.36      | 77.34      |
| 87.82      | 82.69      | 86.29      | 79.79      | 83.80      | 76.14      |
| 89.29      | 84.82      | 88.42      | 82.19      | 87.91      | 81.68      |
| 89.80      | 84.91      | 88.65      | 82.60      | 88.74      | 82.05      |
| 88.92      | 83.06      | ----       | ----       | 86.06      | 77.01      |
| 89.89      | 83.53      | ----       | ----       | 86.67      | 77.78      |
| 91.37      | 86.16      | 89.85      | 83.34      | 89.29      | 83.06      |
| **92.02**      | **87.22**      | **90.54**      | **84.54**      | **90.72**      | **84.73**      |
| 89.62      | 84.08      | ----       | ----       | 86.22      | 76.19      |
| 91.28      | 86.11      | ----       | ----       | 86.71      | 77.01      |

</td>
<td>

| Gold POS<br>UAS  | Gold POS<br>LAS |Auto POS<br>UAS  | Auto POS<br>LAS | No POS<br>UAS  | No POS<br>LAS |
|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| 89.47      | 82.60      | 86.27         | 76.22         | 86.59         | 76.81         |
| 89.82      | 83.18      | 86.59         | 76.52         | 86.80         | 76.87         |
| 89.20      | 82.27      | 86.33         | 76.53         | 86.02         | 76.02         |
| 89.41      | 82.62      | 86.24         | 76.70         | 86.37         | 76.55         |
| 90.15      | 83.57      | 87.05         | 77.60         | 87.19         | 77.64         |
| 90.04      | 83.74      | **87.26**         | 77.55         | 87.09         | 77.68         |
| 89.93      | 83.42      | 86.82         | 77.09         | 86.54         | 77.07         |
| 89.77      | 83.42      | 87.00         | **77.68**         | 86.76         | 77.61         |
| 86.33      | 79.64      | 84.25         | 74.59         | 84.77         | 74.41         |
| 87.99      | 81.01      | 81.44         | 71.50         | 85.62         | 75.53         |
| 88.75      | 82.25      | 85.73         | 76.12         | 86.40         | 76.56         |
| 89.48      | 82.98      | 86.03         | 76.40         | 85.84         | 76.14         |
| ----       | ----       | ----          | ----          | ----          | ----          |
| ----       | ----       | ----          | ----          | ----          | ----          |
| 90.12      | 83.30      | 86.31         | 76.60         | 87.01         | 77.39         |
| **90.90**      | **84.54**      | 86.93         | 77.51         | **87.39**         |**78.09**         |
| ----       | ----       | ----          | ----          | ----          | ----          |
| ----       | ----       | ----          | ----          | ----          | ----          |
</td>
</tr>

</table>



*Table 3 : Evaluation results of each model on each treebank’s test split. T = Transition-based, G = Graph-based, S = Arc-standard, E = Arc-eager, A = Augmented with sentence and super token embeddings, W = Wangchan-
BERTa, P = PhayaThaiBERT.* <br> *Open-source models are all graph-based.*

<table><tr><th style="text-align:center">UPOS</th><th style="text-align:center">Thai PUD</th><th style="text-align:center">TUD</th></tr>
<tr>
<td>

|.|
|--------------|
| ADJ          | 
| ADP          |
| ADV          |
| AUX          |
| CCONJ        |
| DET          |
| NOUN         |
| NUM          |
| PART         |
| PRON         |
| PROPN        |
| PUNCT        |
| SCONJ        |
| SYM          |
| VERB         |
| *MacroAverage* |

</td>
<td>

 W        | P        |
|:----------:|:----------:|
| 0.7978   | **0.8508**   |
| 0.9578   | **0.9677**   |
| 0.8528   | **0.8705**   |
| 0.9565   | **0.9710**   |
| 0.9434   | **0.9636**   |
| 0.9469   | **0.9596**   |
| 0.9597   | **0.9711**   |
| **1.0000**   | **1.0000**   |
| 0.9556   | **0.9663**   |
| 0.9552   | **0.9925**   |
| 0.9341   | **0.9375**   |
| **1.0000**   | **1.0000**   |
| -       | -          |
| **1.0000**   | **1.0000**   |
| 0.9502   | **0.9610**   |
| 0.9458   | **0.9580**  |

</td>
<td>

| W        | P        |
|:----------:|:----------:|
| 0.6486   | **0.6852**   |
| 0.9206   | **0.9272**   |
| 0.7665   | **0.7792**   |
| 0.8483   | **0.8508**   |
| 0.8675   | **0.8813**   |
| 0.9007   | **0.9122**   |
| 0.9640   | **0.9672**   |
| **0.9391**   | 0.9264   |
| 0.8395   | **0.8402**   |
| 0.9330   | **0.9418**   |
| 0.9037   | **0.9223**   |
| **0.9881**   | 0.9843   |
|0.8205   | **0.8479**   |
| **1.0000**   | **1.0000**   |
| 0.9240   | **0.9292**   |
| 0.8843   | **0.8930**   |

</td>
</tr>
</table>

*Table 4 : F1 scores of our UPOS taggers on each label and treebank. W=WangchanBERTa. P=PhayathaiBERT. ADP tag is used instead for SCONJ in Thai-PUD*


| Factor                          | Coefficient (UAS) | p-value (UAS)   | Coefficient (LAS) | p-value (LAS)   |
|---------------------------------|-------------------|-----------------|-------------------|-----------------|
| (Intercept)                     | 84.7953           | <0.001***       | 76.0953           | <0.001***       |
| ModelCategory:Open-sourceModels | 2.0221            | <0.001***       | 1.7424            | 0.035*          |
| Architecture:Transition-Standard| 1.0420            | 0.011*          | 0.0563            | 0.937           |
| Architecture:Transition-Eager   | 1.0622            | 0.010*          | 0.4103            | 0.566           |
| Encoder:PhayaThaiBERT           | 1.2665            | <0.001***       | 1.6927            | 0.001**         |
| Augmented:Yes                   | 0.4487            | 0.174           | 0.7599            | 0.195           |
| UPOSQuality:Gold                | 2.2607            | <0.001***       | 4.5311            | <0.001***       |
| UPOSQuality:Auto                | 0.4217            | 0.259           | 0.8011            | 0.227           |

*Table 5 : Linear regression results for UAS(R2=0.54) and LAS(R2=0.507). 
The reference categories are baseline models graph-based architecture, WangchanBERTa as encoder, non-augmented, and agnostic UPOS.*
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


<center>

<table><tr><th style="text-align:center">Thai-PUD</th><th style="text-align:center">TUD</th></tr>
<tr><td>

| Rank | UPOS Confusion         | Tokens                       | DEPREL Confusion     | Tokens                      |
|:------:|:------------------------|:------------------------------|:----------------------|:-----------------------------|
| 1 |NOUN-PROPN            | ดิสแพตช์, ไลน์             | compound-flat:name   | ที่, เซนต์                |
| 2 |ADJ-VERB              | เฉลี่ย, ใกล้ชิด            | acl-xcomp            | ดู, ใช้                     |
| 3 |ADJ-ADV               | ใหม่, น้อย                  | nmod-obl             | ปี, ทะเล                   |
| 4 |ADP-ADV               | กว่า, จึง                   | nsubj-obj            | ที่, ซึ่ง                  |
| 5 |AUX-VERB              | เป็น, ได้                   | obj-obl              | กัน, จรรยาบรรณ             |
| 6 |ADJ-NOUN              | ปัจจุบัน, หนุ่ม             | compound-obj         | ประภาคาร, พื้นฐาน          |
| 7 |ADV-VERB              | พร้อม, สมบูรณ์              | nsubj-obl:tmod       | ที่, อัน                  |
| 8 |ADP-NOUN              | เชิง                        | advcl-root           | ก่อ, แหล่ง                |
| 9 |NOUN-VERB             | ประดิษฐ์, โชว์              | appos-flat:name      | ไมเคิล, ปีเตอร์            |
| 10  |ADP-VERB              | ตั้ง, ต่อ                   | clf-compound         | กลุ่ม, เฮกตาร์             |

</td><td>

| Rank | UPOS Confusion         | Tokens                       | DEPREL Confusion     | Tokens                      |
|:------:|:------------------------|:------------------------------|:----------------------|:-----------------------------|
|1 |ADV-VERB              | มา, ไป                      | compound-nmod        | ประเทศ, สาว                |
|2 |ADV-AUX               | ได้, อยู่                   | nmod-obl             | การ, ประเทศ                 |
|3 |AUX-VERB              | เป็น, ได้                   | advmod-compound      | มา, ไป                     |
|4 |ADJ-VERB              | ดี, ร้าย                    | advmod-aux           | ได้, แล้ว                  |
|5 |NOUN-PROPN            | เมทริกซ์, มะกัน             | acl-compound         | พนัน, เสพ                  |
|6 |NOUN-VERB             | คมนาคม, พนัน                | nsubj-obj            | ที่                         |
|7 |CCONJ-SCONJ           | โดย, ซึ่ง                   | clf-nmod             | คน, แบบ                   |
|8 |SCONJ-VERB            | ให้                         | compound-obj         | การ, ชีวิต                 |
|9 |ADJ-ADV               | มาก, น้อย                   | obj-obl              | ที่, ความ                  |
|10 |ADP-VERB              | ถึง, ให้                    | ccomp-compound       | เรียน, เชื่อม             |

</td></tr></table>

*Table 6 : Top ten of common confusions made by the taggers for UPOS and the parsers for DEPREL along with their most frequentlt associated tokens.*
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
