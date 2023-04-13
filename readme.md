**This repository contains the code for SemEval 23 Task 11 Learning with Disagreement**

The code in this repo is used for the paper titled *SafeWebUH at SemEval-2023 Task 11: Learning Annotator Disagreement in Derogatory Text: Comparison of Direct Training vs Aggregation* 

Please check out the folder called **SafeWebUH_codes_paper**.


## How to run the codes:

** Please modify the paths for data

** Please note, there is post-aggregation learning for HS-Brexit, and ArMIS only, since the annotators are inconsistent across other two datasets. 

1. For Brexit disagreement learning (dis-learning):

without metadata: ```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/Brexit_dis_learning.py --batch_size 8 --dropout 0.1 --epochs 4 --hidden_size 32 --lr 5e-5 --use_metadata no```

with metadata: ```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/Brexit_dis_learning.py --batch_size 8 --dropout 0.1 --epochs 4 --hidden_size 32 --lr 5e-5 --use_metadata yes```

2. For Brexit post-aggregation learning (Post-Agg):

without metadata: ```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/Brexit_post_agg.py --batch_size 8 --dropout 0.1 --epochs 4 --hidden_size 32 --lr 5e-5 --use_metadata no```

with metadata: ```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/Brexit_post_agg.py --batch_size 8 --dropout 0.1 --epochs 4 --hidden_size 32 --lr 5e-5 --use_metadata yes```

3. For ConvAbuse  disagreement learning (dis-learning):  

without metadata: ```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/ConvAbuse_dis_learning.py --batch_size 8 --dropout 0.1 --epochs 7 --hidden_size 32 --lr 5e-5 --use_metadata no```

with metadata: ```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/ConvAbuse_dis_learning.py --batch_size 8 --dropout 0.1 --epochs 7 --hidden_size 32 --lr 5e-5 --use_metadata yes```

4. For MD-Agreement disagreement learning (dis-learning) (No metadata)

```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/MD_dis_learning.py --batch_size 8 --dropout 0.1 --epochs 4 --hidden_size 32 --lr 5e-5```

5. For ArMIS disagreement learning (dis-learning) (No metadata)

```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/Armis_dis_learning.py --batch_size 8 --dropout 0 --epochs 7 --hidden_size 32 --lr 5e-5```

6. For ArMIS post-aggregation learning (Post-Agg) (No metadata)

witho ```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/Armis_post_agg.py --batch_size 8 --dropout 0.1 --hidden_size 32 --lr 5e-5``

### How to cite:

Coming Soon !

**Reference**

I got the help on coding from a huge number of papers, repos, blogs, tutorials and many other online resources. If you think, I missed some references, please let me know. 

1. https://www.tensorflow.org/text/tutorials/classify_text_with_bert

2. https://github.com/philschmid/deep-learning-pytorch-huggingface

3. https://github.com/xuyige/BERT4doc-Classification

4. https://github.com/abyanjan/Fine-Tune-BERT-for-Text-Classification
