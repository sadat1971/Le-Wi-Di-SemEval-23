**This repository contains the code for SemEval 23 Task 11 Learning with Disagreement**


For codes submitted to the SemEval 2023 shared task (final submission), please check out the codes in *final_submitted_codes* folder

We further made some experiments. The codes can be found in the *all_code_dumped* folder


## How to run the codes:

** Please modify the paths for data

** Please note, there is post-aggregation learning for HS-Brexit only, since the annotators are inconsistent across other datasets. 

1. For Brexit disagreement learning (dis-learning):

without metadata: ```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/Brexit_dis_learning.py --batch_size 8 --dropout 0.1 --epochs 4 --hidden_size 32 --lr 5e-5 --use_metadata no```

with metadata: ```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/Brexit_dis_learning.py --batch_size 8 --dropout 0.1 --epochs 4 --hidden_size 32 --lr 5e-5 --use_metadata yes```

2. For Brexit post-aggregation learning (Post-Agg):

without metadata: ```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/Brexit_post_agg.py --batch_size 8 --dropout 0.1 --epochs 4 --hidden_size 32 --lr 5e-5 --use_metadata no```

with metadata: ```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/Brexit_post_agg.py --batch_size 8 --dropout 0.1 --epochs 4 --hidden_size 32 --lr 5e-5 --use_metadata yes```

3. For ConvAbuse  disagreement learning (dis-learning):  

without metadata: ```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/ConvAbuse_dis_learning.py --batch_size 8 --dropout 0.1 --epochs 7 --hidden_size 32 --lr 5e-5 --use_metadata no```

with metadata: ```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/ConvAbuse_dis_learning.py --batch_size 8 --dropout 0.1 --epochs 7 --hidden_size 32 --lr 5e-5 --use_metadata yes```


**Reference**

I got the help on coding from a huge number of papers, repos, blogs, tutorials and many other online resources. If you think, I missed some references, please let me know. 

1. https://www.tensorflow.org/text/tutorials/classify_text_with_bert

2. https://github.com/philschmid/deep-learning-pytorch-huggingface

3. https://github.com/xuyige/BERT4doc-Classification

4. https://github.com/abyanjan/Fine-Tune-BERT-for-Text-Classification
