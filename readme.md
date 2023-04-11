**This repository contains the code for SemEval 23 Task 11 Learning with Disagreement**


For codes submitted to the SemEval 2023 shared task (final submission), please check out the codes in *final_submitted_codes* folder

We further made some experiments. The codes can be found in the *all_code_dumped* folder


## How to run the codes:

** Please modify the paths for data


1. For Brexit disagreement learning (dis-learning):

without metadata: ```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/Brexit_dis_learning.py --batch_size 8 --dropout 0.1 --epochs 4 --hidden_size 32 --lr 5e-5 --use_metadata no```

with metadata: ```python3 /Le-wi-di-semeval-23/SafeWebUH_codes_paper/Brexit_dis_learning.py --batch_size 8 --dropout 0.1 --epochs 4 --hidden_size 32 --lr 5e-5 --use_metadata yes```
