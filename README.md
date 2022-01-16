# Code for TitleGen-FL

This is the source code for the paper **"TitleGen-FL: Quality Prediction-based Filter for Automated Issue Title Generation"**.

## Package Requirement

To run this code, some packages are needed as following:
```
torch==1.5.1
torchtext==0.6.0
nltk==3.6.5
gensim==3.8.3
word2vec==0.9.4
rouge==1.0.0
venn==0.1.3
pandas==1.0.1
numpy==1.21.5
```
and any other else used in project.

**Notice:** before run Main.py, you must first download these two folders: **dataset_train_DL_model** and **dataset_train_pos_neg** from the url: https://drive.google.com/drive/folders/1jTEfIY6V6f8HySQDCjWrP-3HxpuXdY5s?usp=sharing and put them in this project path, and we recommend you download all files through this url： https://drive.google.com/drive/folders/1jTEfIY6V6f8HySQDCjWrP-3HxpuXdY5s?usp=sharing, which concludes all of files storage on google drive mentioned below. And, put them into this project's path.

## How to prepare the dataset for TitleGen-FL

### The iTAPE dataset(which is used in our paper and of course provided in this project)

This dataset is origined from the paper《Stay professional and efficient: automatically generate titles for your bug reports》, which published in ASE 2020, and you can download it from the url: **https://drive.google.com/drive/folders/1y1Slg5D03iQ6uCQPrtTyd_fLCSVJsh5F?usp=sharing** and find it in this path: **"./iTAPE_dataset"**. It is worth noting that the TXT document with".pred" in its name is the title of the corresponding body generated by using the trained iTAPE model, not the file that comes with the iTAPE data set. We provide our trained iTAPE model and you can download through the url: **https://drive.google.com/drive/folders/1v1O39zjl5i9zETjaBXyjHu8DR7jY-heW?usp=sharing**, find it in this path: **"./pre_trained_models/ ITape_step_25000.pt"**. If you need train one new iTAPE model by yourself, you can visit the iTAPE paper project home page on Github.

### If you want to use your own dataset

Follow the format of the dataset in **"./iTAPE_dataset"**, which you can find through the url: **https://drive.google.com/drive/folders/1y1Slg5D03iQ6uCQPrtTyd_fLCSVJsh5F?usp=sharing**.

## How to run the code

Only need to change the path and parameters in the **Main.py**.

### Use our pretrained models
First, copy the model .pkl file you want to use from the folder: **"./pre_trained_models"**(you can find it through the url: https://drive.google.com/drive/folders/1v1O39zjl5i9zETjaBXyjHu8DR7jY-heW?usp=sharing) and paste it into the folder: **"./done_model"**.
Second, follow the annotation, change the content in list **"model_names"**, and change the first parameter of function **"DL_component_bleu_threshold"** to **'use'**.
Third, run the Main.py and get final results in the folder: **"./result_last"**.

Train new models
First, follow the annotation, change the content in list "model_names" into the model name you want(we provide five choices: **'CNN', 'RNN', 'RCNN', 'RNN_Attention', 'Transformer'**), and change the first parameter of function **"DL_component_bleu_threshold"** to **'train'**.
Second, run the **Main.py** and get final results in the folder: **"./result_last"**.

## How long the code need to run
We have introduced the CPU and GPU(RTX 3090) in our paper. In fact, under mine circumstance, for DL module, it will cost about half to an hour to train a new model or about 10-15 min to use a pretrained model and get result. For IR module, it will cost at least 1h15min to get the result, to save your time, the IR module result also can download through the url: https://drive.google.com/drive/folders/11Vtl66VEfqEpabfMxER2HHq1gYDlLzCM?usp=sharing, to use it, you only need use the **load_data()** function in utils.tools.py.

## Acknowledgements

I would like to thank my tutor for his guidance, all those who participated in this paper, and all those who read this paper and visited this Github project. Let's strive to make a greater contribution to the field of software engineering. Thank you very much!

## References

For more details about this project, please refer to the "code comments" and our paper "TitleGen-FL: Quality Prediction-based Filter for Automated Issue Title Generation".
