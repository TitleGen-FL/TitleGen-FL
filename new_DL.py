import nltk
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torch
import re
from decimal import Decimal
import csv
import random
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import os
from torchtext import data
from tqdm import tqdm

from DataSet import getTEXT

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


"""生成深度学习模型训练用数据集（进一步将pos和neg两个部分细分为train、test、valid三个数据集）"""
"""Generating training data set for deep learning model (further subdivide pos and NEG into three data sets train, test and valid)"""
def gen_train_DL_model_dataset(file_dir):
    data_list = []
    all_csv_list = os.listdir(file_dir)
    for single_csv in tqdm(all_csv_list, desc='将pos和neg细分为train、test、valid'):
        with open(os.path.join(file_dir, single_csv), encoding='utf-8') as file:
            for line in file:
                if line == '':
                    continue
                line = line.replace('\0', 'deleted_null_0')
                label = line.replace('\n', '').split('\t')[1]
                if (int(label) == 0):
                    label = 0
                else:
                    label = 1
                sentence = line.replace('\n', '').split('\t')[2]
                data_list.append([sentence, label])

    random.shuffle(data_list)
    # 将全部语料按3:7分为验证集与训练集
    # Split the data into valid set and trainging set as 3 : 7
    n = len(data_list) // 10
    dev_list = data_list[:n * 3]
    train_list = data_list[n * 3:]
    test_list = dev_list

    print('Num of data in training set： {}'.format(str(len(train_list))))
    print('Num of data in valid set： {}'.format(str(len(dev_list))))
    name = ['Sentence', 'Label']
    csv_train = pd.DataFrame(columns=name, data=train_list)
    csv_train.to_csv('dataset_train_DL_model/csv_train.csv', encoding='utf8', index=False)
    csv_train = pd.DataFrame(columns=name, data=test_list)
    csv_train.to_csv('dataset_train_DL_model/csv_test.csv', encoding='utf8', index=False)
    csv_train = pd.DataFrame(columns=name, data=dev_list)
    csv_train.to_csv('dataset_train_DL_model/csv_dev.csv', encoding='utf8', index=False)


"""训练深度学习模型部分"""
"""Train deep learning model"""
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model(test_iter, model, device):
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    accuracy = 0
    y_true = []
    y_pred = []
    total_test_num = len(test_iter.dataset)
    for batch in test_iter:
        feature = batch.text
        target = batch.label
        with torch.no_grad():
            feature = torch.t(feature)
        feature, target = feature.to(device), target.to(device)
        out = model(feature)
        loss = F.cross_entropy(out, target)
        total_loss += loss.item()
        accuracy += (torch.argmax(out, dim=1)==target).sum().item()
        y_true.extend(target.cpu().numpy())
        y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
    print('>>> Test loss:{}, Accuracy:{} \n'.format(total_loss/total_test_num, accuracy/total_test_num))
    score = accuracy_score(y_true, y_pred)
    print(score)
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_true, y_pred)
    print(confusion_matrix)
    from sklearn.metrics import classification_report
    target_names = ['negative', 'positive']
    print(classification_report(y_true, y_pred, target_names=target_names))

def train_model(train_iter, dev_iter, model, model_name, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    epochs = 10
    print('training...')
    highest_valid_acc = idx_acc_highest = 0
    lowest_valid_loss = 1
    idx_loss_lowest = 0
    lowest_valid_acc = 1
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        accuracy = 0
        total_train_num = len(train_iter.dataset)
        for batch in train_iter:
            feature = batch.text
            target = batch.label
            with torch.no_grad():
                feature = torch.t(feature)
            feature, target = feature.to(device), target.to(device)
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            accuracy += (torch.argmax(logit, dim=1)==target).sum().item()
        print('>>> Epoch_{}, Train loss is {}, Accuracy:{} \n'.format(epoch,loss.item()/total_train_num, accuracy/total_train_num))
        model.eval()
        total_loss = 0.0
        accuracy = 0
        total_valid_num = len(dev_iter.dataset)
        for batch in dev_iter:
            feature = batch.text  # (W,N) (N)
            target = batch.label
            with torch.no_grad():
                feature = torch.t(feature)
            feature, target = feature.to(device), target.to(device)
            out = model(feature)
            loss = F.cross_entropy(out, target)
            total_loss += loss.item()
            accuracy += (torch.argmax(out, dim=1)==target).sum().item()
        if total_loss / total_valid_num < lowest_valid_loss:
            lowest_valid_loss = total_loss / total_valid_num
            idx_loss_lowest = epoch
        if accuracy / total_valid_num > highest_valid_acc:
            highest_valid_acc = accuracy / total_valid_num
            idx_acc_highest = epoch
            saveModel(model, model_name)
        print('>>> Epoch_{}, Valid loss:{}, Accuracy:{} \n'.format(epoch, total_loss/total_valid_num, accuracy/total_valid_num))
    print("Lowest_loss_epoch: {} Lowest_idx: {}".format(idx_loss_lowest, lowest_valid_loss))
    print("Highest_acc_epoch: {} Highest_idx: {}".format(idx_acc_highest, highest_valid_acc))


def saveModel(model, name):
    torch.save(model, 'done_model/' + name + '_model.pkl')


def train_DL_classify_model(model_name):
    import DataSet
    from utils.model.TextCNN import TextCNN
    from utils.model.TextRCNN import TextRCNN
    from utils.model.TextRNN import TextRNN
    from utils.model.TextRNN_Attention import TextRNN_Attention
    from utils.model.Transformer import Transformer

    print(model_name)
    if model_name == 'CNN':
        model = TextCNN()
    elif model_name == 'RNN':
        model = TextRNN()
    elif model_name == 'RCNN':
        model = TextRCNN()
    elif model_name == 'RNN_Attention':
        model = TextRNN_Attention()
    elif model_name == 'Transformer':
        model = Transformer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iter, val_iter, test_iter = DataSet.getIter()
    train_model(train_iter, val_iter, model, model_name, device)
    # saveModel(model,model_name)
    test_model(test_iter, model, device)


"""使用训练好的深度学习模型分类部分"""
"""Use pretrained deep learning model to classify test set"""
def getModel(model_name):
    model = torch.load('done_model/' + model_name + '_model.pkl')
    return model

def predict_sentiment(model, sentence):
    model.eval()
    tokenized = nltk.word_tokenize(sentence)[:100]
    indexed = [getTEXT().vocab.stoi[t] for t in tokenized]
    if(len(indexed)<100):
        for i in range(len(indexed), 100):
            indexed.append(getTEXT().vocab.stoi['<pad>'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor = torch.LongTensor(indexed).to(device)
    tensor = torch.unsqueeze(tensor, 0)
    out = model(tensor)
    prediction = torch.argmax(out, dim=1).item()
    return prediction


def classify_test_dataset_use_trained_DL_model(model_name, wait_filtered_body, wait_filtered_title):
    model = getModel(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    filtered_body = []
    filtered_title = []
    reserved_id = []
    for idx, sent in tqdm(enumerate(wait_filtered_body), desc='Classify the test set'):
        if predict_sentiment(model, sent) == 1:
            filtered_body.append(sent)
            filtered_title.append(wait_filtered_title[idx])
            reserved_id.append(idx)
    return filtered_body, filtered_title, reserved_id

def DL_component_bleu_threshold(function_mode, model_name, train_body, train_title, train_pred, valid_body, valid_title, valid_pred, test_body, test_title):
    print("——————————————————————————————————————————————————————————DL Component begin——————————————————————————————————————————————————————————")
    if function_mode == 'train':
        print("DL module mode: Train new models.")
        # 训练数据初始化
        # Initial the training data
        train_plus_val_body = [body for body in train_body]
        train_plus_val_title = [title for title in train_title]
        train_plus_val_pred = [pred for pred in train_pred]
        for body in valid_body:
            train_plus_val_body.append(body)
        for title in valid_title:
            train_plus_val_title.append(title)
        for pred in valid_pred:
            train_plus_val_pred.append(pred)
        # 训练数据标签化
        # Label the training data
        body_0 = []
        title_0 = []
        body_1 = []
        title_1 = []
        smooth = SmoothingFunction()
        for idx, title in enumerate(train_plus_val_title):
            reference = []
            candiate = []
            for unit in title.split():
                for word in re.split("\!|\?|\,|\.|\'|\"|\{|\}|\`|\<|\>|\*|\@|\-|\#|\~|\$|\^|\+|\_", unit):  # ""
                    if not word == '' and word not in reference:
                        reference.append(word)
            for unit in train_plus_val_pred[idx].split():
                for word in re.split("\!|\?|\,|\.|\'|\"|\{|\}|\`|\<|\>|\*|\@|\-|\#|\~|\$|\^|\+|\_", unit):
                    if not word == '' and word not in candiate:
                        candiate.append(word)
            bleu = float(Decimal(sentence_bleu([reference], candiate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function
            =smooth.method1)).quantize(Decimal("0.1"), rounding="ROUND_HALF_UP"))
            if bleu < 0.1:
                body_0.append(train_plus_val_body[idx])
                title_0.append(train_plus_val_title[idx])
            else:
                body_1.append(train_plus_val_body[idx])
                title_1.append(train_plus_val_title[idx])
        print("Pos: {}, Neg：{}".format(len(body_1), len(body_0)))

        """准备train set对应深度学习模型所用数据集"""
        """Prepare the data set for deep learning model train"""
        fp_pos = open('dataset_train_pos_neg/iTAPE_train_positive.csv', 'w', encoding='utf-8-sig', newline='')
        writer_pos = csv.writer(fp_pos)
        fp_neg = open('dataset_train_pos_neg/iTAPE_train_negative.csv', 'w', encoding='utf-8-sig', newline='')
        writer_neg = csv.writer(fp_neg)
        for idx in range(len(body_1)):
            sss = [str(idx) + '\t' + "1" + '\t' + body_1[idx]]
            writer_pos.writerow(sss)
        for idx in range(len(body_0)):
            sss = [str(idx) + '\t' + "0" + '\t' + body_0[idx]]
            writer_neg.writerow(sss)

        file_dir = "dataset_train_pos_neg/"
        gen_train_DL_model_dataset(file_dir)

        """使用训练集训练深度学习分类器"""
        """Train deep learning classifier use training set"""
        train_DL_classify_model(model_name)
    else:
        print("DL module mode: Use pretrained Models.")
        print(model_name)

    """使用验证集上训练出的最好模型去筛选测试集"""
    """Use the best model trained on the valid set to classify the test set"""
    filtered_body, filtered_title, DL_Component_reserved_id = classify_test_dataset_use_trained_DL_model(
        model_name, test_body, test_title)
    return DL_Component_reserved_id
