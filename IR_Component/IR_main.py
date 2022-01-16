import re
from numpy.ma import arange
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from tqdm import tqdm
from rouge import FilesRouge
from decimal import Decimal
from utils.tools import load_data, save_data

# TF mode = 1  /  TF-idf mode = 2
Word_bag_mode = 'TF-IDF'
# Strategy = 1/2/3
Strategy = 3
calculation_refs = 0
global_bleu = 0


class obj:
    def __init__(self, key=0, weight=0.0):
        self.key = key
        self.weight = weight


def IsCandidate(sentence1, sentence2):
    a = [x for x in sentence1 if x in sentence2]
    if len(a) < 1:
        return 0
    else:
        return 1


def similarity(train_body, test_body, n=5):  # train_body：valid, test_body：test
    counter = CountVectorizer()
    global Word_bag_mode
    if Word_bag_mode == 'TF':
        # TF mode
        train_matrix = counter.fit_transform(train_body)
        test_matrix = counter.transform(test_body)
    elif Word_bag_mode == 'TF-IDF':
        # TF-IDF mode
        transformer = TfidfTransformer()
        train_matrix = transformer.fit_transform(counter.fit_transform(train_body))
        test_matrix = transformer.transform(counter.transform(test_body))

    similarities = cosine_similarity(test_matrix, train_matrix)  # len(test_matrix) * len(train_matrix)
    test_body_score = []
    count = -1
    for idx, test_simi in enumerate(similarities):
        count += 1
        score = 0
        global Strategy  # 初始代码定义为3
        if Strategy == 1:
            str1 = train_body[test_simi.argsort()[-1]]
            str2 = test_body[idx]
            score = obj(count, cosine_similarity(counter.transform([str1]), counter.transform([str2]))[0][0])
        elif Strategy == 2:
            score_temp = 0
            for i in range(n):
                str1 = train_body[test_simi.argsort()[-(i+1)]]
                str2 = test_body[idx]
                score_temp += cosine_similarity(counter.transform([str1]), counter.transform([str2]))[0][0] / (i+1)
            score = obj(count, score_temp)
        elif Strategy == 3:
            ref = []
            str2 = test_body[idx]
            for i in range(n):
                str1 = train_body[test_simi.argsort()[-(i + 1)]]  # 相当于str1从train_body中选取和当前test_body最为相似的train_body数据，该for循环为从最相似到第五相似依次选取
                ref.append(str1.split())
            smooth = SmoothingFunction()
            score = sentence_bleu(ref, str2.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)  # 这个score本质上来说是cosine相似度排序后的前几句的bleu相似度
        test_body_score.append(score)
    return test_body_score


def IR_component(train_body, test_body, test_title, test_pred):
    print("——————————————————————————————————————————————————————————IR Component begin——————————————————————————————————————————————————————————")
    # 阈值初始化
    # Initial the threshold
    thresholds = []
    for threshold in arange(0, 1, 0.05):
        thresholds.append(float(Decimal(threshold).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")))
    # 计算相似度
    # Calculate similarities
    test_body_score = []
    n = 5000
    sim_cal_units = [test_body[i:i + n] for i in range(0, len(test_body), n)]
    for sim_cal_unit in tqdm(sim_cal_units, desc='Calculate similarity for IR component'):
        for each_sim in similarity(train_body, sim_cal_unit):
            test_body_score.append(each_sim)
    # 结果存放容器初始化
    # Initial the variables to storage result
    best_threshold = best_total_bleu = best_reserve_ratio = 0
    best_reserved_id = []
    # 选取最佳阈值
    # Choose best threshold
    for threshold in thresholds:
        IR_reserved_id = []
        for idx, score in enumerate(test_body_score):  # 遍历所有test数据集中数据与valid数据集最相似body的bleu-4相似度
            if score >= threshold:  # 相似度如果没有达到当前的阈值就舍弃
                IR_reserved_id.append(idx)
        IR_reserved_id_save_path = "./" + str(threshold) + "IR_reserved_id.txt"
        save_data(IR_reserved_id, IR_reserved_id_save_path)
        smooth = SmoothingFunction()
        references = []
        candiates = []
        for id in IR_reserved_id:
            reference = []
            candiate = []
            for unit in test_title[int(id)].split():
                for word in re.split("\!|\?|\,|\.|\'|\"|\{|\}|\`|\<|\>|\*|\@|\-|\#|\~|\$|\^|\+|\_", unit):  # ""
                    if not word == '' and word not in reference:
                        reference.append(word)
            for unit in test_pred[int(id)].split():
                for word in re.split("\!|\?|\,|\.|\'|\"|\{|\}|\`|\<|\>|\*|\@|\-|\#|\~|\$|\^|\+|\_", unit):
                    if not word == '' and word not in candiate:
                        candiate.append(word)
            references.append([reference])
            candiates.append(candiate)
        if len(IR_reserved_id) == 0:
            print("Notice: Begin from threshold {} IR reserved NONE, so break.".format(threshold))
            break
        total_bleu = corpus_bleu(references, candiates, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
        if total_bleu < 0.1:
            print("corpus_bleu on {} lower than 0.1 not qualified.".format(threshold))
            continue
        else:
            reserve_rate = len(IR_reserved_id) / len(test_title)
            print("threshold: {}  total_bleu: {} reserve_rate: {}%".format(threshold, total_bleu, float(Decimal(reserve_rate).quantize(Decimal("0.0001"), rounding="ROUND_HALF_UP")) * 100))
            if reserve_rate > best_reserve_ratio:
                best_threshold = threshold
                best_total_bleu = total_bleu
                best_reserve_ratio = reserve_rate
                best_reserved_id = IR_reserved_id
            elif reserve_rate == best_reserve_ratio:
                if total_bleu > best_total_bleu:
                    best_threshold = threshold
                    best_total_bleu = total_bleu
                    best_reserved_id = IR_reserved_id
    print("——————————————————————————————————————————————————————————IR Component result——————————————————————————————————————————————————————————")
    print("IR Component: best_threshold: {}, best_reserve_ratio: {}, best_bleu_total: {}".format(best_threshold, best_reserve_ratio,
                                                                                    best_total_bleu))
    return best_reserved_id
