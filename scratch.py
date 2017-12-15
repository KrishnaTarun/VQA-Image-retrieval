from nltk import word_tokenize
import json
import os
from collections import defaultdict, Counter
from functools import reduce
import argparse
import h5py
import glob
import numpy as np
import torch
from nltk.corpus import stopwords
from wordcloud import WordCloud
from os import path
import random
import matplotlib.pyplot as plt

d = path.dirname(__file__)

stop_w = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
          'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their'
    , 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was'
    , 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
    'and','but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
          'on',
          'off',
          'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
          'any',
          'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'nor', 'not', 'only', 'own', 'same', 'so',
          'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
          've',
          'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
          'needn',
          'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'ye']

question_words = ['what', 'where', 'when', "who", 'how', 'why', 'did', 'do', 'does', 'have', 'has', 'am', 'is', 'are',
                  'can', 'could', 'may', 'would', 'will', 'should', "didn't", "doesn't", "haven't", "isn't", "aren't",
                  "can't",
                  "couldn't", "wouldn't", "won't", "shouldn't", "which", "what's", "any", "anyone", "anything",
                  "anybody"]

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
count_q = Counter()

torch.manual_seed(1)
random.seed(1)
unique_dialogues = []
count1_w_a = Counter()
one_answer_d = []
aword_in_q = Counter()
nwords_per_question = Counter()
nwords_per_answer = Counter()
nwords_per_dialogue = Counter()
nwords_per_dialogue_set = Counter()
nwords_per_caption = Counter()
one_qna = []
one_qna_count = Counter()
one_q_with_a = []
one_qwith_count = Counter()
one_ainq = []
one_ainq_count = Counter()
one_w_q = []
one_w_q_count = Counter()
two_w_q = []
two_w_q_count = Counter()
three_w_q = []
three_w_q_count = Counter()
cap_in_q = []
cap_in_q_count = Counter()
cap_in_a = []
cap_in_a_count = Counter()
cap_in_qa = []
cap_in_qa_count = Counter()


# CUDA = torch.cuda.is_available()
# print("CUDA: %s" % CUDA)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dialog', type=int, help='consider only Q&A', default=0)
    parser.add_argument(
        '--caption', type=int, help='consider caption only', default=0)
    parser.add_argument(
        '--combine', type=int, help='combine both dialog and caption', default=1)
    parser.add_argument(
        '--path_folder', type=str, help='path of folder containing data', default="data/VQA_IR_data")
    parser.add_argument(
        '--type', type=str, help='Easy or Hard', default="Easy")
    parser.add_argument(
        '--img_feat', help='folder to image features', default="data/img_feat")

    # Array for all arguments passed to script
    args = parser.parse_args()
    # print(args)
    return args


def read_dataset(process):
    # data file path
    fld_path = os.path.join(args.path_folder, args.type)
    filename = 'IR_' + process + '_' + args.type.lower()

    print(fld_path, filename)
    # data
    with open(os.path.join(fld_path, filename + '.json')) as json_data:
        data = json.load(json_data)

    for key, val in data.items():
        word_d, word_c, img_list, target_ind, img_id = val['dialog'], val['caption'], val['img_list'], val['target'], \
                                                       val['target_img_id']
        list_q = []
        list_ans = []
        if len(img_list) == 10:
            for i, sen in enumerate(word_d):
                #     sen = ps.preprocess_sen(sen[0])
                #     stack_d += sen
                # word_c = ps.preprocess_sen(word_c)
                # print(i,sen)
                # print(type(sen))
                temp = sen[0].split('?')
                # print(temp)
                list_q.append(temp[0])
                list_ans.append(temp[1])
            # print(list_q)
            # print(list_ans)
            # print(word_c)
            yield [list_q, list_ans, word_c]


def process_qna(train_data):
    n_dialogue = 0

    for example in train_data:
        caption_token = word_tokenize(example[2])
        nwords_per_caption[len(caption_token)] += 1

        for dialogues_q in example[0]:
            n_dialogue += 1
            # print("dialog:",dialogues_q)
            qword_token = dialogues_q.split()
            nwords_per_question[len(qword_token)] += 1
            # print(qword_token)
            if qword_token[0] in question_words:
                # print(qword_token[0])
                count_q[qword_token[0]] += 1
            else:
                unique_dialogues.append(dialogues_q)
                # question word inside dialogue
                for i in question_words:
                    if i in qword_token:
                        count_q[i] += 1
                        break
            for word in caption_token:
                if word in qword_token and word not in stopwords.words('english'):
                    cap_in_q.append(word)
                    cap_in_q_count[word] += 1

        for dialogues_a in example[1]:
            # print(dialogues_a)
            aword_token = dialogues_a.split()
            nwords_per_answer[len(aword_token)] += 1

            if len(aword_token) == 1:
                count1_w_a[aword_token[0]] += 1

            for word in caption_token:
                if word in aword_token and word not in stopwords.words('english'):
                    cap_in_a.append(word)
                    cap_in_a_count[word] += 1

        n_dia_set = 0
        temp_len = 0
        for i, dialogues_q in enumerate(example[0]):
            qword_token = dialogues_q.split()
            aword_token = example[1][i].split()
            nwords_per_dialogue[len(qword_token) + len(aword_token)] += 1
            n_dia_set += 1
            temp_len += len(qword_token) + len(aword_token)
            if n_dia_set == 10:
                nwords_per_dialogue_set[temp_len] += 1
                temp_len = 0
                n_dia_set = 0


            for word in caption_token:
                if word in aword_token and word in qword_token and word not in stopwords.words('english'):
                    cap_in_qa.append(word)
                    cap_in_qa_count[word] += 1

            if len(qword_token) == 1:
                one_w_q.append((qword_token[0], example[1][i]))
                one_w_q_count[qword_token[0] + '/' + example[1][i]] += 1
            elif len(qword_token) == 2:
                two_w_q.append((dialogues_q, example[1][i]))
                two_w_q_count[dialogues_q + '/' + example[1][i]] += 1
            elif len(qword_token) == 3:
                three_w_q.append((dialogues_q, example[1][i]))
                three_w_q_count[dialogues_q + '/' + example[1][i]] += 1

            if len(aword_token) == 1:
                # print("dialog:", dialogues_q)
                temp = []
                temp.append(dialogues_q)
                # print(aword_token)
                temp.append(aword_token[0])
                one_answer_d.append(temp)
                one_qna.append((qword_token[0], aword_token[0]))
                one_qna_count[qword_token[0]] += 1
                one_qwith_count[qword_token[0] + "/" + aword_token[0]] += 1
                one_q_with_a.append(qword_token[0] + "/" + aword_token[0])

                if aword_token[0] in qword_token:
                    # print(aword_token[0],qword_token)
                    aword_in_q[aword_token[0]] += 1
                    one_ainq.append((qword_token[0], aword_token[0]))
                    one_ainq_count[qword_token[0]] += 1

    print("***\nTotal number of Dialogue:", n_dialogue)


args = get_args()

# Loading img_features
path_to_h5_file = glob.glob(args.img_feat + "/*.h5")[0]
img_features = np.asarray(h5py.File(path_to_h5_file, 'r')['img_features'])

path_to_json_file = glob.glob(args.img_feat + "/*.json")[0]
with open(path_to_json_file, 'r') as f:
    visual_feat_mapping = json.load(f)['IR_imgid2id']
# -------------------------------------------------------


train_data = list(read_dataset('train'))


# f = open("Train_data_hard.txt","w+")
# f.write(str(train_data))
# f.close()

def bar_plot(x, y,x_label,y_label):
    n = sum(y)  # maximum value of list y
    x = [int(i) for i in x]
    x_axis = np.array(x)
    y_axis = [i / n for i in y]

    plt.scatter(x_axis,y_axis,alpha=0.4)
    plt.bar(x_axis, y_axis, facecolor='#9999ff', edgecolor='white')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def word_cloud(text):
    wordcloud = WordCloud(stopwords=stop_w, background_color='white').generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


process_qna(train_data)

# most_common_nwpq = nwords_per_question.most_common(20)
# # print(type(most_common),most_common)
# most_common_x = [i[0] for i in most_common_nwpq]
# most_common_y = [i[1] for i in most_common_nwpq]
# bar_plot(most_common_x,most_common_y,"Number of words per question","ratio of question in dialogue")
# # bar_plot(nwords_per_question.keys(),nwords_per_question.values())

# most_common_nwpq = nwords_per_answer.most_common(20)
# # print(type(most_common),most_common)
# most_common_x = [i[0] for i in most_common_nwpq]
# most_common_y = [i[1] for i in most_common_nwpq]
# bar_plot(most_common_x,most_common_y,"Number of words per answer","ratio of answer in dialogue")
#
# most_common_nwpq = nwords_per_dialogue.most_common(200)
# # print(type(most_common),most_common)
# most_common_x = [i[0] for i in most_common_nwpq]
# most_common_y = [i[1] for i in most_common_nwpq]
# bar_plot(most_common_x,most_common_y,"Number of words per dialogue","ratio of dialogue in dataset")

most_common_nwpq = nwords_per_dialogue_set.most_common(200)
# print(type(most_common),most_common)
most_common_x = [i[0] for i in most_common_nwpq]
most_common_y = [i[1] for i in most_common_nwpq]
bar_plot(most_common_x,most_common_y,"Number of words per dialogue set","ratio of dialogue in dataset")

most_common_nwpq = nwords_per_caption.most_common(50)
print(type(most_common_nwpq),most_common_nwpq)
most_common_x = [i[0] for i in most_common_nwpq]
most_common_y = [i[1] for i in most_common_nwpq]
bar_plot(most_common_x,most_common_y,"Number of words per caption","ratio of words in caption")



# f = open("Unique_dialogue_hard.txt","w+")
# f.write(str(unique_dialogues))
# f.close()
#
#
#
# f = open("One_word_answer_question_type_only_hard.txt","w+")
# f.write(str(reduce(lambda x,y: x + " " + y,[i[0] for i in one_qna])))
# f.close()

# f = open("One_word_answer_with_question_type_only.txt","w+")
# f.write(str(reduce(lambda x,y: x + " " + y,[i for i in one_q_with_a])))
# f.close()


# f = open("One_word_answer_only_hard.txt","w+")
# f.write(str(reduce(lambda x,y: x + " " + y,[i[1] for i in one_answer_d])))
# f.close()
# # #
# f = open("One_word_fromQ_only_hard.txt","w+")
# f.write(str(reduce(lambda x,y: x + " " + y,[i[1] for i in one_ainq])))
# f.close()
# # #
# f = open("One_word_Q_only_hard.txt","w+")
# f.write(str(reduce(lambda x,y: x + " " + y,[i[0] for i in one_w_q])))
# f.close()
#
# f = open("Cap_word_QA_only_hard.txt","w+")
# f.write(str(reduce(lambda x,y: x + " " + y,[i for i in cap_in_qa])))
# f.close()


print("***\nMost common question word", count_q.most_common(30))
print("***\nDialogue that has question words:", sum(count_q.values()))
# print("***\nQuestion type of one word answer",set(one_qna))
print("***\nQuestion type of one word answer counter", one_qna_count)
print("***\nQuestion type of one word with answer counter", one_qwith_count)
print("***\nQuestion type of answer word from question", set(one_ainq))
print("***\nQuestion type of answer word from question counter", one_ainq_count)
print("***\nFrequency of number of words per question:", nwords_per_question)
# # print("One word question ", one_w_q)
print("***\nOne word question count", one_w_q_count)
print("***\nOne word question Total count", sum(one_w_q_count.values()))
# # print("Two word question ", two_w_q)
print("***\nTwo word question count", two_w_q_count)
print("***\nTwo word question Total count", sum(two_w_q_count.values()))
# # print("Three word question ", three_w_q)
print("***\nThree word question count", three_w_q_count.most_common(20))
print("***\nThree word question Total count", sum(three_w_q_count.values()))
print("***\nTotal number of dialogue where answer is a single word from the question.", sum(aword_in_q.values()))
#
#
print("***\nOne word answers", sorted(count1_w_a.items(), reverse=True, key=lambda _: _[1]))
print("***\nOne word answers total:", sum(count1_w_a.values()))
print("***\nFrequency of number of words per answer:", nwords_per_answer)
#
print("***\nFrequency of number of words per dialogue:", nwords_per_dialogue)

# print("***\nCaption word in question:", cap_in_q)
print("***\nCaption word in question count:", cap_in_q_count)
print("***\nCaption word in question total:", sum(cap_in_q_count.values()))

# print("***\nCaption word in answer:", cap_in_a)
print("***\nCaption word in answer count:", cap_in_a_count)
print("***\nCaption word in answer total:", sum(cap_in_a_count.values()))

# print("***\nCaption word in answer:", cap_in_a)
print("***\nCaption word in both question and answer count:", cap_in_qa_count)
print("***\nCaption word in both question and answer total:", sum(cap_in_qa_count.values()))
