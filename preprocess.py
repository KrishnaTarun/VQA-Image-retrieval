from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

def preprocess_sen(sentence):
    #print(sentence)
    sen = word_tokenize(sentence)
    sen = [word.lower() for word in sen if word not in stopwords.words('english')]

    return sen

def filter_words(w2i,count):
    w2i_temp = defaultdict(lambda: len(w2i_temp))
    UNK = w2i_temp["<unk>"]

    number = 0

    for i in w2i:
        if count[i] > 4:
            # print("FILTERED",i,count[i],w2i[i])
            w2i_temp[i]
        else:
            number += 1

    # print(number)
    return w2i_temp

