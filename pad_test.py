from nltk.corpus import stopwords
from wordcloud import WordCloud
from os import path
import matplotlib.pyplot as plt

d = path.dirname(__file__)

stop_w = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
          'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their'
    , 'theirs', 'themselves', 'this', 'that', 'these', 'those', 'am', 'are', 'was'
    , 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'did', 'doing', 'a', 'an', 'the', 'and',
          'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between'
    , 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
          'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any',
          'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'nor', 'not', 'only', 'own', 'same', 'so',
          'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
          'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn',
          'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn','ye','doe']

removed_stopword = ['is','no','when', 'where', 'why', 'how','what', 'which', 'who', 'whom','do', 'does']

text = open(path.join(d,"Cap_word_QA_only_hard.txt")).read()
print(text)
wordcloud = WordCloud(stopwords=stop_w,background_color='white').generate(text)

plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()

# print(str(stopwords.words('english')))