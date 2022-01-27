import json
from afinn import Afinn
import random
from sklearn.metrics import precision_recall_fscore_support
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

not_list = ['not', "isn't", "doesn't", "didn't", "don't", "wouldn't", "couldn't", "shouldn't"]
year = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
year_history = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]

for y in year_history:
    for i in range(12):
        y[year[i]] = [0, 1]
    y['total'] = [0, 1]

json_file = 'tweets.json'

json_data = open(json_file, encoding='utf-8')
data = json.load(json_data)
json_data.close()

ps = PorterStemmer()

for i in range(len(data)):

    # text
    if '\n' in data[i]['text']:
        data[i]['text'] = data[i]['text'].replace('\n', ' ')

    # date
    d = data[i]['created_at'].split(' ')
    data[i]['created_at'] = (d[1], d[-1])


def AFINN():
    with open("AFINN-111.txt", 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        pn = {}
        for line in lines:
            w = line.split('\t')
            pn[ps.stem(w[0])] = w[1][:-1]
    return pn


def AFINN_sentiment(tweet, polarity):
    tweet = word_tokenize(tweet)
    sentiment = 0

    for i in range(len(tweet)):
        tweet[i] = tweet[i].lower()

    for i in range(len(tweet)):
        if tweet[i] in polarity:
            is_not = check_not(tweet, i)
            if is_not:
                sentiment -= int(polarity[tweet[i]])
            else:
                sentiment += int(polarity[tweet[i]])
            if int(polarity[tweet[i]]) <= 0:
                sentiment += 1.5 * int(polarity[tweet[i]])
    return sentiment


def check_not(text, position):
    for i in range(position - 3, position):
        if text[i] in not_list:
            return True
    return False


def get_sample(src_list, sample_list):
    for i in range(100):
        r = random.randint(1, len(src_list))
        sample_list.append(src_list.pop(r))
    return sample_list


def final_normalize(lst):
    lst.reverse()
    for i in range(4):
        lst.pop()
    lst.reverse()
    for i in range(5):
        lst.pop()


polarity_dict = AFINN()

sample = []
sample = get_sample(data, sample)

total_history = []

for d in data:
    sent = AFINN_sentiment(d['text'], polarity_dict)
    total_history.append((d['created_at'], sent))

    year_history[int(d['created_at'][1]) - 2009][d['created_at'][0]][0] += sent
    year_history[int(d['created_at'][1]) - 2009][d['created_at'][0]][1] += 1
    year_history[int(d['created_at'][1]) - 2009]['total'][0] += sent
    year_history[int(d['created_at'][1]) - 2009]['total'][1] += 1

final = []
for y in year_history:
    for m in year:
        final.append(y[m][0] / y[m][1])

final_normalize(final)

classifier = Afinn(language='en')
real = []
pred = []
for item in sample:
    if classifier.score(item['text']) >= 0:
        real.append('pos')
    else:
        real.append('neg')
    if AFINN_sentiment(item['text'], polarity_dict) >= 0:
        pred.append('pos')
    else:
        pred.append('neg')

accuracy = precision_recall_fscore_support(real, pred)

print('Precision:\t', accuracy[0])
print('Recall:\t\t', accuracy[1])
print('F1-Score:\t', accuracy[2])


color = []
for i in final:
    if i >= 0:
        color.append("green")
    else:
        color.append("red")

plt.bar([0], [0], color="red", label="Negative")

plt.bar([i for i in range(111)], final, color=color, label="Positive")
plt.xticks([i for i in range(9, 111, 12)], ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018'])
plt.title("Sentiment of tweets")
plt.grid()
plt.xlabel("Tweets per month")
plt.ylabel("Avg. sentiment")
plt.legend()
plt.show()
plt.savefig('./hist.png')
