# Twitter Semisupervised Sentiment Analysis

> This repository is made for the NLP course project - Apr 2018.

**Dependencies:**
- [JSON](https://docs.python.org/3/library/json.html)
- [AFINN](https://pypi.org/project/afinn/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [NLTK](https://www.nltk.org/)
- [Matplotlib](https://matplotlib.org/)

**Dataset:**
- [AFINN-111](http://www2.imm.dtu.dk/pubdb/pubs/6010-full.html)
- [Tweets of a twitter account](/tweets.json)

---

A semisupervised sentiment analysis for tweets of a twitter account over time.

Steps:
- Collect all Tweets of an account in a `json` file with the following format:
```
{
  "source": "Twitter for iPhone",
  "text": "Some text",
  "created_at": "Sun Jul 08 21:58:52 +0000 2018",
  "retweet_count": 64399,
  "favorite_count": 183994,
  "is_retweet": false,
  "id_str": "1016079192604139520"
} 
```
- Use `NLTK` for Lemmatization and Tokenization.
- Based on AFINN dataset, each word is given a score, from +5 (very positive) to -5 (very negative).
- Use `scikit-learn` to calculate precision, recall, and f1-score.
- Use `Matplotlib` to plot a histogram of the sentiment analysis over time.

![sentiment analysis histogram](/histogram.png)
