from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import numpy as np
import feedparser
import requests
import torch

def fetch_articles(rss_feeds):
    articles = []
    for feed in rss_feeds:
        parsed = feedparser.parse(feed)
        articles.extend(parsed.entries)
    return articles

def get_embeddings(text, model, tokenizer):
    with torch.no_grad():  # Disable gradient tracking
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def rank_articles(articles, topic, model, tokenizer):
    topic_embedding = get_embeddings(topic, model, tokenizer)
    
    ranked = []
    for article in articles:
        text = f"{article.title} {article.description}"
        article_embedding = get_embeddings(text, model, tokenizer)
        similarity = cosine_similarity(topic_embedding, article_embedding)[0][0]
        ranked.append(({'title': article.title, 'link': article.link}, similarity))
    
    return sorted(ranked, key=lambda x: x[1], reverse=True)

# Usage
rss_feeds = [
              "https://news.ycombinator.com/rss",
              "http://radar.oreilly.com/index.rdf",
              "https://hackernoon.com/feed",
              "blog.google/rss",
              "https://jda.bearblog.dev/feed/?type=rss",
              "https://rss.arxiv.org/rss/cs",
              "redislabs.com/feed",
              "mongodb.com/blog/rss",
              "https://aws.amazon.com/blogs/database/feed/",
              "planet.mysql.com/rss20.xml",
              "blogs.oracle.com/database/rss",
              "sqlshack.com/feed",
              "http://www.reddit.com/r/cpp/.rss",
              "http://blog.knatten.org/feed/"
              "blog.rust-lang.org/feed.xml",
              "elixirstatus.com/rss"
            ]
topic = "mysql"

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

articles = fetch_articles(rss_feeds)
ranked_articles = rank_articles(articles, topic, model, tokenizer)
rank = 1
for article in ranked_articles:
    print(str(rank) + ". " + article[0]['title'] + " " + article[0]['link'])
    rank += 1

print("FINISHED")
