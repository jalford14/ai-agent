# RSS Feed Agent

Designed to be a simple ranker for multiple RSS feeds. Input a topic you want to
learn about and have the agent return some related articles.

## Installation
```
pip install torch transformers feedparser requests scikit-learn numpy
```

## Usage
The agents usage is very simple. Just add a topic you want to learn more about 
and it will return a list of related topics along with a URL you can click on.
```
python3 agent.py "distributed systems"

3. How Skello uses AWS DMS to synchronize data from a monolithic application to microservices https://aws
.amazon.com/blogs/database/how-skello-uses-aws-dms-to-synchronize-data-from-a-monolithic-application-to-m
icroservices/
2. Diving in to Systems Design https://jda.bearblog.dev/diving-in-to-systems-design/
1. Notes on Theory of Distributed Systems http://www.freetechbooks.com/notes-on-theory-of-distributed-sys
tems-t1371.html
```
