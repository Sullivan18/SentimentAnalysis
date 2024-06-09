from ntscraper import Nitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import unicodedata
import json
import re
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import numpy as np
from autocorrect import Speller
import pandas as pd

# Inicializar o scraper
nitter = Nitter()

# Definir o nome de usuário do perfil do X
username = "depressaorelato"

# Buscar os últimos 100 tweets do usuário
tweets_data = nitter.get_tweets(username, mode='user', number=100)

# Inicializar o classificador de sentimentos
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model_name)

# Função para normalizar, corrigir e truncar texto para 512 caracteres
def preprocess(text, max_length=512):
    text = re.sub(r"http\S+", "", text)  # Remover URLs
    text = re.sub(r"@\w+", "", text)  # Remover menções
    text = re.sub(r"#\w+", "", text)  # Remover hashtags
    text = Speller()(text)  # Corrigir erros ortográficos
    normalized_text = unicodedata.normalize('NFKC', text)
    return normalized_text[:max_length]

# Função para análise temporal
def analyze_sentiments(tweets):
    sentiments_over_time = defaultdict(list)
    for tweet in tweets:
        if 'text' in tweet and 'date' in tweet:
            normalized_text = preprocess(tweet['text'])
            if len(normalized_text) > 0:
                sentiment_result = classifier(normalized_text)[0]
                tweet['sentiment'] = sentiment_result
                tweet_date = datetime.strptime(tweet['date'], '%b %d, %Y · %I:%M %p %Z')
                sentiments_over_time[tweet_date].append(sentiment_result['label'])
    return sentiments_over_time

# Analisar sentimentos dos tweets
if 'tweets' in tweets_data:
    tweets = tweets_data['tweets']
    sentiments_over_time = analyze_sentiments(tweets)
    
    # Exportar os dados em formato JSON com os sentimentos
    with open("tweets_sentiments.json", "w") as file:
        json.dump(tweets_data, file, indent=4)

    print("Análise de sentimentos concluída e salva em tweets_sentiments.json")

    # Salvar resultados em CSV
    df = pd.DataFrame(tweets)
    df.to_csv('tweets_sentiments.csv', index=False)

    # Visualização dos resultados
    dates = sorted(sentiments_over_time.keys())
    positive_counts = [sentiments_over_time[date].count('Positive') for date in dates]
    neutral_counts = [sentiments_over_time[date].count('Neutral') for date in dates]
    negative_counts = [sentiments_over_time[date].count('Negative') for date in dates]

    plt.figure(figsize=(10, 5))
    plt.plot(dates, positive_counts, label='Positive', color='green')
    plt.plot(dates, neutral_counts, label='Neutral', color='blue')
    plt.plot(dates, negative_counts, label='Negative', color='red')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.title('Sentiment Analysis Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Formato de dados inesperado, chave 'tweets' não encontrada.")
