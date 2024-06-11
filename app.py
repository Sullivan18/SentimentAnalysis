from transformers import pipeline
import unicodedata
import json
from ntscraper import Nitter

# Inicializar o scraper
nitter = Nitter()

# Definir o nome de usuário do perfil do X
username = "depressaorelato"

# Buscar os últimos 100 tweets do usuário
tweets_data = nitter.get_tweets(username, mode='user', number=100)
print("Tweets Data:", tweets_data)  # Adicionar para depuração

# Inicializar o classificador de sentimentos
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
classifier = pipeline('sentiment-analysis', model=model_name)

# Função para normalizar e truncar texto para 512 caracteres
def normalize_and_truncate_text(text, max_length=512):
    normalized_text = unicodedata.normalize('NFKC', text)
    return normalized_text[:max_length]

# Dicionário de palavras e pesos para categorias específicas
word_weights = {
    "felicidade": {"alegria": 0.8, "sorriso": 0.7, "feliz": 0.9},
    "tristeza": {"depressão": 0.9, "triste": 0.8, "chorar": 0.7},
    # Adicione mais categorias e palavras conforme necessário
}

# Função para classificar texto baseado no dicionário de palavras e pesos
def classify_based_on_words(text, word_weights):
    scores = {category: 0 for category in word_weights}
    words = text.split()
    for word in words:
        for category, weights in word_weights.items():
            if word in weights:
                scores[category] += weights[word]
    return scores  # Retorna os escores de todas as categorias

# Função para combinar resultados do BERT e dicionário de palavras
def combine_classifications(bert_result, word_scores, weight_bert=0.5, weight_words=0.5):
    combined_scores = {}
    # Combinar os escores das categorias
    for category in word_scores:
        combined_scores[category] = word_scores.get(category, 0) * weight_words

    # Ajustar com base no resultado do BERT
    if bert_result['label'] == 'positive':
        combined_scores['felicidade'] = combined_scores.get('felicidade', 0) + bert_result['score'] * weight_bert
    elif bert_result['label'] == 'negative':
        combined_scores['tristeza'] = combined_scores.get('tristeza', 0) + bert_result['score'] * weight_bert
    # Adicione mais ajustes conforme necessário para outras categorias

    return max(combined_scores, key=combined_scores.get)  # Retorna a categoria com o maior escore

# Verificar se o resultado contém a chave 'tweets'
if 'tweets' in tweets_data:
    tweets = tweets_data['tweets']
    print("Tweets:", tweets)  # Adicionar para depuração

    # Analisar sentimentos dos tweets e adicionar classificação combinada
    for tweet in tweets:
        if 'text' in tweet:
            normalized_text = normalize_and_truncate_text(tweet['text'])
            bert_sentiment_result = classifier(normalized_text)[0]
            word_scores = classify_based_on_words(normalized_text, word_weights)
            combined_category = combine_classifications(bert_sentiment_result, word_scores)
            tweet['sentiment'] = bert_sentiment_result
            tweet['combined_category'] = combined_category

    # Exportar os dados em formato JSON com os sentimentos e categorias combinadas
    with open("tweets_combined_sentiments.json", "w") as file:
        json.dump(tweets_data, file, indent=4)

    print("Análise de sentimentos e classificação combinada concluída e salva em tweets_combined_sentiments.json")
else:
    print("Formato de dados inesperado, chave 'tweets' não encontrada.")
