import transformers

def get_sentiment(text):
    finbert = transformers.AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    tokenizer = transformers.AutoTokenizer.from_pretrained("ProsusAI/finbert")

    encoded_inputs = tokenizer(text=text, return_tensors="pt")
    predictions = finbert(**encoded_inputs)
    print(predictions)
    return predictions[0][0]

if __name__ == "__main__":
    text = "The stock market is up today."
    sentiment = get_sentiment(text)
    print("")
