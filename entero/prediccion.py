# Clean and preprocess your own input premise and hypothesis
my_premise_cleaned = limpiarTexto("A man is playing golf")
my_hypothesis_cleaned = limpiarTexto("A man is sitting down")

print("Cleaned premise:", my_premise_cleaned)
print("Cleaned hypothesis:", my_hypothesis_cleaned)

# Remove stopwords from your own input premise and hypothesis
my_premise_cleaned = remove_stopwords(my_premise_cleaned)
my_hypothesis_cleaned = remove_stopwords(my_hypothesis_cleaned)

print("Premise without stopwords:", my_premise_cleaned)
print("Hypothesis without stopwords:", my_hypothesis_cleaned)

# Tokenize your own input premise and hypothesis
my_premise_tokenized = tokenizador.texts_to_sequences([my_premise_cleaned])
my_hypothesis_tokenized = tokenizador.texts_to_sequences([my_hypothesis_cleaned])

print("Tokenized premise:", my_premise_tokenized)
print("Tokenized hypothesis:", my_hypothesis_tokenized)

# Pad the tokenized sequences of your own input premise and hypothesis
my_premise_padded = pad_sequences(my_premise_tokenized, maxlen=45)
my_hypothesis_padded = pad_sequences(my_hypothesis_tokenized, maxlen=45)

print("Padded premise:", my_premise_padded)
print("Padded hypothesis:", my_hypothesis_padded)

#Embedding


# Make predictions on your own input data
predictions = model.predict([my_premise_padded, my_hypothesis_padded])

# Decode the predictions to get the corresponding labels
predicted_labels = np.argmax(predictions, axis=1)

# Map the predicted label to its corresponding category
label_mapping = {1: "contradiction", 2: "entailment"}
predicted_label = label_mapping[predicted_labels[0]]

print("Predicted Label:", predicted_label)