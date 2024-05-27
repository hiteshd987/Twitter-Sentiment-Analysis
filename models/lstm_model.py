import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.downloader import load

def calc_lstm(texts, labels):

    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    # Convert text data to sequences of integers
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad sequences to ensure uniform length
    max_words = 100  # Adjust as needed based on your data
    padded_sequences = sequence.pad_sequences(sequences, maxlen=max_words)

    # Convert labels to numeric format
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

    # Load GloVe embeddings using gensim downloader
    glove_model = load("glove-wiki-gigaword-100")

    # Defining LSTM model
    vocab_size = len(tokenizer.word_index) + 1
    embd_len = glove_model.vector_size  # Embedding dimension for GloVe embeddings

    # Create embedding matrix
    embedding_matrix = np.zeros((vocab_size, embd_len))
    for word, i in tokenizer.word_index.items():
        if word in glove_model:
            embedding_matrix[i] = glove_model[word]

    # Create embedding layer without weights
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embd_len)

    embedding_layer.build((None, ))

    # Set the weights of the embedding layer
    embedding_layer.set_weights([embedding_matrix])

    embedding_layer.trainable = False

    # Define the LSTM model
    lstm_model = Sequential(name="LSTM_Model")
    lstm_model.add(embedding_layer)
    lstm_model.add(LSTM(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    lstm_model.add(Dense(1, activation='sigmoid'))


    # Compile the model
    lstm_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Training the model with early stopping
    history = lstm_model.fit(x_train, y_train, batch_size=64, epochs=50, verbose=2, validation_split=0.1, callbacks=[early_stopping])

    # Displaying the model accuracy on test data
    print()
    print("LSTM model ---> ", lstm_model.evaluate(x_test, y_test, verbose=0))

    y_pred = lstm_model.predict(x_test)
    y_pred_LSTM = (y_pred > 0.5).astype(int)
    
    return y_pred_LSTM, y_test


    
