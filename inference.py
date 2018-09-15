from keras.models import Model
from keras.layers import Input, Dense, Reshape, dot
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence

import urllib
import collections
import os
import zipfile

import numpy as np
import tensorflow as tf



# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with open(filename) as data:
        data = data.read().split(" ")
        return data


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def collect_data(vocabulary_size=10000):
    filename = "./parsed_lines.txt"
    vocabulary = read_data(filename)
    print(vocabulary[:7])
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocabulary_size)
    del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary

vocab_size = 10000
data, count, dictionary, reverse_dictionary = collect_data(vocabulary_size=vocab_size)
print(data[:7])

window_size = 3
vector_dim = 300

valid_size = 8     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

sampling_table = sequence.make_sampling_table(vocab_size)
couples, labels = skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

print(couples[:10], labels[:10])

# create some input variables
input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

# setup a cosine similarity operation which will be output in a secondary model
similarity = dot([target, context], axes=0, normalize=True)

dot_product = Reshape((1,))(similarity)
# add the sigmoid output layer
output = Dense(1, activation='sigmoid')(dot_product)
# create the primary training model
model = Model(input=[input_target, input_context], output=output)
model.compile(loss='binary_crossentropy', optimizer='adam')


arr_1 = np.zeros((16,))
arr_2 = np.zeros((16,))
arr_3 = np.zeros((16,))
for cnt in range(epochs):
    for i in range(16):
        idx = np.random.randint(0, len(labels)-1)
        arr_1[i,] = word_target[idx]
        arr_2[i,] = word_context[idx]
        arr_3[i,] = labels[idx]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if cnt % 100 == 0:
        print("Iteration {}, loss={}".format(cnt, loss))
    if cnt % 50000 == 0 and not cnt == 0:
        sim_cb.run_sim()

model.load_weights('./weights.h5')

for i in range(5):
    model.layers.pop()

from scipy.spatial.distance import cosine

def infer_sentence_similarity(reference_vector, sentence_words):
    """
    reference_vector: the reference sentence vector to compare this sentence to
    sentence_words: the list of words in the sentence
    """
    comp_vector = np.zeros(shape=(vector_dim,))
    for word in sentence_words:
        comp_vector = np.add(comp_vector, compute_word_vector(word))
    comp_vector /= len(sentence_words)
    return cosine(comp_vector, reference_vector)


def infer_word_similarity(reference_vector, given_word):
    """
    reference_vector: the vector to compare to
    given_word: the word to compute cosine similarity to
    word_contexts: the words within the 3-gram window
    """
    # gets the cosine similarity of reference vector to the computed word vector for the given word
    # where the computed word vector is the average word vector over the 3-gram window as computed
    # by the model
    comp_vector = compute_word_vector(given_word)
    return cosine(comp_vector, reference_vector)


def compute_word_vector(given_word):
    """
    Computes the word vector for the given word.
    given_word: the word to compute a word vector for, in string form
    """
    x_word = np.zeros(shape=(1,))
    x_context = np.zeros(shape=(1,))

    x_word[0] = dictionary[given_word]
    x_context[0] = dictionary[given_word]
    # predict
    vectors = model.predict([x_word, x_context])    
    return avg_vector[0]


from flask import Flask, request
app = Flask(__name__)

"""
Classifies a given sentence given its similarity to pre defined templates.
"""
@app.route("/classify_sentence", methods=['POST'])
def classify_sentence():
    req_data = request.get_json()
    sentence = req_data['sentence']


    return "Hello World!"

"""
Fills speech defined spaces/blanks in a given template sentence.
"""
@app.route("/fill_template_blanks", methods=['POST'])
def fill_template_blanks():
    req_data = request.get_json()
    # template sentence to match against
    template = req_data['template']
    # speech sentence to use for matching
    speech_sentence = req_data['speech_sentence']
    

if __name__ == '__main__':
    app.run(debug=True, port=5000) #run app in debug mode on port 5000