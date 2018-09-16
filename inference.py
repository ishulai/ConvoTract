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
model.load_weights('./weights.h5')
model.compile(loss='binary_crossentropy', optimizer='adam')

new_model = Model(model.inputs, model.layers[-5].output)
print(new_model.summary())
new_model.set_weights(model.get_weights())
model = new_model
model._make_predict_function()

from scipy.spatial.distance import cosine

def infer_sentence_similarity(reference_vector, sentence_words):
    """
    reference_vector: the reference sentence vector to compare this sentence to
    sentence_words: the list of words in the sentence
    """
    comp_vector = compute_sentence_vector(sentence_words)
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


def get_idx(word):
    if word in dictionary:
        return dictionary[word]
    else:
        return dictionary['UNK']

def compute_word_vector(given_word):
    """
    Computes the word vector for the given word.
    given_word: the word to compute a word vector for, in string form
    """
    x_word = np.zeros(shape=(1,))
    x_context = np.zeros(shape=(1,))

    x_word[0] = get_idx(given_word)
    x_context[0] = get_idx(given_word)
    # predict
    vectors = model.predict([x_word, x_context])    
    return [vectors[0][i][0] for i in range(len(vectors[0]))]

def compute_sentence_vector(words):
    x_word = np.zeros(shape=(len(words),1))
    x_context = np.zeros(shape=(len(words),1))
    for i, word in enumerate(words):
        x_word[i][0] = get_idx(word)
        x_context[i][0] = get_idx(word)
    vectors = model.predict([x_word, x_context])
    vector = np.zeros(shape=(vector_dim,))
    for v in vectors:
        vector = np.add(v, vector)
    return [(vector / len(words))[i][0] for i in range(vector_dim)]

from flask import Flask, request, jsonify
from loaders import load_template, get_all_sentence_vectors, get_templates
app = Flask(__name__)

@app.route('/fill_out', methods=['POST'])
def fill_out():
    json = request.json
    text = json['text']
    print("text", text)
    text = text.split(".")

    # choose a template to use
    idxes = []
    freq_dict = {}
    for item in text:
        sentence_type, supertemplate = classify_sentence(item)
        print(sentence_type, supertemplate)
        if supertemplate in freq_dict:
            freq_dict[supertemplate] += 1
        else:
            freq_dict[supertemplate] = 1
        idxes.append(sentence_type)
    supertemplate = "rent"
    print('test1', supertemplate)

    templates_to_use = get_templates(supertemplate)
    template_idx = 0

    out_str = " "
    min_idxes = 0
    while template_idx < len(templates_to_use):
        if min_idxes > len(text):
            break
        # find the template and sentence to use
        for i in range(min_idxes, len(idxes)):
            # use it to fill out blanks and merge with overall contract text
            if infer_sentence_similarity(load_template(template_idx)[0], text[idxes[i]]) > 0.999:
                response, local_text = fill_template_blanks(templates_to_use[template_idx], text[idxes[i]])
                local_text = local_text.split("]")
                
                new_str = ""
                new_str += local_text[0]
                new_str += response[0]
                if len(local_text) > 1:
                    new_str += local_text[1]
                print("new_str", new_str)
                out_str += new_str
                min_idxes += 1
                break

        template_idx += 1
    print("out_str", out_str)
    return jsonify(text=out_str)
        



"""
Classifies a given sentence given its similarity to pre defined templates.
"""
def classify_sentence(sentence):
    # preprocess sentence
    import re
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = sentence.lower()
    sentence = sentence.split(" ")

    # compare to all possible sentence templates
    vectors, supertemplates = get_all_sentence_vectors()
    print(supertemplates)
    max_idx = 0
    supertemplate = "null_template"
    for i, vector in enumerate(vectors):
        similarity = infer_sentence_similarity(vector, sentence)
        if similarity > 0.95:
            max_idx = i
            supertemplate = supertemplates[i]
    print(supertemplate)
    return max_idx, "rent"

"""
Fills speech defined spaces/blanks in a given template sentence.
template: template sentence index to match against
speech_sentence: the speech sentence to match
return: template_responses; a list of strings, each matching its corresponding blank in the selected template
"""
def fill_template_blanks(template, speech_sentence):
    # preprocessing
    import re
    speech_sentence = re.sub(r'[^\w\s]','',speech_sentence)
    speech_sentence = speech_sentence.lower()
    speech_sentence = speech_sentence.split(" ")

    reference_vector, reference_text = load_template(template)

    template_responses = []
    # search for most similar in each
    # consecutive similarities are grouped together
    for i, vector in enumerate(reference_vector):
        # compute word similarity
        last_similarity = 0
        words = []
        for j in range(len(speech_sentence)):
            similarity = infer_word_similarity(reference_vector[i], speech_sentence[j])
            if (not similarity > 0.999) and last_similarity > 0.999:
                break
            if len(words) > 0 and last_similarity > 0.999 and similarity > 0.999:
                words.append(speech_sentence[j])
            elif len(words) == 0 and similarity > 0.999:
                words.append(speech_sentence[j])
            last_similarity = similarity

        out_str = ""
        for word in words:
            out_str += word
        template_responses.append(out_str)
    return template_responses, reference_text


if __name__ == '__main__':
    app.run(debug=True, port=5000) #run app in debug mode on port 5000