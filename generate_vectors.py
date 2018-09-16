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
    return [vectors[0][i][0] for i in range(len(vectors[0]))]

print(model.summary())
lease = [ 'IN CONSIDERATION OF the Landlord leasing certain premises to the Tenant and other valuable consideration, the receipt and sufficiency of which consideration is hereby acknowledged, the Parties agree as follows:'
          'Leased Property:'
          '1. The Landlord agrees to rent to the Tenant the apartment, municipally described as ], ], Massachusetts  ] (the "Property"), for use as residential premises only.'
		  '2. No guests of the Tenants may occupy the Property for longer than one week without the prior written consent of the Landlord.'
		  '3. No animals are allowed to be kept in or about the Property without the revocable written permission of the Landlord.'
		  '4. Subject to the provisions of this Lease, the Tenant is entitled to the use of parking on or about the Property.'
		  '5. The Tenant and members of Tenant\'s household will not smoke anywhere in the Property nor permit any guests or visitors to smoke in the Property.'
		  'Term'
		  '6. The term of the Lease commences on ] and ends at ].'
		  'Rent'
		  '7. Subject to the provisions of this Lease, the rent for the Property is $] per month (the "rent").'
		  '8. The Tenant will pay the Rent on or before the ] of each and every month of the term of this Lease to the Landlord at ] or at such other place as the Landlord may later designate by ]. An additional security deposit of [ must be included with the monthly rent.
		  '9. The Tenant is hereby advised and understands that the personal property of the Tenant is not insured by the Landlord for either damage or loss, and the Landlord assumes no liability for any such loss.'
		  '10. The Tenant will not assign this Lease, or sublet or grant any concession or license to use the Property or any part of the Property. Any assignment, subletting, concession, or license, whether by operation of law or otherwise, will be void and will, at Landlord's option, terminate this Lease.'
		  ]

formatted = []
for line in lease:
    new_obj = {
        "reference_vectors": [],
        "reference_text": line,
        "supertemplate": "rent"
    }
    formatted.append(new_obj)

import pickle
with open('./rent.pkl', 'wb') as rent_pkl:
    pickle.dump(formatted, rent_pkl)