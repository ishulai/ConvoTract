# templates holds the list of template indices for each supertemplate (ie rent)
templates = {
    "rent": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # holds indices of each sentence template in template_lookup for the rent supertemplate
}
# template_lookup holds templates for all sentences
template_lookup = [
    {
        # reference vectors are labeled vectors of the answers to be inserted, taken from actual conversations
        "reference_vector": [],
        "reference_text": "",
        "supertemplate": "rent",
        # sentence vector is the averaged vector of all words in a sentence
        "sentence_vector": []
    }
]

import pickle
with open('./rent.pkl', 'rb') as rent_pkl:
    template_lookup = pickle.load(rent_pkl)



def load_template(template):
    reference_vector = template_lookup[template]['reference_vector']
    reference_text = template_lookup[template]['reference_text']
    return reference_vector, reference_text

def get_all_sentence_vectors():
    return [template_lookup[i]["sentence_vector"] for i in range(len(template_lookup))], [template_lookup[i]["supertemplate"] for i in range(len(template_lookup))]


def get_templates(supertemplate):
    return templates[supertemplate]