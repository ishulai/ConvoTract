# templates holds the list of template indices for each supertemplate (ie rent)
templates = {
    "rent": [] # holds indices of each sentence template in template_lookup for the rent supertemplate
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


def load_template(template):
    reference_vectors = template_lookup[template].reference_vectors
    reference_text = template_lookup[template].reference_text
    return reference_vectors, reference_text

def get_all_sentence_vectors():
    return [template_lookup[i]["sentence_vector"] for i in range(len(template_lookup))], [template_lookup[i]["supertemplate"] for i in range(len(template_lookup))]


def get_templates(supertemplate):
    return templates[supertemplate]