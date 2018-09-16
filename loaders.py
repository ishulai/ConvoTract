templates = {
    "rent": []
}

template_lookup = [
    {
        "reference_vectors": [],
        "reference_text": 0,
        "names": [],
        "supertemplate": "rent"
    }
]


def load_template(template):
    reference_vectors = template_lookup[template].reference_vectors
    reference_text = template_lookup[template].reference_text
    return reference_vectors, reference_text

def get_all_sentence_vectors():
    sentence_vectors = []
    names = []
    supertemplates = []
    for j, template in enumerate(template_lookup):
        for i in range(len(template_lookup[j]['reference_vectors'])):
            sentence_vectors.append(template_lookup[j]["reference_vectors"][i])
            names.append(template_lookup[j]["names"][i])
            supertemplates.append(template_lookup[j]["supertemplate"])
    return sentence_vectors, names, supertemplates


def get_templates(supertemplate):
    return templates[supertemplate]