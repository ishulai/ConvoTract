from inference import compute_word_vector, compute_sentence_vector


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

lease_convo = [

]

formatted = []
for i, line in enumerate(lease):
    new_obj = {
        # reference vectors are labeled vectors of the answers to be inserted, taken from actual conversations
        "reference_vector": [],
        "reference_text": line,
        "supertemplate": "rent",
        # sentence vector is the averaged vector of all words in a sentence
        "sentence_vector": []
    }
    # create the sentence vector
    sentence = re.sub(r'[^\w\s]', '', line)
    sentence = sentence.lower()
    sentence = sentence.split(" ")
    new_obj["sentence_vector"] = compute_sentence_vector(sentence)

    new_obj["reference_vector"] = compute_word_vector(lease_convo[i])

    formatted.append(new_obj)

import pickle
with open('./rent.pkl', 'wb') as rent_pkl:
    pickle.dump(formatted, rent_pkl)