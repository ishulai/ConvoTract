lines = []

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

with open('./movie_lines.txt', 'rb') as movie_lines:
    movie_lines = movie_lines.read().decode(errors='ignore')
    for line in movie_lines:
        # get rid of bullshit starting lines
        index = find_nth(line, "+++$+++", 4)
        index = index + 7
        new_line = line[index:]

        lines.append(new_line)

import pickle
pickle.dump(lines, open('./parsed_lines.pkl', 'wb'))