lines = []

import string

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

with open('./parsed_lines.txt', 'w') as parsed_lines:
    with open('./movie_lines.txt', 'rb') as movie_lines:
        movie_lines = movie_lines.read().decode(errors='ignore')
        movie_lines = movie_lines.split('\n')
        new_str = ""
        for line in movie_lines:
            # get rid of bullshit starting lines
            index = find_nth(line, "+++$+++", 4)
            index = index + 7
            new_line = line[index:]
            import re
            new_line = re.sub(r'[^\w\s]','',new_line)
            new_line = new_line.lower()

            new_str += new_line
        print(new_str, file=parsed_lines)            
