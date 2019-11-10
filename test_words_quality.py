word_to_candidate = {}
with open('./crawl-300d-2M.vec') as f:
    for line in f:
        word = line.split()[0]
        candidates = word_to_candidate.get(word.lower(), [])
        candidates.append(word)
        word_to_candidate[word] = candidates

for k, v in word_to_candidate.items():
    if len(v) > 1:
        print("{}:{}".format(k, v))