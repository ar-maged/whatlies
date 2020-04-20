def _selected_idx_spacy(string):
    if "[" not in string:
        if "]" not in string:
            return 0, len(string.split(" "))
    start, end = 0, -1
    split_string = string.split(" ")
    for idx, word in enumerate(split_string):
        if word[0] == "[":
            start = idx
        if word[-1] == "]":
            end = idx + 1
    return start, end
