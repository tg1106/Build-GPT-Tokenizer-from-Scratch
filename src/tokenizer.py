# Day 1: a character-level tokenizer.
# Run from the project root:  python3 src/tokenizer.py

# "with open(...)" opens a file and closes it again when the block ends.
# f.read() returns the whole file as one string (a piece of text).
with open("data/input.txt", encoding="utf-8") as f:
    text = f.read()

# set(text) keeps each distinct character once.
# sorted(...) puts them in order and returns a list (an ordered collection).
chars = sorted(set(text))

# A dict (dictionary) maps a key to a value, like a phone book: name -> number.
# enumerate(chars) gives pairs (0, first char), (1, second char), ...
stoi = {}  # string to integer
itos = {}  # integer to string
for i, ch in enumerate(chars):
    stoi[ch] = i
    itos[i] = ch


# A function packages steps under a name. "def" defines it, "return" hands back the answer.
def encode(s):
    ids = []  # start with an empty list
    for ch in s:  # a for loop repeats once per character
        ids.append(stoi[ch])  # look up the number, add it to the end of the list
    return ids


def decode(ids):
    out = ""
    for i in ids:
        out = out + itos[i]  # + on strings glues them together
    return out


# f"..." is an f-string: {x} inside it is replaced by the value of x.
print(f"text length: {len(text)} characters")
print(f"vocabulary size: {len(chars)} unique characters")
print("vocabulary:", repr("".join(chars)))  # repr shows newline as \n so we can see it

print("encode('hello'):", encode("hello"))
print("decode(encode('hello')):", decode(encode("hello")))

# assert stops the program with an error if the claim is False.
# Here: encoding then decoding the whole book must give back the exact same text.
assert decode(encode(text)) == text
print("round trip on the full text: OK")
