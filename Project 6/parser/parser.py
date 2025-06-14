import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> S Conj S | S Conj VP
S -> NP VP
AP -> Adj | Adj AP
AdvP -> Adv | Adv AdvP
NP -> N | Det NP | AP NP | N PP
PP -> P NP
VP -> V | V NP | VP PP | AdvP VP | VP AdvP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    tokens = nltk.word_tokenize(sentence)
    words = []
    for token in tokens:
        for char in token:
            # If token has at least one alphabetic character
            if char.isalpha():
                # Add it to the list
                words.append(token.lower())
                break
    return words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    np_chunk_list = []
    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            # Check if the subtree contains any other NP subtrees
            contains_np = False
            for subsubtree in subtree.subtrees():
                # Ignore the subtree itself
                if subsubtree == subtree:
                    continue
                if subsubtree.label() == "NP":
                    contains_np = True
                    break
            # If it doesn't contain any NP subtrees
            if not contains_np:
                np_chunk_list.append(subtree)
    return np_chunk_list


if __name__ == "__main__":
    main()
