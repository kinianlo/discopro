from discopy import Word

def is_pregroup(self):
    """
    Determine if a diagram is pregroup or not.
    """
    words_idx = [i for i, box in enumerate(self.boxes) if isinstance(box, Word)]
    return words_idx[-1] == len(words_idx) - 1

def words_and_cups(self):
    """
    Given a pregroup diagram, return separately the words
    and cups (and possibly swaps).
    """
    words_idx = [i for i, box in enumerate(self.boxes) if isinstance(box, Word)]
    if words_idx[-1] != len(words_idx) - 1:
        raise ValueError("The given diagram is not a pregroup diagram")
    last_word_box_idx = words_idx[-1]
    return self[:last_word_box_idx+1], self[last_word_box_idx+1:]

def tensor(self, other):
    """
    Return the tensor product of two pregroup diagrams.
    """
    f0, g0 = words_and_cups(self)
    f1, g1 = words_and_cups(other)
    return f0 @ f1 >> g0 @ g1

