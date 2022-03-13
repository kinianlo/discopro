from discopy import Word

def is_pregroup(self):
    """
    Determine if a diagram is pregroup or not.
    """
    words_idx = [i for i, box in enumerate(self.boxes) if isinstance(box, Word)]
    return len(words_idx) == 0 or words_idx[-1] == len(words_idx) - 1

def words_and_cups(self):
    """
    Given a pregroup diagram, return separately the words
    and cups (and possibly swaps).
    """
    words_idx = [i for i, box in enumerate(self.boxes) if isinstance(box, Word)]
    if len(self) > 0 and words_idx[-1] != len(words_idx) - 1:
        raise ValueError("The given diagram is not a pregroup diagram")
    last_word_box_idx = words_idx[-1] if len(self) > 0 else -1
    return self[:last_word_box_idx+1], self[last_word_box_idx+1:]

def _tensor(self, other):
    """
    Return the tensor product of two pregroup diagrams.
    """
    if len(self) == 0:
        return other
    if len(other) == 0:
        return self
    f0, g0 = words_and_cups(self)
    f1, g1 = words_and_cups(other)
    return f0 @ f1 >> g0 @ g1

def tensor(self, *others):
    """
    Return the tensor product of pregroup diagrams.
    """
    if len(others) == 1:
        return _tensor(self, others[0])
    else:
        return _tensor(self, _tensor(others[0], *others[1:]))
