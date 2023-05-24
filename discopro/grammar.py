from discopy import Word
from discopy.rigid import Diagram
from discopy import messages

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
    if len(others) == 0:
        return self
    elif len(others) == 1:
        return _tensor(self, others[0])
    else:
        return _tensor(self, tensor(others[0], *others[1:]))

def draw(diagram, **params):
    """
    Draws a pregroup diagram, i.e. of shape :code:`word @ ... @ word >> cups`.

    Parameters
    ----------
    width : float, optional
        Width of the word triangles, default is :code:`2.0`.
    space : float, optional
        Space between word triangles, default is :code:`0.5`.
    textpad : pair of floats, optional
        Padding between text and wires, default is :code:`(0.1, 0.2)`.
    draw_type_labels : bool, optional
        Whether to draw type labels, default is :code:`True`.
    aspect : string, optional
        Aspect ratio, one of :code:`['equal', 'auto']`.
    margins : tuple, optional
        Margins, default is :code:`(0.05, 0.05)`.
    fontsize : int, optional
        Font size for the words, default is :code:`12`.
    fontsize_types : int, optional
        Font size for the types, default is :code:`12`.
    figsize : tuple, optional
        Figure size.
    path : str, optional
        Where to save the image, if :code:`None` we call :code:`plt.show()`.
    pretty_types : bool, optional
        Whether to draw type labels with superscript, default is :code:`False`.
    triangles : bool, optional
        Whether to draw words as triangular states, default is :code:`False`.

    Raises
    ------
    ValueError
        Whenever the input is not a pregroup diagram.
    """
    from discopy.rigid import Swap, Id, Ty, Cup, Spider
    from discopro import drawing
    if not isinstance(diagram, Diagram):
        raise TypeError(messages.type_err(Diagram, diagram))
    words, is_pregroup = Id(Ty()), True
    for _, box, right in diagram.layers:
        if isinstance(box, Word):
            if right:  # word boxes should be tensored left to right.
                is_pregroup = False
                break
            words = words @ box
        else:
            break
    cups = diagram[len(words):].foliation().boxes\
        if len(words) < len(diagram) else []
    is_pregroup = is_pregroup and words and all(
        isinstance(box, Cup) or isinstance(box, Swap)\
        or isinstance(box, Spider)
        for s in cups for box in s.boxes)
    if not is_pregroup:
        raise ValueError(messages.expected_pregroup())
    drawing.pregroup_draw(words, cups, **params)