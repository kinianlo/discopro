from discopy import Word, Diagram, Id, Ty, Cup, Cap, Swap
from discopro.grammar import words_and_cups, is_pregroup

def _surround_cup(diag, cup):
    """
    Given a general diagram, surround `diag` with the given `cup`
    while making the neccessary swaps to apply the cup.
    """
    ids = Diagram(diag.cod, diag.cod, [], [])
    cups = _insert_cup(ids, 0, len(diag.cod) - 1, cup)
    new_diag = Diagram(cup.dom[0:1] @ diag.dom @ cup.dom[1:2],
                        diag.cod,
                        [diag, cups],
                        [1, 0])
    return new_diag.flatten()

def _insert_cup(diag, left, right, cup):
    """
    Given a diagram of only cups, insert a cup such that the left input 
    of the cup is between [left-1, left] and the right input of the cup is
    between [right, right+1].
    """
    swaps = diag.swap(cup.dom[0:1], diag.dom[left:right+1])
    new_dom = diag.dom[:left] @ cup.dom[0:1] @ diag.dom[left:right+1] @ cup.dom[1:2] @ diag.dom[right+1:]

    new_diag = Diagram(new_dom, diag.cod,
                    [swaps, cup, diag],
                    [left, right+1, 0])
    return new_diag.flatten()

def _insert_cup_min_swaps(diag, left, right, cup):
    """
    Given a diagram of only cups, insert a cup such that the left input 
    of the cup is between [left-1, left] and the right input of the cup is
    between [right, right+1].
    The cups that are local to [left, right] will be applied before the
    added cup. This is done so to introduce the minimal number of swaps.
    
    TODO
    ----
    Currently this function only works for `diag` that contains only cups.
    Make this also works for `diag` that contains also swaps, which will 
    allow for recursive application of insert_cup.
    """
    n_diag_dom = len(diag.dom)
    wires = list(range(len(diag.dom)))
    
    local_boxes_idx = list()
    for i, (box, offset) in enumerate(zip(diag.boxes, diag.offsets)):
        box_left = wires[offset]
        box_right = wires[offset+len(box.dom)-1]
        if left <= box_left and box_right <= right:
            local_boxes_idx.append(i)
        if len(box.cod) == 0: # cup
            # Remove the wires eaten up by the cap
            wires = wires[:offset] + wires[offset+2:]
        elif len(box.dom) == len(box.cod): # swap
            # Do nothing here 
            pass
        else:
            raise NotImplementedError("Only cups and swaps allowed")
    # Move the local cups to the top
    n_local_boxes = len(local_boxes_idx)
    for target_idx, local_box_idx in enumerate(local_boxes_idx):
        diag = diag.interchange(local_box_idx, target_idx)
        
    # Split the diagram into top (only local boxes) and down (the other boxes)
    local_boxes = diag[:n_local_boxes]
    other_boxes = diag[n_local_boxes:]

    # Peel off the Ids on the left and on the right
    local_boxes = Diagram(local_boxes.dom[left:right+1], local_boxes.cod[left:-(n_diag_dom-right)+1], 
                          local_boxes.boxes, list(map(lambda x: x-left, local_boxes.offsets)))
    # Surround the local cups diagram with the input cup
    local_boxes = _surround_cup(local_boxes, cup)
    
    # put back the peeled of Ids
    local_boxes = Id(diag.dom[:left]) @ local_boxes @ Id(diag.dom[right+1:])
  
    return local_boxes >> other_boxes

def connect_anaphora(diag, pro_box_idx, ref_box_idx, min_swaps=False):
    """
    Return a pregroup diagram with the pronoun word and the referent word connected 
    with a Cup.

    Parameters
    ----------
    diag: discopy.rigid.Diagram
        Any pregroup diagram.
    pro_box_idx: int
        Index of the pronoun word box.
    ref_box_idx: int
        Index of the referent word box.
    min_swaps: bool
        Whether to try to minimise the number swaps.
    """
    if not is_pregroup(diag):
        raise ValueError(f"The given diagram is not a pregroup one")
    if not 0 <= pro_box_idx < len(diag):
        raise ValueError(f"Invalid pronoun box index: {pro_box_idx}")
    if not 0 <= ref_box_idx < len(diag):
        raise ValueError(f"Invalid referent box index: {ref_box_idx}")
    pro_box = diag.boxes[pro_box_idx]
    ref_box = diag.boxes[ref_box_idx]

    last_word_box_idx = max(i for i, box in enumerate(diag.boxes) if isinstance(box, Word))
    words_diag = diag[:last_word_box_idx+1]
    cups_diag = diag[last_word_box_idx+1:]
    
    # the index of the left most type srrounded by the added Cup
    left = sum(len(box.cod) for box in words_diag.boxes[:ref_box_idx+1])
    # the index of the right most type srrounded by the added Cup
    right = sum(len(box.cod) for box in words_diag.boxes[:pro_box_idx]) - 1
    
    N = Ty('n')
    # insert a cup to existing cups
    if min_swaps:
        cups_diag = _insert_cup_min_swaps(cups_diag, left, right, Cup(N, N.r))
    else:
        cups_diag = _insert_cup(cups_diag, left, right, Cup(N, N.r))
   
    # Create a new word diagrams with the pronoun box and referent box upgraded
    # to each has an extra output to allow connection via the cup
    new_boxes = words_diag.boxes
    
    new_boxes[pro_box_idx] = Word(pro_box.name, N.r @ pro_box.cod)
    new_boxes[ref_box_idx] = Word(ref_box.name, ref_box.cod @ N)
    
    new_words_diag = Id(Ty()).tensor(*new_boxes)
    
    return new_words_diag >> cups_diag

def connect_anaphora_on_top(diag, pro_box_idx, ref_box_idx, use_cap=False):
    """
    Return a non-pregroup diagram with the pronoun word box and
    the referent word box connected via a Cup on the top of the
    diagram.
    This method of connecting anaphora introduces no swaps but
    outputs a non-pregroup diagram. That means this can only be done
    once and therefore not recursively.

    Parameters
    ----------
    diag: discopy.rigid.Diagram
        Any pregroup diagram.
    pro_box_idx: int
        Index of the pronoun word box.
    ref_box_idx: int
        Index of the referent word box.
    use_cap: bool
        Whether to use a cap to connect or not.
    """
    if not is_pregroup(diag):
        raise ValueError(f"The given diagram is not a pregroup one")

    pro_box = diag.boxes[pro_box_idx] 
    ref_box = diag.boxes[ref_box_idx]

    words, cups = words_and_cups(diag)

    left_words_boxes = words.boxes[:ref_box_idx]
    mid_words_boxes = words.boxes[ref_box_idx+1:pro_box_idx]
    right_words_boxes = words.boxes[pro_box_idx+1:]

    left_words = Id(Ty()).tensor(*left_words_boxes)
    mid_words = Id(Ty()).tensor(*mid_words_boxes)
    right_words = Id(Ty()).tensor(*right_words_boxes)

    N = Ty('n')

    if not use_cap:
        new_diag = left_words @ Word(ref_box.name, ref_box.cod @ N) @ Word(pro_box.name, N.r @ pro_box.cod) @ right_words
        new_diag >>= Id(left_words.cod @ ref_box.cod) @ Cup(N, N.r) @ Id(pro_box.cod @ right_words.cod)
        new_diag >>= Id(left_words.cod @ ref_box.cod) @ mid_words @ Id(pro_box.cod @ right_words.cod)
    else:
        new_diag = Cap(N.r, N)
        new_diag >>= left_words @ Word(ref_box.name, ref_box.cod @ N) @ Id(N.r) @ mid_words @ Id(N) @ Word(pro_box.name, N.r @ pro_box.cod) @ right_words

        new_diag >>= Id(left_words.cod @ ref_box.cod) @ Cup(N, N.r) @ Id(mid_words.cod) @ Cup(N, N.r) @ Id(pro_box.cod @ right_words.cod)

    return new_diag >> cups
