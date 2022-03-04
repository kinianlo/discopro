from discopy import Diagram, Id, Ty, Cup, Cap, Swap, Word, Functor
from lambeq.rewrite import SimpleRewriteRule
from lambeq.core.types import AtomicType
from discopy.monoidal import Swap as monoidal_Swap

pronoun_rule = SimpleRewriteRule(
        cod=AtomicType.NOUN >> AtomicType.NOUN,
        template=Cap(AtomicType.NOUN.r, AtomicType.NOUN),
        words=['he', 'she', 'it', 'they'])

def follow_wire_up(diag, box_idx, wire_offset):
    """
    Follow an input wire of a box until the wire meets a box or the boundary.

    Parameters
    ----------
    diag: discopy.rigid.diagram
        Any diagram.
    box_idx: int
        The index of the box to start following from. Such that `diag.boxes[box_idx]` is 
        the box object.
    wire_offset:
        The offset of the wire to follow from. 
    """
    if not 0 <= wire_offset - diag.offsets[box_idx] < len(diag.boxes[box_idx].dom):
        raise ValueError("The wire does not belong to the box.")
   
    left_obstruction, right_obstuction = [], []
    while box_idx > 0: # Go up the diagram one layer at a time
        box_idx -= 1
        box, offset = diag.boxes[box_idx], diag.offsets[box_idx]
        if offset <= wire_offset < offset + len(box.cod): # wire hits a box
            return box_idx, wire_offset, left_obstruction, right_obstuction
        if offset <= wire_offset: # wire is on the right of the box
            wire_offset += len(box.dom) - len(box.cod)
            left_obstruction.append(box_idx)
        else:
            right_obstuction.append(box_idx)
    return -1, wire_offset, left_obstruction, right_obstuction

def follow_wire_down(diag, box_idx, wire_offset):
    if not 0 <= wire_offset - diag.offsets[box_idx] < len(diag.boxes[box_idx].cod):
        raise ValueError("The wire does not belong to the box.")

    left_obstruction, right_obstuction = [], []
    while box_idx < len(diag) - 1: # Go down the diagram one layer at a time
        box_idx += 1
        box, offset = diag.boxes[box_idx], diag.offsets[box_idx]
        if offset <= wire_offset < offset + len(box.dom): # wire hits a box
            return box_idx, wire_offset, left_obstruction, right_obstuction
        if offset <= wire_offset: # wire is on the right of the box
            wire_offset += len(box.cod) - len(box.dom)
            left_obstruction.append(box_idx)
        else:
            right_obstuction.append(box_idx)
    return len(diag), wire_offset, left_obstruction, right_obstuction

def follow_wire(diag, box_idx, wire_offset, direction='cod'):
    """
    Follow a wire until it meets a box that is not a 
    cup, a cap or a swap.

    Parameters
    ----------
    wire_offset: int
        Offset of the starting wire. 
    direction: str ('cod' or 'dom')
        Whether to start following on the domain or the codomain.
        
    Returns
    -------
    paths: list
        A list of paths. Each path contains the starting point 
        and the ending point, along with a list of boxes which
        was observed to be on the left of the wire and another
        list of boxes which was observed to be on the right of
        the wire.
    """
    paths = []
    while True:
        path = dict()
        path['start'] = (box_idx, wire_offset)

        if direction == 'cod':
            box_idx, wire_offset, left_obstr, right_obstr = follow_wire_down(diag, box_idx, wire_offset)
        elif direction == 'dom':
            box_idx, wire_offset, left_obstr, right_obstr = follow_wire_up(diag, box_idx, wire_offset)
        else:
            raise ValueError(f"Unknown direction: {direction}")

        path['end'] = (box_idx, wire_offset)
        path['obstruction'] = (left_obstr, right_obstr)
        paths.append(path)

        # decide whether to continue
        if box_idx == len(diag):
            return paths
        if isinstance(diag.boxes[box_idx], Cup):
            if wire_offset == diag.offsets[box_idx]:
                wire_offset += 1
            elif wire_offset == diag.offsets[box_idx] + 1:
                wire_offset -= 1
            else:
                raise ValueError("Something is wrong")
            direction = 'dom'
        elif isinstance(diag.boxes[box_idx], Cap):
            if wire_offset == diag.offsets[box_idx]:
                wire_offset += 1
            elif wire_offset == diag.offsets[box_idx] + 1:
                wire_offset -= 1
            else:
                raise ValueError("Something is wrong")
            direction = 'cod'
        elif isinstance(diag.boxes[box_idx], monoidal_Swap):
            if wire_offset == diag.offsets[box_idx]:
                wire_offset += 1
            elif wire_offset == diag.offsets[box_idx] + 1:
                wire_offset -= 1
            else:
                raise ValueError("Something is wrong")
        else:
            return paths

def _try_contract_new(diag):
    """
    Attempts to contract a box that is connected to only one 
    other box. If not such box is found, no contraction will be
    performed and None is returned.

    Method:

    """
    pass



def _try_contract_leaf(diag):
    """
    Attempts to contract a box that is connected to only one 
    other box. If not such box is found, no contraction will be
    performed and None is returned.
    """
    for self_box_idx, self in enumerate(diag.boxes):
        if isinstance(self, Cup) or isinstance(self, Cap) or isinstance(self, monoidal_Swap):
            continue

        paths_list = [follow_wire(diag, self_box_idx, diag.offsets[self_box_idx]+ob_idx) for ob_idx in range(len(self.cod))]

        other_boxes_idx = [paths[-1]['end'][0] for paths in paths_list]

        if len(set(other_boxes_idx)) != 1: # the box is connected more than one box
            continue
        other_box_idx = other_boxes_idx[0]

        if other_box_idx == len(diag): # self is connected to the bottom boundary
            continue

        other, other_offset = diag.boxes[other_box_idx], diag.offsets[other_box_idx]

        # follow_wire should not terminate at a Cap, a Cup or a Swap
        assert not ( isinstance(other, Cup) or isinstance(other, Cap) or isinstance(other, monoidal_Swap) )

        # Up until now, we have confirmed that self is connect to another word box 
        other_wire_offsets = [paths[-1]['end'][1] for paths in paths_list]
        other_ob_idx = list(map(lambda x: x-other_offset, other_wire_offsets))

        if max(other_ob_idx) - min(other_ob_idx) + 1 != len(other_ob_idx):
            # if there is a gap in the outputs of the other box 
            continue

        other_ob_z = [other.cod[i].z for i in other_ob_idx]
        self_ob_z = [ob.z for ob in self.cod]

        ob_z_diff_list = [other-self for other, self in zip(other_ob_z, self_ob_z)]
        if len(set(ob_z_diff_list)) != 1:
            continue
        ob_z_diff = ob_z_diff_list[0]

        assert ob_z_diff % 2 == 1

        new_self = self
        for i in range(abs(ob_z_diff)):
            new_self = getattr(new_self, 'r' if ob_z_diff > 0 else 'l')
        new_self = new_self.dagger()

        perm = diag.permutation([i - min(other_ob_idx) for i in other_ob_idx[::-1]],
                                other.cod[min(other_ob_idx): max(other_ob_idx)+1])
        new_other = Diagram(other.dom, 
                            Ty(*[ob for i, ob in enumerate(other.cod) if i not in other_ob_idx]),
                            [other, perm, new_self],
                            [0, min(other_ob_idx), min(other_ob_idx)])

        new_boxes = diag.boxes
        new_offsets = diag.offsets

        new_boxes[other_box_idx] = new_other
        # adjust offsets for right obstruction
        right_obstruction = [i for paths in paths_list for path in paths for i in path['obstruction'][1]]
        for i in right_obstruction:
            new_offsets[i] -= 1

        # remove self box and the cups, caps and swaps in the paths
        remove_boxes_idx = [path['end'][0] for paths in paths_list for path in paths[:-1]]
        remove_boxes_idx.append(self_box_idx)
        new_boxes = [box for i, box in enumerate(new_boxes) if i not in remove_boxes_idx]
        new_offsets = [off for i, off in enumerate(new_offsets) if i not in remove_boxes_idx]

        return Diagram(diag.dom, diag.cod, new_boxes, new_offsets), True
    return diag, False

def _try_contract_self_loop(diag):
    """
    Try to identify a self-loop and contract the loop.
    """
    for self_box_idx, self in enumerate(diag.boxes):
        if isinstance(self, Cup) or isinstance(self, Cap):
            continue

        paths_list = [follow_wire(diag, self_box_idx, diag.offsets[self_box_idx]+ob_idx) for ob_idx in range(len(self.cod))]

        other_boxes_idx = [paths[-1]['end'][0] for paths in paths_list]

        self_to_self_ob_idx = [ob_idx for ob_idx, box_idx in enumerate(other_boxes_idx) if box_idx == self_box_idx]
        if self_to_self_ob_idx == 0:
            continue

        for self_ob_idx in self_to_self_ob_idx:
            other_ob_idx = paths_list[self_ob_idx][-1]['end'][1] - diag.offsets[self_box_idx]
            if other_ob_idx - self_ob_idx == 1:
                # this loop doesn't encircle anything else, proceed with this pair
                break
        else:
            continue

        new_self = Diagram(self.dom,
                           self.cod[:self_ob_idx] @ self.cod[other_ob_idx+1:],
                           [self, Cup(Ty(self.cod[self_ob_idx]), Ty(self.cod[other_ob_idx]))],
                           [0, self_ob_idx])

        new_boxes = diag.boxes
        new_offsets = diag.offsets

        new_boxes[self_box_idx] = new_self
        # adjust offsets for right obstruction
        right_obstruction = [i for path in paths_list[self_ob_idx] for i in path['obstruction'][1]]
        for i in right_obstruction:
            new_offsets[i] -= 1

        # remove self box and the cups, caps and swaps in the paths
        remove_boxes_idx = [path['end'][0] for path in paths_list[self_ob_idx][:-1]]
        new_boxes = [box for i, box in enumerate(new_boxes) if i not in remove_boxes_idx]
        new_offsets = [off for i, off in enumerate(new_offsets) if i not in remove_boxes_idx]

        return Diagram(diag.dom, diag.cod, new_boxes, new_offsets), True
    return diag, False

def _try_contract_bending(diag):
    for self_box_idx, self in enumerate(diag.boxes):
        if isinstance(self, Cup) or isinstance(self, Cap) or isinstance(self, monoidal_Swap):
            continue

        if len(self.cod) == 0:
            continue

        paths_list = [follow_wire(diag, self_box_idx, diag.offsets[self_box_idx]+ob_idx) for ob_idx in range(len(self.cod))]

        other_boxes_idx = [paths[0]['end'][0] for paths in paths_list]

        if not all(isinstance(diag.boxes[i] if i < len(diag) else None, Cup) for i in other_boxes_idx):
            continue

        cups_idx = other_boxes_idx 

        cups_ob_idx = [paths_list[i][0]['end'][1] - diag.offsets[other_boxes_idx[i]] for i in range(len(self.cod))]
        assert all(i == 0 or i == 1 for i in cups_ob_idx)

        if not all(i == 0 or i == 1 for i in cups_ob_idx):
            continue

        if not len(set(cups_ob_idx)) == 1:
            continue

        right_bend = cups_ob_idx[0] == 0

        if right_bend:
            new_self = self.transpose(left=True).r.dagger()
        else:
            new_self = self.transpose(left=False).l.dagger()

        # Starting to construct the new diagram
        new_boxes = diag.boxes
        new_offsets = diag.offsets

        if right_bend:
            new_boxes[self_box_idx] = self.r.dagger()
        else:
            new_boxes[self_box_idx] = self.l.dagger()

        if not right_bend:
            new_offsets[self_box_idx] += len(self.cod)

        self_offset = diag.offsets[self_box_idx]
        for i in range(len(self.cod)):
            new_boxes.insert(self_box_idx + i, new_self.boxes[i])
            new_offsets.insert(self_box_idx + i, self_offset + i)


        new_diag = Diagram(diag.dom, diag.cod, new_boxes, new_offsets)
        new_diag = new_diag.normal_form()

        n_cups = sum(1 for box in diag.boxes if isinstance(box, Cup))
        new_n_cups = sum(1 for box in new_diag.boxes if isinstance(box, Cup))
        if new_n_cups >= n_cups:
            continue
        return new_diag, True
    return diag, False

def contract(diag, brute_force=False):
    success = True
    while success:
        diag, success = _try_contract_self_loop(diag)

    success = True
    while success:
        diag, success = _try_contract_leaf(diag)

    success = True
    while success:
        diag, success = _try_contract_bending(diag)

    return diag.flatten()
