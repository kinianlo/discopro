from discopy import Diagram, Id, Ty, Cup, Cap, Swap, Word
from discopy.monoidal import Swap as monoidal_Swap

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

def _try_contract(diag):
    """
    Attempts to contract a box that is connected to only one 
    other box. If not such box is found, no contraction will be
    performed and None is returned.
    """
    for box_idx, self in enumerate(diag.boxes):
        if not isinstance(self, Word):
            continue

        paths_list = [follow_wire(diag, box_idx, diag.offsets[box_idx]+ob_idx) for ob_idx in range(len(self.cod))]

        other_boxes_idx = [paths[-1]['end'][0] for paths in paths_list]

        if len(other_boxes_idx) != 1:
            # the box is connected more than one box
            continue
        other_box_idx = other_boxes_idx[0]
        other = diag.boxes[other_box_idx]
        other_offset = diag.offsets[other_box_idx]

        # Now we have confirmed that 
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

        right_obstruction = [i for paths in paths_list for path in paths for i in path['obstruction'][1]]

        wireboxes_idx = [path['end'][0] for paths in paths_list for path in paths[:-1]]

        new_self = self.dagger()
        if ob_z_diff > 0:
            for i in range(ob_z_diff):
                new_self = new_self.r
        elif ob_z_diff < 0:
            for i in range(-ob_z_diff):
                new_self = new_self.l

        print(other_ob_idx)
        perm = diag.permutation([i - min(other_ob_idx) for i in other_ob_idx],
                other.cod[min(other_ob_idx): max(other_ob_idx)+1])
        new_other = Diagram(other.dom, 
                Ty(*[ob for i, ob in enumerate(other.cod) if i not in other_ob_idx]),
                [other, perm, new_self],
                [0, min(other_ob_idx), min(other_ob_idx)])

        new_boxes = diag.boxes
        new_offsets = diag.offsets
        # adjust offsets for right obstruction
        for i in right_obstruction:
            new_offsets[i] -= 1

        new_boxes[other_box_idx] = new_other

        new_boxes = [box for i, box in enumerate(new_boxes) if i not in wireboxes_idx]
        new_offsets = [off for i, off in enumerate(new_offsets) if i not in wireboxes_idx]

        new_boxes = [box for i, box in enumerate(new_boxes) if i != box_idx]
        new_offsets = [off for i, off in enumerate(new_offsets) if i != box_idx]

        return Diagram(diag.dom, diag.cod, new_boxes, new_offsets)
    return None

def contract(diag):
    while True:
        contracted_diag = _try_contract(diag)
        if contracted_diag is None:
            return diag.flatten()
        diag = contracted_diag
