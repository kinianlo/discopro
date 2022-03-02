import pytest
from discopro.rewriting import follow_wire_down, follow_wire_up, follow_wire, _try_contract, contract
from discopy import Ty, Word, Cup, Id, Swap
from discopy.monoidal import Swap as monoidal_Swap
from discopro.anaphora import connect_anaphora_on_top, connect_anaphora

@pytest.fixture
def alice_loves_bob_diagram():
    N = Ty('n')
    S = Ty('s')
    alice = Word('Alice', N)
    loves = Word('loves', N.r @ S @ N.l)
    bob = Word('Bob', N)

    return alice @ loves @ bob >> Cup(N, N.r) @ Id(S) @ Cup(N.l, N)

@pytest.fixture
def alice_bob_loves_diagram():
    N = Ty('n')
    S = Ty('s')
    alice = Word('Alice', N)
    loves = Word('loves', N.r @ S @ N.l)
    bob = Word('Bob', N)

    diag = alice @ bob @ loves
    diag >>= Id(N) @ Swap(N, N.r) @ Id(S @ N.l)
    diag >>= Cup(N, N.r) @ Swap(N, S) @ Id(N.l)
    diag >>= Id(S) @ Cup(N, N.l)

    return diag 

@pytest.fixture
def alice_loves_bob_he_does_diagram():
    N = Ty('n')
    S = Ty('s')
    alice = Word('Alice', N)
    loves = Word('loves', N.r @ S @ N.l)
    bob = Word('Bob', N)
    he = Word('He', N)
    does = Word('does', N.r @ S)

    return alice @ loves @ bob @ he @ does >> Cup(N, N.r) @ Id(S) @ Cup(N.l, N) @ Cup(N, N.r) @ Id(S)

@pytest.fixture
def alice_loves_bob_she_does_diagram():
    N = Ty('n')
    S = Ty('s')
    alice = Word('Alice', N)
    loves = Word('loves', N.r @ S @ N.l)
    bob = Word('Bob', N)
    she = Word('She', N)
    does = Word('does', N.r @ S)

    return alice @ loves @ bob @ she @ does >> Cup(N, N.r) @ Id(S) @ Cup(N.l, N) @ Cup(N, N.r) @ Id(S)

def test_follow_wire_down(alice_loves_bob_diagram):
    diag = alice_loves_bob_diagram

    box_idx, wire_offset, left_obstruction, right_obstuction = follow_wire_down(diag, 0, 0)

    assert box_idx == 3 
    assert wire_offset == 0 
    assert left_obstruction == []
    assert right_obstuction == [1, 2]

    box_idx, wire_offset, left_obstruction, right_obstuction = follow_wire_down(diag, 1, 2)

    assert box_idx == 5
    assert wire_offset == 0
    assert left_obstruction == [3]
    assert right_obstuction == [2, 4]

def test_follow_wire_up(alice_loves_bob_diagram):
    diag = alice_loves_bob_diagram

    box_idx, wire_offset, left_obstruction, right_obstuction = follow_wire_up(diag, 3, 0)

    assert box_idx == 0
    assert wire_offset == 0
    assert left_obstruction == []
    assert right_obstuction == [2, 1]

    box_idx, wire_offset, left_obstruction, right_obstuction = follow_wire_up(diag, 3, 1)

    assert box_idx == 1
    assert wire_offset == 1
    assert left_obstruction == []
    assert right_obstuction == [2]

def test_follow_wire(alice_loves_bob_diagram):
    diag = alice_loves_bob_diagram

    paths = follow_wire(diag, 0, 0, direction='cod')

    assert len(paths) == 2
    path1, path2 = paths 

    assert path1["start"] == (0, 0)
    assert path1["end"] == (3, 0)
    assert path1["obstruction"] == ([], [1, 2])

    assert path2["start"] == (3, 1)
    assert path2["end"] == (1, 1)
    assert path2["obstruction"] == ([], [2])

    paths = follow_wire(diag, 1, 3, direction='cod')

    assert len(paths) == 2
    path1, path2 = paths 

    assert path1["start"] == (1, 3)
    assert path1["end"] == (4, 1)
    assert path1["obstruction"] == ([3], [2])

    assert path2["start"] == (4, 2)
    assert path2["end"] == (2, 4)
    assert path2["obstruction"] == ([3], [])

def test_follow_with_swaps(alice_bob_loves_diagram):
    diag = alice_bob_loves_diagram

    paths = follow_wire(diag, 0, 0, direction='cod')

    assert len(paths) == 3
    path1, path2, path3 = paths 

    assert path1["start"] == (0, 0)
    assert path1["end"] == (4, 0)
    assert path1["obstruction"] == ([], [1,2,3])

    assert path2["start"] == (4, 1)
    assert path2["end"] == (3, 1)
    assert path2["obstruction"] == ([], [])

    assert path3["start"] == (3, 2)
    assert path3["end"] == (2, 2)
    assert path3["obstruction"] == ([], [])

    paths = follow_wire(diag, 1, 1, direction='cod')
    print(paths[3])

    assert len(paths) == 4
    path1, path2, path3, path4 = paths 

    assert path1["start"] == (1, 1)
    assert path1["end"] == (3, 1)
    assert path1["obstruction"] == ([], [2])

    assert path2["start"] == (3, 2)
    assert path2["end"] == (5, 0)
    assert path2["obstruction"] == ([4], [])

    assert path3["start"] == (5, 1)
    assert path3["end"] == (6, 1)
    assert path3["obstruction"] == ([], [])

    assert path4["start"] == (6, 2)
    assert path4["end"] == (2, 4)
    assert path4["obstruction"] == ([5, 4, 3], [])

def test_try_contract(alice_loves_bob_diagram):
    diag = alice_loves_bob_diagram
    diag = _try_contract(diag)
    n_cups = sum(1 for box in diag.boxes if isinstance(box, Cup))
    assert n_cups == 1

    diag = _try_contract(diag)
    n_cups = sum(1 for box in diag.boxes if isinstance(box, Cup))
    assert n_cups == 0

def test_try_contract_with_swaps(alice_bob_loves_diagram):
    diag = alice_bob_loves_diagram
    diag = _try_contract(diag)
    assert diag is not None
    n_cups = sum(1 for box in diag.boxes if isinstance(box, Cup))
    n_swaps = sum(1 for box in diag.boxes if isinstance(box, Swap))
    assert n_cups == 1
    assert n_swaps == 1

    diag = _try_contract(diag)
    assert diag is not None
    n_cups = sum(1 for box in diag.boxes if isinstance(box, Cup))
    n_swaps = sum(1 for box in diag.boxes if isinstance(box, Swap))
    assert n_cups == 0
    assert n_swaps == 0

def test_contract(alice_loves_bob_diagram, 
        alice_bob_loves_diagram,
        alice_loves_bob_she_does_diagram):
    diag = alice_loves_bob_diagram
    diag = contract(diag)
    n_cups = sum(1 for box in diag.boxes if isinstance(box, Cup))
    assert n_cups == 0

    diag = alice_bob_loves_diagram
    diag = contract(diag)
    n_cups = sum(1 for box in diag.boxes if isinstance(box, Cup))
    assert n_cups == 0

    diag = alice_loves_bob_she_does_diagram
    diag = connect_anaphora_on_top(diag, 3, 0)
    diag = contract(diag)
    n_cups = sum(1 for box in diag.boxes if isinstance(box, Cup))
    assert n_cups == 3

    diag = alice_loves_bob_she_does_diagram
    diag = connect_anaphora(diag, 3, 0)
    diag = contract(diag)
    n_cups = sum(1 for box in diag.boxes if isinstance(box, Cup))
    assert n_cups == 3
