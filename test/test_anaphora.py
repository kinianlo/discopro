import pytest
from discopy import Id, Cup, Cap, Word, Ty, Swap
from discopy.monoidal import Swap as monoidal_Swap
from discopro.grammar import words_and_cups
from discopro.anaphora import _insert_cup, _surround_cup, _insert_cup_min_swaps, connect_anaphora, connect_anaphora_on_top

@pytest.fixture
def alice_loves_bob_diagram():
    N = Ty('n')
    S = Ty('s')
    alice = Word('Alice', N)
    loves = Word('loves', N.r @ S @ N.l)
    bob = Word('Bob', N)

    return alice @ loves @ bob >> Cup(N, N.r) @ Id(S) @ Cup(N.l, N)

@pytest.fixture
def john_sleeps_he_snores_diagram():
    N = Ty('n')
    S = Ty('s')
    john = Word('John', N)
    sleeps = Word('sleeps', N.r @ S)
    he = Word('He', N)
    snores = Word('snores', N.r @ S)

    return john @ sleeps @ he @ snores >> Cup(N, N.r) @ Id(S) @ Cup(N, N.r) @ Id(S)

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

def test_insert_cup(alice_loves_bob_diagram):
    diag = alice_loves_bob_diagram
    words, cups = words_and_cups(diag)
    t = Ty('t')
    cups_one_more = _insert_cup(cups, 1, 3, Cup(t, t.r))
    n_swaps = sum(1 for box in cups_one_more.boxes if isinstance(box, monoidal_Swap))

    assert n_swaps == 3
    assert cups.cod == cups_one_more.cod

def test_insert_cup_min_swaps(alice_loves_bob_diagram):
    diag = alice_loves_bob_diagram
    words, cups = words_and_cups(diag)
    t = Ty('t')
    cups_one_more = _insert_cup_min_swaps(cups, 0, 2, Cup(t, t.r))
    n_swaps = sum(1 for box in cups_one_more.boxes if isinstance(box, monoidal_Swap))

    assert n_swaps == 1
    assert cups.cod == cups_one_more.cod

def test_surround_cup(alice_loves_bob_diagram):
    diag = alice_loves_bob_diagram
    words, cups = words_and_cups(diag)
    T = Ty('t')
    cups_one_more = _surround_cup(cups, Cup(T, T.r))
    n_swaps = sum(1 for box in cups_one_more.boxes if isinstance(box, monoidal_Swap))

    assert n_swaps == 1
    assert cups.cod == cups_one_more.cod

def test_connect_anaphora(alice_loves_bob_he_does_diagram):
    diag = alice_loves_bob_he_does_diagram

    diag_conn = connect_anaphora(diag, 3, 0, min_swaps=False) 
    n_swaps = sum(1 for box in diag_conn.boxes if isinstance(box, monoidal_Swap))
    assert n_swaps == 4

    diag_conn = connect_anaphora(diag, 3, 0, min_swaps=True) 
    n_swaps = sum(1 for box in diag_conn.boxes if isinstance(box, monoidal_Swap))
    assert n_swaps == 2

def test_connect_anaphora_on_top(alice_loves_bob_he_does_diagram,
        alice_loves_bob_she_does_diagram):
    diag = alice_loves_bob_he_does_diagram

    diag_conn = connect_anaphora_on_top(diag, 3, 0) 
    n_swaps = sum(1 for box in diag_conn.boxes if isinstance(box, monoidal_Swap))
    n_cups = sum(1 for box in diag_conn.boxes if isinstance(box, Cup))
    n_caps = sum(1 for box in diag_conn.boxes if isinstance(box, Cap))
    assert n_swaps == 0
    assert n_caps == 0
    assert n_cups == 4

    diag_conn = connect_anaphora_on_top(diag, 3, 0, use_cap=True) 
    n_swaps = sum(1 for box in diag_conn.boxes if isinstance(box, monoidal_Swap))
    n_cups = sum(1 for box in diag_conn.boxes if isinstance(box, Cup))
    n_caps = sum(1 for box in diag_conn.boxes if isinstance(box, Cap))
    assert n_swaps == 0
    assert n_caps == 1
    assert n_cups == 5

    diag = alice_loves_bob_she_does_diagram

    diag_conn = connect_anaphora_on_top(diag, 3, 2) 
    n_swaps = sum(1 for box in diag_conn.boxes if isinstance(box, monoidal_Swap))
    n_cups = sum(1 for box in diag_conn.boxes if isinstance(box, Cup))
    n_caps = sum(1 for box in diag_conn.boxes if isinstance(box, Cap))
    assert n_swaps == 0
    assert n_caps == 0
    assert n_cups == 4

    diag_conn = connect_anaphora_on_top(diag, 3, 2, use_cap=True) 
    n_swaps = sum(1 for box in diag_conn.boxes if isinstance(box, monoidal_Swap))
    n_cups = sum(1 for box in diag_conn.boxes if isinstance(box, Cup))
    n_caps = sum(1 for box in diag_conn.boxes if isinstance(box, Cap))
    assert n_swaps == 0
    assert n_caps == 1
    assert n_cups == 5





    
