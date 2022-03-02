import pytest
from discopy import Word, Cup, Id, Ty
from discopro.grammar import is_pregroup, tensor, words_and_cups

@pytest.fixture
def alice_loves_bob_diagram():
    N = Ty('n')
    S = Ty('s')
    alice = Word('Alice', N)
    loves = Word('loves', N.r @ S @ N.l)
    bob = Word('Bob', N)

    return alice @ loves @ bob >> Cup(N, N.r) @ Id(S) @ Cup(N.l, N)

def test_is_pregroup(alice_loves_bob_diagram):
    diag = alice_loves_bob_diagram
    assert is_pregroup(diag)
    assert not is_pregroup(diag.normal_form())

def test_words_and_cups(alice_loves_bob_diagram):
    diag = alice_loves_bob_diagram

    words, cups = words_and_cups(diag)
    assert len(words) == 3
    assert len(cups) == 2
    
    assert all(isinstance(box, Word) for box in words.boxes)
    assert all(not isinstance(box, Word) for box in cups.boxes)

def test_words_and_cups_not_pregroup(alice_loves_bob_diagram):
    diag = alice_loves_bob_diagram
    with pytest.raises(ValueError) as e_info:
        words_and_cups(diag.normal_form())

def test_tensor(alice_loves_bob_diagram):
    diag = alice_loves_bob_diagram
    assert is_pregroup(tensor(diag, diag))




