import pytest
from qiskit import Aer, transpile
from discopy import Ty, Word, Cup, Id
from discopy.quantum import Measure
from pytket.extensions.qiskit.backends.aer import AerBackend
from pytket.extensions.qiskit.qiskit_convert import tk_to_qiskit
from lambeq.circuit import IQPAnsatz
from numpy.random import rand
from discopro.backend import eval_fast
import numpy as np

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

def test_eval(alice_loves_bob_he_does_diagram):
    backend_tket = AerBackend()
    backend_fast = Aer.get_backend('aer_simulator')
    n_shots = 2**16
    N = Ty('n')
    S = Ty('s')
    ansatz = IQPAnsatz({N: 1, S:1}, n_layers=2)
    diag = alice_loves_bob_he_does_diagram
    circ = ansatz(diag) >> Measure() @ Measure()
    circ_qiskit = tk_to_qiskit(circ.to_tk())
    circ_qiskit = transpile(circ_qiskit, backend_fast)
    post_selection = circ.to_tk().post_selection
    symbols = list(circ.free_symbols)

    for i in range(10):
        params = rand(len(symbols))
        output = circ.lambdify(*symbols)(*params).eval(backend=backend_tket, 
                                    compilation=backend_tket.default_compilation_pass(2), 
                                    n_shots=n_shots)
        array_tket = np.array(output.array)
        array_fast = eval_fast([circ_qiskit], [post_selection], n_shots, symbols, params, backend=backend_fast)[0]
        array_tket = array_tket/array_tket.sum()
        array_fast = array_fast/array_fast.sum()
        assert np.allclose(array_tket, array_fast, rtol=0.5, atol=0.1)

def test_eval_seed(alice_loves_bob_he_does_diagram):
    backend_fast = Aer.get_backend('aer_simulator')
    backend_fast.set_options(seed_simulator=0)
    n_shots = 2**10
    N = Ty('n')
    S = Ty('s')
    ansatz = IQPAnsatz({N: 1, S:1}, n_layers=2)
    ansatz = IQPAnsatz({N: 1, S:1}, n_layers=2)
    diag = alice_loves_bob_he_does_diagram
    circ = ansatz(diag) >> Measure() @ Measure()
    circ_qiskit = tk_to_qiskit(circ.to_tk())
    circ_qiskit = transpile(circ_qiskit, backend_fast)
    post_selection = circ.to_tk().post_selection
    symbols = list(circ.free_symbols)
    params = rand(len(symbols))
    array_fast1 = eval_fast([circ_qiskit], [post_selection], n_shots, symbols, params, backend=backend_fast)[0]
    array_fast2 = eval_fast([circ_qiskit], [post_selection], n_shots, symbols, params, backend=backend_fast)[0]
    assert np.allclose(array_fast1, array_fast2, rtol=0, atol=1e-9)
