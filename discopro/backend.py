import numpy as np
from itertools import product

def _get_counts(circuits, n_shots, symbols, params, backend):
    params_dict = {s.name:p for s, p in zip(symbols, params)}
    binds = [{p: params_dict[p.name] for p in c.parameters} for c in circuits]
    circuits_binded = [c.bind_parameters(b) for c, b in zip(circuits, binds)]

    job = backend.run(circuits_binded, shots=n_shots)
    counts = job.result().get_counts()

    if len(circuits) == 1:
        counts = [counts]

    return counts

def post_select(counts, post_selection, return_dict=False):
    n_qubits = len(next(iter(counts)))

    assert all(b < n_qubits for b in post_selection)

    free_qubits = [b for b in range(n_qubits) if b not in post_selection]
    bitstring = [post_selection.get(q, None) for q in range(n_qubits)]

    if return_dict:
        selected = dict()
    else:
        selected = np.zeros([2]*len(free_qubits), dtype=float)
        
    for p in product(range(2), repeat=len(free_qubits)):
        for b, i in zip(free_qubits, p):
            bitstring[b] = i
        bitstring_qiskit = ''.join(map(str, reversed(bitstring)))
        selected[p] = counts.get(bitstring_qiskit, 0)
    return selected

def eval_fast(circuits_qiskit, post_selections, n_shots, symbols, params, **kwargs):
    backend = kwargs.get('backend', None) 
    counts_raw = _get_counts(circuits_qiskit, n_shots, symbols, params, backend)
    counts = [post_select(c, p) for c, p in zip(counts_raw, post_selections)]
    counts = [c/n_shots for c in counts]
    return counts
