from discopy import Circuit, Id, Measure 
from sympy import default_sort_key
from multiprocessing import Pool, cpu_count
from pytket.extensions.qulacs import QulacsBackend
from lambeq.ansatz import Symbol
from itertools import product
from sympy import lambdify
import numpy as np

def get_rng(seed):
    return np.random.default_rng(seed)

def eval_circuit(circ, params, n_shots=1, seed=0, backend=QulacsBackend, compilation=None):
    """
    Return the evaluation result of
    an input circuit.
    """
    if compilation is None:
        compilation = backend.default_compilation_pass(2)
    return Circuit.eval(circ(*params), 
                        backend=backend, 
                        compilation=compilation, 
                        n_shots=n_shots)

def get_sorted_symbols(circuits):
    """
    Return a sorted list of free symbols in all of the
    given circuits.
    The sorting uses the default_sort_key in sympy
    """
    symbols = set()
    for circ in circuits:
        symbols.update(circ.free_symbols)
    return sorted(symbols, key=default_sort_key)

def random_params(symbols, rng):
    return rng.random(len(symbols))

def get_reduction_tensor(name, n_inputs):
    symbols = [Symbol(f"{name}_{''.join(str(t))}") for t in product([0, 1], repeat=n_inputs+1)]
    tensor = np.array(symbols, dtype=Symbol)
    tensor = tensor.reshape([2] * (n_inputs + 1))
    return tensor

def make_pred_fn(circuits, symbols, post_process=None, **kwargs):
    backend = kwargs.get('backend', None)
    compilation = kwargs.get('compilation', None)
    n_shots = kwargs.get('n_shots', 1)
    seed = kwargs.get('seed', 0)
    parallel_eval = kwargs.get('parallel_eval', True)
    
    measured_circuits = [c >> Id().tensor(*[Measure()] * len(c.cod)) for c in circuits]
    circuit_fns = [c.lambdify(*symbols) for c in measured_circuits]   

    if parallel_eval:
        def predict(params):
            pool = Pool(cpu_count())
            outputs = pool.map(eval_circuit, [(c, params, n_shots, seed) for c in circuit_fns])
            if post_process:
                outputs = list(map(post_process, outputs))
                outputs = [post_process(o, params) for o in outputs]
            assert all(np.array(o).shape == (2,) for o in outputs)
            assert all(abs(sum(o) - 1) < 1e-6 for o in outputs)
            return np.array(outputs)
    else:
        def predict(params):
            outputs = Circuit.eval(*(c_fn(*params) for c_fn in circuit_fns), **kwargs)
            if post_process:
                outputs = list(map(post_process, outputs))
                outputs = [post_process(o, params) for o in outputs]
            assert all(np.array(o).shape == (2,) for o in outputs)
            assert all(abs(sum(o) - 1) < 1e-6 for o in outputs)
            return np.array(outputs)

    return predict

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def make_default_post_process(reduction_tensor, symbols):
    tensor = reduction_tensor
    tensor_fn = lambdify(symbols, tensor)

    def post_process(output, params):
        output = np.array(output.array)
        if output.shape == (2,):
            output += 1e-9
            output = output / output.sum()
        elif output.shape == (2, 2):
            output = np.einsum('pij,ij->p', tensor_fn(params), output)
            output = softmax(output)
        else:
            raise NotImplementedError("Diagrams with more than 2 sentence outputs are not supported yet")
        return output
    return post_process

def make_cost_fn(pred_fn, labels):
    def cost_fn(params, **kwargs):
        predictions = pred_fn(params)

        cost = -np.sum(labels * np.log(predictions)) / len(labels)  # binary cross-entropy loss
        costs.append(cost)

        acc = np.sum(np.round(predictions) == labels) / len(labels) / 2  # half due to double-counting
        accuracies.append(acc)

        return cost

    costs, accuracies = [], []
    return cost_fn, costs, accuracies

