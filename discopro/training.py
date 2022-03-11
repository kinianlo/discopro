from discopy import Circuit, Id, Measure 
from sympy import default_sort_key
from multiprocessing import cpu_count
from pytket.extensions.qulacs.backends.qulacs_backend import QulacsBackend
from pytket.extensions.qiskit.backends.aer import AerBackend
from lambeq.ansatz import Symbol
from itertools import product
from sympy import lambdify
import numpy as np
from numpy.random import default_rng
from tqdm.auto import tqdm

def get_rng(seed):
    return np.random.default_rng(seed)

_QulacsBackend = QulacsBackend()
_AerBackend = AerBackend()

def eval_circuit(circuits, symbols, params, options):
    """
    Return the evaluation result of
    an input circuit.
    """
    n_shots = options.get('n_shots', 1)
    seed = options.get('seed', 0)
    backend_name = options.get('backend_name', None)
    compilation_optim_level = options.get('compilation_optim_level', 1)

    if backend_name == 'qulacs':
        backend = _QulacsBackend
        compilation = backend.default_compilation_pass(compilation_optim_level)
    elif backend_name == 'aer':
        backend = _AerBackend
        compilation = backend.default_compilation_pass(compilation_optim_level)
    else:
        backend = None
        compilation = None

    circuits_lam = [c.lambdify(*symbols) for c in circuits]

    return [Circuit.eval(circ_lam(*params), 
                        backend=backend, 
                        compilation=compilation, 
                        n_shots=n_shots,
                        seed=seed) for circ_lam in circuits_lam]

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
    return tensor, symbols

def make_pred_fn(circuits, symbols, post_process=None, **kwargs):
    pool = kwargs.get('pool', None)

    measured_circuits = [c >> Id().tensor(*[Measure()] * len(c.cod)) for c in circuits]
    if pool:
        # Make sure we only pass pickleable things to the pool.starmap
        wanted_keys = ['backend_name', 'compilation_optim_level', 'n_shots', 'seed']
        clean_kwargs = {key: val for key, val in kwargs.items() if key in wanted_keys}
        n_batches = cpu_count()
        batches = np.array_split(measured_circuits, n_batches)
        n_circuits = len(measured_circuits)
        def predict_parallel(params):
            outputs = pool.starmap(eval_circuit, [(batch, symbols, params, clean_kwargs) for batch in batches])
            outputs = [circ_eval for batch in outputs for circ_eval in batch]
            assert len(outputs) == n_circuits
            if post_process:
                outputs = [post_process(o, params) for o in outputs]
            assert all(np.array(o).shape == (2,) for o in outputs)
            assert all(abs(sum(o) - 1) < 1e-6 for o in outputs)
            return np.array(outputs)
        return predict_parallel
    else:
        circuit_fns = [c.lambdify(*symbols) for c in measured_circuits]   
        def predict(params):
            outputs = Circuit.eval(*(c_fn(*params) for c_fn in circuit_fns), **kwargs)
            if post_process:
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
            output = np.einsum('pij,ij->p', tensor_fn(*params), output)
            output = softmax(output)
        else:
            raise NotImplementedError("Diagrams with more than 2 sentence outputs are not supported yet")
        return output
    return post_process

def make_cost_fn(pred_fn, labels):
    labels = np.array(labels)
    def cost_fn(params, **kwargs):
        predictions = pred_fn(params)

        cost = -np.sum(labels * np.log(predictions)) / len(labels)  # binary cross-entropy loss

        acc = np.sum(np.round(predictions) == labels) / len(labels) / 2  # half due to double-counting

        return cost, acc
    return cost_fn

def minimizeSPSA(func, x0, niter=100, start_from=0,
                 a=1.0, alpha=0.602, c=1.0, gamma=0.101,
                 func_dev=None, rng=None):
    """
    Minimization of an objective function by a simultaneous perturbation
    stochastic approximation algorithm.
    This algorithm approximates the gradient of the function by finite differences
    along stochastic directions Deltak. The elements of Deltak are drawn from
    +- 1 with probability one half. The gradient is approximated from the 
    symmetric difference f(xk + ck*Deltak) - f(xk - ck*Deltak), where the evaluation
    step size ck is scaled according ck =  c/(k+1)**gamma.
    The algorithm takes a step of size ak = a/(0.01*niter+k+1)**alpha along the
    negative gradient.
    
    See Spall, IEEE, 1998, 34, 817-823 for guidelines about how to choose the algorithm's
    parameters (a, alpha, c, gamma).
    Parameters
    ----------
    func: callable
        objective function to be minimized:
        called as `func(x, *args)`,
        if `paired=True`, then called with keyword argument `seed` additionally
    x0: array-like
        initial guess for parameters 
    niter: int
        number of iterations after which to terminate the algorithm
    a: float
       scaling parameter for step size
    alpha: float
        scaling exponent for step size
    c: float
       scaling parameter for evaluation step size
    gamma: float
        scaling exponent for evaluation step size 
    callback: callable
        called after each iteration, as callback(xk), where xk are the current parameters
    Returns
    -------
    x_hist, func_minus_hist, func_plus_hist, grad_hist
    """
    if rng is None:
        rng = default_rng(0)

    history = dict()
    history['params'] = list()
    history['cost_plus'] = list()
    history['cost_minus'] = list()
    history['acc_plus'] = list()
    history['acc_minus'] = list()
    history['grad'] = list()

    if func_dev:
        history['cost_dev'] = list()
        history['acc_dev'] = list()

    A = 0.01 * niter
    x, N = x0, len(x0)

    for k in tqdm(range(start_from, niter)):
        ak, ck = a/(k+1.0+A)**alpha, c/(k+1.0)**gamma

        Deltak = rng.choice([-1, 1], size=N)
        cost_plus, acc_plus = func(x + ck*Deltak)
        cost_minus, acc_minus = func(x - ck*Deltak)
        grad = (cost_plus - cost_minus) / (2*ck*Deltak)

        history['params'].append(x)
        history['cost_plus'].append(cost_plus)
        history['cost_minus'].append(cost_minus)
        history['acc_plus'].append(acc_plus)
        history['acc_minus'].append(acc_minus)
        history['grad'].append(grad)

        x -= ak*grad

        if func_dev:
            cost_dev, acc_dev = func_dev(x)
            history['cost_dev'].append(cost_dev)
            history['acc_dev'].append(acc_dev)
    return x, history

def plot_train_history(history):
    import matplotlib.pyplot as plt

    fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 6))
    ax_tl.set_title('Training set')
    ax_tr.set_title('Development set')
    ax_bl.set_xlabel('Iterations')
    ax_br.set_xlabel('Iterations')
    ax_bl.set_ylabel('Accuracy')
    ax_tl.set_ylabel('Loss')

    colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    ax_tl.plot(history['cost_plus'], color=next(colours))  # training evaluates twice per iteration
    ax_bl.plot(history['acc_plus'], color=next(colours))   # so take every other entry
    ax_tr.plot(history['cost_dev'], color=next(colours))
    ax_br.plot(history['acc_dev'], color=next(colours))

    fig.show()
