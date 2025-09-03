"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import os

os.environ["DDEBACKEND"] = "tensorflow"
import deepxde as dde
import numpy as np
from deepxde.backend import tf
import skopt
from distutils.version import LooseVersion
from scipy.integrate import solve_ivp


def quasirandom(n_samples, sampler):
    space = [(0.0, 2.0)]
    if sampler == "LHS":
        sampler = skopt.sampler.Lhs(
            lhs_type="centered", criterion="maximin", iterations=1000
        )
    elif sampler == "Halton":
        sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
    elif sampler == "Hammersley":
        sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    elif sampler == "Sobol":
        # Remove the first point [0, 0, ...] and the second point [0.5, 0.5, ...], which
        # are too special and may cause some error.
        if LooseVersion(skopt.__version__) < LooseVersion("0.9"):
            sampler = skopt.sampler.Sobol(min_skip=2, max_skip=2, randomize=False)
        else:
            sampler = skopt.sampler.Sobol(skip=0, randomize=False)
            return np.array(
                sampler.generate(space, n_samples + 2)[2:]
            )
    return np.array(sampler.generate(space, n_samples))


def main(NumDomain, method):
    def masse_spring(t, u, m=1, c=4, k=400):
        F = 0
        E_matrix = np.array([[0, 1], [(-k / m), (-c / m)]])
        Q_matrix = np.array([0, F / m])
        return E_matrix @ u + Q_matrix

    m = 1
    c = 4
    k = 400
    initial_conditions = [1, 0]  # 初始条件
    t_start, t_fin = 0, 2

    res = solve_ivp(masse_spring, [t_start, t_fin], initial_conditions, args=(m, c, k), dense_output=True)

    def ode(t, y):
        dy_dt = dde.grad.jacobian(y, t)
        d2y_dt2 = dde.grad.hessian(y, t)
        return m * d2y_dt2 + c * dy_dt + k * y - 0

    def func(t):
        return res.sol(t[:, 0]).T[:, 0:1]

    def du(t):
        return res.sol(t[:, 0]).T[:, 1:2]

    def gen_traindata(num):
        tvals = np.linspace(0, 2, num)
        uvals = u(tvals)

        return np.reshape(tvals, (-1, 1)), np.reshape(uvals, (-1, 1))

    # geom = dde.geometry.Interval(0, 1)

    def boundary(t, on_initial):
        return on_initial and dde.utils.isclose(t[0], 0)

    geom = dde.geometry.TimeDomain(0, 2)

    def boundary_l(t, on_initial):
        return on_initial and dde.utils.isclose(t[0], 0)

    def bc_func1(inputs, outputs, X):
        return outputs + 1

    def bc_func2(inputs, outputs, X):
        return dde.grad.jacobian(outputs, inputs, i=0, j=None) - 0

    ic1 = dde.icbc.IC(geom, lambda x: 1, lambda _, on_initial: on_initial)
    ic2 = dde.icbc.OperatorBC(geom, bc_func2, boundary_l)

    if method == 'Grid':
        data = dde.data.TimePDE(geom, ode, ic_bcs=[ic1, ic2], num_domain=NumDomain - 2, num_boundary=2,
                                train_distribution="uniform", solution=func, num_test=500)
    elif method == 'Random':
        data = dde.data.TimePDE(geom, ode, ic_bcs=[ic1, ic2], num_domain=NumDomain - 2, num_boundary=2,
                                train_distribution="pseudo", solution=func, num_test=500)
    elif method in ['LHS', 'Halton', 'Hammersley', 'Sobol']:
        sample_pts = quasirandom(NumDomain - 2, method)
        data = dde.data.TimePDE(geom, ode, ic_bcs=[ic1, ic2], num_domain=NumDomain - 2, num_boundary=2,
                                train_distribution="uniform",
                                solution=func, num_test=500, anchors=sample_pts)

    # data = dde.data.TimePDE(geom, ode, ic_bcs=[ic1, ic2], num_domain=NumDomain-2, num_boundary=2, solution=func, num_test=500)
    layer_size = [1] + [50] * 5 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile(
        "adam", lr=0.01, metrics=["l2 relative error"], loss_weights=[0.0001, 1, 0.1]
    )
    losshistory, train_state = model.train(iterations=50000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    errors = losshistory.metrics_test.copy()
    errors = np.array(errors).reshape(-1, 1)
    # error_u = errors[:, 0:1]
    # error_du = errors[:, 1:2]
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    np.savetxt(f'errors_RLC2-uniform.txt', errors)
    # np.savetxt(f'error_du_RAD.txt', error_du)
    return errors


if __name__ == "__main__":
    main(NumDomain=100, method='Grid')
    # main(NumDomain=100, method='Random')
    main(NumDomain=100, method='LHS')
    main(NumDomain=100, method='Halton')
    # main(NumDomain=20, method='Hammersley')
    # main(NumDomain=20, method='Sobol')

