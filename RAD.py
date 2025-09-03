"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np


def main(NumDomain):
    d, w0 = 2, 20  # 定义某些物理参数
    m, c, k = 1, 2 * d, w0 ** 2  # 根据给定的物理参数计算mu和k

    def ode(t, y):
        dy_dt = dde.grad.jacobian(y, t)
        d2y_dt2 = dde.grad.hessian(y, t)
        return m * d2y_dt2 + c * dy_dt + k * y - 0

    def func(t):
        # 定义欠阻尼谐振子问题的解析解
        d = 2
        w0 = 20
        w = np.sqrt(w0 ** 2 - d ** 2)  # 计算欠阻尼频率
        phi = np.arctan(-d / w)  # 计算初始相位
        A = 1 / (2 * np.cos(phi))  # 计算振幅
        cos = np.cos(phi + w * t)  # 计算余弦项
        exp = np.exp(-d * t)  # 计算指数衰减项
        u = exp * 2 * A * cos  # 计算解
        return u

    geom = dde.geometry.TimeDomain(0, 2)

    def boundary_l(t, on_initial):
        return on_initial and dde.utils.isclose(t[0], 0)

    def bc_func1(inputs, outputs, X):
        return outputs + 1

    def bc_func2(inputs, outputs, X):
        return dde.grad.jacobian(outputs, inputs, i=0, j=None) - 0

    ic1 = dde.icbc.IC(geom, lambda x: 1, lambda _, on_initial: on_initial)
    ic2 = dde.icbc.OperatorBC(geom, bc_func2, boundary_l)

    data = dde.data.TimePDE(geom, ode, ic_bcs=[ic1, ic2], num_domain=NumDomain - 2, num_boundary=2, solution=func,
                            num_test=500)
    layer_size = [1] + [32] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile(
        "adam", lr=0.001, metrics=["l2 relative error"], loss_weights=[0.0001, 1, 0.1]
    )
    losshistory, train_state = model.train(iterations=10000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    # model.compile(
    #     "adam", lr=0.0001, metrics=["l2 relative error"], loss_weights=[0.0001, 1, 0.1]
    # )
    for i in range(40):
        X = geom.random_points(1000)
        Y = np.abs(model.predict(X, operator=ode)).astype(np.float64)
        err_eq = Y / Y.mean() + 1
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        X_ids = np.random.choice(a=len(X), size=NumDomain - 2, replace=False, p=err_eq_normalized)
        X_selected = X[X_ids]
        data.replace_with_anchors(X_selected)
        losshistory, train_state = model.train(epochs=1000, callbacks=[])

    errors = losshistory.metrics_test.copy()
    errors = np.array(errors).reshape(-1, 1)
    # error_u = errors[:, 0:1]
    # error_du = errors[:, 1:2]
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    np.savetxt(f'errors_RLC1.txt', errors)
    # np.savetxt(f'error_du_RAD.txt', error_du)
    return errors


if __name__ == "__main__":
    main(NumDomain=100)
