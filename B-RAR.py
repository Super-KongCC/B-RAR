import os

os.environ["DDEBACKEND"] = "tensorflow"
import numpy as np
import deepxde as dde
from deepxde.backend import tf
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def plot_residual_heatmap(model, operator_func, save_path="residual_heatmap.png"):
    # 均匀采样 1000 个时间点
    t_test = np.linspace(0, 2, 1000).reshape(-1, 1)
    # 预测每个点的残差值
    residuals = np.abs(model.predict(t_test, operator=operator_func)).flatten()

    # 绘图
    plt.figure(figsize=(10, 2))
    plt.scatter(t_test, residuals, c=residuals, cmap='hot', s=20)
    plt.colorbar(label="Residual Magnitude")
    plt.xlabel("Time t")
    plt.ylabel("Residual")
    plt.title("PDE Residual Distribution across Time Domain")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.show()


def main(NumDomain):
    def masse_spring(t, u, m=1, c=4, k=400):
        F = 0
        E_matrix = np.array([[0, 1], [(-k / m), (-c / m)]])
        Q_matrix = np.array([0, F / m])
        return E_matrix @ u + Q_matrix

    m, c, k = 1, 4, 400
    initial_conditions = [1, 0]
    t_start, t_fin = 0, 2
    res = solve_ivp(masse_spring, [t_start, t_fin], initial_conditions, args=(m, c, k), dense_output=True)

    def ode(t, y):
        dy_dt = dde.grad.jacobian(y, t)
        d2y_dt2 = dde.grad.hessian(y, t)
        return m * d2y_dt2 + c * dy_dt + k * y

    def func(t):
        return res.sol(t[:, 0]).T[:, 0:1]

    def boundary_l(t, on_initial):
        return on_initial and dde.utils.isclose(t[0], 0)

    def bc_func2(inputs, outputs, X):
        return dde.grad.jacobian(outputs, inputs, i=0, j=None)

    geom = dde.geometry.TimeDomain(0, 2)
    ic1 = dde.icbc.IC(geom, lambda x: 1, lambda _, on_initial: on_initial)
    ic2 = dde.icbc.OperatorBC(geom, bc_func2, boundary_l)

    data = dde.data.TimePDE(geom, ode, ic_bcs=[ic1, ic2], num_domain=NumDomain - 2,
                            num_boundary=2, solution=func, num_test=500)

    layer_size = [1] + [60] * 5 + [1]
    net = dde.nn.FNN(layer_size, activation="tanh", kernel_initializer="Glorot uniform")

    model = dde.Model(data, net)
    B = 1  # Bootstrap个数

    model.compile("adam", lr=0.01, metrics=["l2 relative error"], loss_weights=[0.0001, 1, 0.1])
    losshistory, train_state = model.train(iterations=10000)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    model.compile("adam", lr=0.001, metrics=["l2 relative error"], loss_weights=[0.0001, 1, 0.1])
    # 自适应采样过程
    for i in range(NumDomain - NumDomain // 2):
        X_candidates = geom.random_points(1000)
        residual_matrix = []

        for b in range(B):
            # Bootstrap 重采样已有训练点
            X_train = data.train_x_all
            idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
            Xb = X_train[idx]

            # 训练 bootstrap 模型
            net_b = dde.nn.FNN(layer_size, activation="tanh", kernel_initializer="Glorot uniform")
            data_b = dde.data.TimePDE(geom, ode, ic_bcs=[ic1, ic2], anchors=Xb, num_test=500)
            model_b = dde.Model(data_b, net_b)
            model_b.compile("adam", lr=0.001, loss_weights=[0.0001, 1, 0.1])
            model_b.train(iterations=5, display_every=20, verbose=0)

            # 计算残差
            res_b = np.abs(model_b.predict(X_candidates, operator=ode)).astype(np.float64)[:, 0]
            residual_matrix.append(res_b)

        # 聚合残差
        residual_matrix = np.array(residual_matrix)  # shape: (B, N)
        mean_residual = np.mean(residual_matrix, axis=0)

        # 选取残差最大的位置
        X_ids = [np.argmax(mean_residual)]
        data.add_anchors(X_candidates[X_ids])
        losshistory, train_state = model.train(epochs=100)

    errors = losshistory.metrics_test.copy()
    errors = np.array(errors).reshape(-1, 1)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    np.savetxt('errors_RLC2-RAR-G.txt', errors)
    print("Relative Errors:", errors[-1, 0])


# 运行主程序
model, errors = main(NumDomain=100)
