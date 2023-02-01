import cvxopt as cv
import neurokit2 as nk
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def eda_decompose(eda_signal: ArrayLike, sampling_rate: float, method: str = "highpass") -> pd.DataFrame:
    """Decomposes EDA signal into tonic and phasic components.

    Args:
        eda_signal (ArrayLike): EDA signal.
        sampling_rate (float): Sampling rate of EDA signal (Hz).
        method (str, optional): Method to be used for decomposition. Defaults to "highpass".

    Raises:
        ValueError: If sampling rate is not greater than 0.
        Exception: If method is not implemented.

    Returns:
        pd.DataFrame: A dataframe composed of Phasic and Tonic components of EDA signal
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    method = method.lower()

    if method == "cvxeda":
        decomposed = _cvxEDA(eda_signal, 1 / sampling_rate)

    elif method == "highpass":
        decomposed = _eda_highpass(eda_signal, sampling_rate)  # Default

    elif method == "bandpass":
        decomposed = _eda_bandpass(eda_signal, sampling_rate)
    else:
        raise Exception("Method not implemented")

    return decomposed


def _eda_highpass(eda_signal: ArrayLike, sampling_rate: float) -> pd.DataFrame:

    # Highpass filter for EDA signal decomposition
    phasic = nk.signal_filter(eda_signal, sampling_rate=sampling_rate, lowcut=0.05, method="butter")
    tonic = nk.signal_filter(eda_signal, sampling_rate=sampling_rate, highcut=0.05, method="butter")

    out = pd.DataFrame({"EDA_Tonic": np.array(tonic), "EDA_Phasic": np.array(phasic)})

    return out


def _eda_bandpass(eda_signal: ArrayLike, sampling_rate: float) -> pd.DataFrame:

    # Bandpass filter for EDA signal decomposition
    phasic = nk.signal_filter(eda_signal, sampling_rate, 0.2, 1)
    tonic = nk.signal_filter(eda_signal, sampling_rate, highcut=0.2)

    out = pd.DataFrame({"EDA_Tonic": np.array(tonic), "EDA_Phasic": np.array(phasic)})

    return out


def _cvxEDA(
    y,
    delta,
    tau0=2.0,
    tau1=0.7,
    delta_knot=10.0,
    alpha=8e-4,
    gamma=1e-2,
    solver=None,
    options={"reltol": 1e-9},
):
    """CVXEDA Convex optimization approach to electrodermal activity processing
    This function implements the cvxEDA algorithm described in "cvxEDA: a
    Convex Optimization Approach to Electrodermal Activity Processing"
    (http://dx.doi.org/10.1109/TBME.2015.2474131, also available from the
    authors' homepages).
    Arguments:
       y: observed EDA signal (we recommend normalizing it: y = zscore(y))
       delta: sampling interval (in seconds) of y
       tau0: slow time constant of the Bateman function
       tau1: fast time constant of the Bateman function
       delta_knot: time between knots of the tonic spline function
       alpha: penalization for the sparse SMNA driver
       gamma: penalization for the tonic spline coefficients
       solver: sparse QP solver to be used, see cvxopt.solvers.qp
       options: solver options, see:
                http://cvxopt.org/userguide/coneprog.html#algorithm-parameters
    Returns (see paper for details):
       r: phasic component
       p: sparse SMNA driver of phasic component
       t: tonic component
       l: coefficients of tonic spline
       d: offset and slope of the linear drift term
       e: model residuals
       obj: value of objective function being minimized (eq 15 of paper)
    """

    n = len(y)
    y = cv.matrix(y)

    # bateman ARMA model
    a1 = 1.0 / min(tau1, tau0)  # a1 > a0
    a0 = 1.0 / max(tau1, tau0)
    ar = (
        np.array(
            [
                (a1 * delta + 2.0) * (a0 * delta + 2.0),
                2.0 * a1 * a0 * delta ** 2 - 8.0,
                (a1 * delta - 2.0) * (a0 * delta - 2.0),
            ]
        )
        / ((a1 - a0) * delta ** 2)
    )
    ma = np.array([1.0, 2.0, 1.0])

    # matrices for ARMA model
    i = np.arange(2, n)
    A = cv.spmatrix(np.tile(ar, (n - 2, 1)), np.c_[i, i, i], np.c_[i, i - 1, i - 2], (n, n))
    M = cv.spmatrix(np.tile(ma, (n - 2, 1)), np.c_[i, i, i], np.c_[i, i - 1, i - 2], (n, n))

    # spline
    delta_knot_s = int(round(delta_knot / delta))
    spl = np.r_[np.arange(1.0, delta_knot_s), np.arange(delta_knot_s, 0.0, -1.0)]  # order 1
    spl = np.convolve(spl, spl, "full")
    spl /= max(spl)
    # matrix of spline regressors
    i = np.c_[np.arange(-(len(spl) // 2), (len(spl) + 1) // 2)] + np.r_[np.arange(0, n, delta_knot_s)]
    nB = i.shape[1]
    j = np.tile(np.arange(nB), (len(spl), 1))
    p = np.tile(spl, (nB, 1)).T
    valid = (i >= 0) & (i < n)
    B = cv.spmatrix(p[valid], i[valid], j[valid])

    # trend
    C = cv.matrix(np.c_[np.ones(n), np.arange(1.0, n + 1.0) / n])
    nC = C.size[1]

    # Solve the problem:
    # .5*(M*q + B*l + C*d - y)^2 + alpha*sum(A,1)*p + .5*gamma*l'*l
    # s.t. A*q >= 0

    old_options = cv.solvers.options.copy()
    cv.solvers.options.clear()
    cv.solvers.options.update(options)
    if solver == "conelp":
        # Use conelp
        def z(m, n):
            return cv.spmatrix([], [], [], (m, n))

        G = cv.sparse(
            [
                [-A, z(2, n), M, z(nB + 2, n)],
                [z(n + 2, nC), C, z(nB + 2, nC)],
                [z(n, 1), -1, 1, z(n + nB + 2, 1)],
                [z(2 * n + 2, 1), -1, 1, z(nB, 1)],
                [z(n + 2, nB), B, z(2, nB), cv.spmatrix(1.0, range(nB), range(nB))],
            ]
        )
        h = cv.matrix([z(n, 1), 0.5, 0.5, y, 0.5, 0.5, z(nB, 1)])
        c = cv.matrix([(cv.matrix(alpha, (1, n)) * A).T, z(nC, 1), 1, gamma, z(nB, 1)])
        res = cv.solvers.conelp(c, G, h, dims={"l": n, "q": [n + 2, nB + 2], "s": []})
        obj = res["primal objective"]
    else:
        # Use qp
        Mt, Ct, Bt = M.T, C.T, B.T
        H = cv.sparse(
            [
                [Mt * M, Ct * M, Bt * M],
                [Mt * C, Ct * C, Bt * C],
                [
                    Mt * B,
                    Ct * B,
                    Bt * B + gamma * cv.spmatrix(1.0, range(nB), range(nB)),
                ],
            ]
        )
        f = cv.matrix([(cv.matrix(alpha, (1, n)) * A).T - Mt * y, -(Ct * y), -(Bt * y)])
        res = cv.solvers.qp(
            H,
            f,
            cv.spmatrix(-A.V, A.I, A.J, (n, len(f))),
            cv.matrix(0.0, (n, 1)),
            solver=solver,
        )
        obj = res["primal objective"] + 0.5 * (y.T * y)
    cv.solvers.options.clear()
    cv.solvers.options.update(old_options)

    l = res["x"][-nB:]
    d = res["x"][n : n + nC]
    t = B * l + C * d
    q = res["x"][:n]
    p = A * q
    r = M * q
    e = y - r - t

    return (np.array(a).ravel() for a in (r, p, t, l, d, e, obj))
