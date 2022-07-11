import numpy as np
import npquad
from scipy.special import erf
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

# Going to define latex strings of each equation here to render in matplotlib labels
quantileMeanStr = r"$t\sqrt{1 - (1 - \frac{ln(N)}{t})^{2}}$"
quantileVarShortTimeStr = (
    r"$(2 ln(N))^{\frac{2}{3}} \frac{(t/ln(N) - 1)^\frac{4}{3}}{2t/ln(N) - 1}$"
)
quantileVarLongTimeStr = r"$t^{1/2}\pi^{1/2}/2$"
PbMeanStr = r"$-I * t + t^{1/3}\sigma M$"
PbVarStr = r"$t^{2/3}\sigma^{2} V$"

TW_mean = -1.77
TW_var = 0.813
TW_x2 = TW_var + TW_mean ** 2
TW_mean_sq = TW_var + TW_mean ** 2

EM_constant = 0.577
KPZ_time = np.array([
    25 / 100,
    35 / 100,
    50 / 100,
    75 / 100,
    12 / 10,
    2,
    35 / 10,
    65 / 10,
    13,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
    20000,
])

KPZ_var = np.array([
    1.75713,
    1.70908,
    1.62923,
    1.53633,
    1.4361,
    1.33806,
    1.24323,
    1.15268,
    1.06801,
    1.00314,
    0.948845,
    0.907622,
    0.869637,
    0.850573,
    0.837569,
    0.82681,
    0.821864,
    0.818774,
    0.815189,
])

KPZ_mean = np.array([
 -0.870678,
 -1.16141,
 -1.39703,
 -1.59726,
 -1.76062,
 -1.87532,
 -1.94673,
 -1.98074,
 -1.98402,
 -1.96854,
 -1.94339,
 -1.91601,
 -1.88247,
 -1.86097,
 -1.84308,
 -1.82444,
 -1.81353,
 -1.80477,
 -1.79884,
])


def v0(N, time):
    logN = np.log(N).astype(float)
    return np.sqrt(1 - (1 - logN / time) ** 2)


def I(v):
    return 1 - np.sqrt(1 - v ** 2)


def I_prime(v):
    return v / np.sqrt(1 - v ** 2)


def I_double_prime(v):
    return 1 / (1 - v ** 2) ** (3 / 2)


def sigma(v):
    return (2 * I(v) ** 2 / (1 - I(v))) ** (1 / 3)


def sigma_prime(v):
    I_val = I(v)
    return 2 ** (1 / 3) / 3 * v / (I_val) ** (2 / 3) / np.sqrt(1 - v ** 2)


def lambda_0(v):
    return sigma(v) / I_prime(v)


def lambda_1(v):
    return lambda_0(v) * (
        sigma_prime(v) / I_prime(v)
        - I_double_prime(v) * sigma(v) / (2 * I_prime(v) ** 2)
    )


def first_order_mean(N, time):
    v0_val = v0(N, time)
    return v0_val * time


def second_order_mean(N, time):
    v0_val = v0(N, time)
    return time ** (1 / 3) * TW_mean * lambda_0(v0_val)

def second_order_mean_written_out(N, t):
    logN = np.log(N).astype(float)
    return t**(-1/3) * logN ** (2/3) * 2**(1/3) * (1 - logN/t)**(2/3) * 1.771 / np.sqrt(1-(1-logN/t)**2)


def third_order_mean(N, time):
    v0_val = v0(N, time)
    return time ** (-1 / 3) * lambda_1(v0_val) * TW_mean_sq

def quantileMean(N, time):
    """
    Returns the mean of the 1/Nth quantile. Remember that the predicted position
    is twice the distance we're recording.

    Parameters
    ----------
    N : float or np.quad
        1/Nth quantile to measure. Should be > 1

    time : numpy array
        Times to record the 1/Nth quantile for

    Returns
    -------
    theory : numpy array
        Mean 1/Nth quantile as a function of time predicted by the BC model for
        diffusion.
    """
    logN = np.log(N).astype(np.float64)
    theory = np.piecewise(
        time,
        [time < logN, time >= logN],
        [lambda t: t, lambda t: first_order_mean(N, t) - second_order_mean_written_out(N, t)], #+ second_order_mean(N, t)],
    )
    return theory


def erf_crossover(N, time, crossover, width):
    theory_short = quantileVarShortTime(N, time)
    theory_long = quantileVarLongTime(N, time)
    error_func = (erf((time - crossover) / width) + 1) / 2
    theory = theory_short * (1 - error_func) + theory_long * (error_func)
    return theory


def quantileVar(N, time, crossover=None, width=None):
    """
    Returns the quantile variance over time. Does this by stitching
    together short time and long time with an error function.

    Parameters
    ----------
    N : float or np.quad
        1/Nth quantile to measure. Should be > 1.

    times : numpy array
        Times to record the 1/Nth quantile variance for.

    crossover : float (optional)
        Time when quantile variance shifts from short to long time regime

    width : float (optional)
        Error function width. A shorter width makes the switch from short to long
        time regimes more dramatic.

    Returns
    -------
    theory : numpy array
        Quantile variance
    """

    if crossover is None:
        crossover = np.log(N).astype(float) ** (3/2)
    if width is None:
        width = np.log(N).astype(float) ** (4/3)

    logN = np.log(N).astype(float)
    theory = np.piecewise(
        time,
        [time < logN, time >= logN],
        [lambda t: 0, lambda t: erf_crossover(N, t, crossover, width)],
    )
    return theory


def quantileVarShortTime(N, time):
    """
    Returns the quantile variance over time in the short time regime (t~log(N)).

    Parameters
    ----------
    N : float or np.quad
        1/Nth quantile to measure. Should be > 1.

    times : numpy array
        Times to record the 1/Nth quantile variance for.

    Returns
    -------
    theory : numpy array
        Quantile variance
    """
    v0_val = v0(N, time)
    first_order = time ** (2 / 3) * TW_var * lambda_0(v0_val) ** 2
    second_order = time ** (-2 / 3) * TW_x2 * lambda_1(v0_val) ** 2
    return first_order


def quantileVarLongTime(N, time):
    """
    Returns the quantile variance over time in the long time regime (t~log(N)^2).

    Parameters
    ----------
    N : float or np.quad
        1/Nth quantile to measure. Should be > 1.

    times : numpy array
        Times to record the 1/Nth quantile variance for.

    Returns
    -------
    theory : numpy array
        Quantile variance
    """

    logN = np.log(N).astype(float)
    that = time / logN**2
    return logN * that / 2 * KPZ_var_fit(4/that)

def quantileMeanLongTime(N, time):
    """
    Returns the variance over time in the long time regime
    """

    logN = np.log(N).astype(float)
    that = time / logN**2
    return (2 * time * logN)**(1/2) + np.sqrt(time / 2 / logN) * KPZ_mean_fit(4 / that)

def probMean(vs, t):
    """
    Get the theoretically predicted ln(Pb(vt, t)) (probability greater than
    index vt at current time)

    Parameters
    ----------
    t : int or float
        Time to calculate ln(Pb(vt, t)) at

    v : numpy array
        List of velocities

    Returns
    -------
    lnPbs : numpy array
        Natural log of probabilities greater than vt
    """
    M = -1.77  # Mean of TW distribution for beta=2
    I = 1 - np.sqrt(1 - vs ** 2)
    sigma = ((2 * I ** 2) / (1 - I)) ** (1 / 3)
    return -I * t + t ** (1 / 3) * sigma * M


def probVariance(v, t):
    """
    For a specified v get the variance of the probability of being greater than
    vt over time. Otherwise known as Var(ln(Pb(vt, t)))

    Parameters
    ----------
    t : numpy float
        Times to calculate ln(Pb(vt, t)) for

    v : float
        Velocity to get probability of. Must satisfy 0 < v < 1.

    Returns
    -------
    numpy array
        Variance of logged probability of being greater than vt or Var(ln(Pb(vt, t))).
    """

    V = 0.813
    I = 1 - np.sqrt(1 - v ** 2)
    sigma = ((2 * I ** 2) / (1 - I)) ** (1 / 3)
    theory = (t ** (2 / 3)) * sigma ** 2 * V
    return theory


def nu(x):
    return 1 / 2 * ((1 + x) * np.log(1 + x) + (1 - x) * np.log(1 - x))


def nu_prime(x):
    return 1 / 2 * (np.log(1 + x) - np.log(1 - x))


def beta(x):
    return 1 / nu_prime(x)


def mu(n, c1):
    return (
        n * c1
        - np.log(n) / (2 * nu_prime(c1))
        + 1 / (2 * nu_prime(c1)) * np.log((1 + c1) / (2 * np.pi * (1 - c1)))
    )

def einstein_mean(N, t, c1):
    c = np.log(N) / t
    b = beta(c1)
    m = mu(t, c1)
    return m + b * EM_constant


def einstein_var(N, c1):
    return beta(c1) ** 2 * np.pi ** 2 / 6

def KPZ_var_theory(t):
    return np.sqrt(np.pi/2) * (t/2)**(-1/6) + (1 + 5/4 * np.pi - 8*np.pi/3/np.sqrt(3))*(t/2)**(1/3)

def KPZ_mean_theory(t):
    return -np.sqrt(np.pi / 8) * (t/2)**(1/6) - (1/2 + 3/8 * np.pi - 8 *np.pi / 9 / np.sqrt(3)) * (t/2)**(2/3)

def KPZ_var_fit(t):
    f = interp1d(KPZ_time, KPZ_var, fill_value = KPZ_var[-1], bounds_error=False)
    y = np.piecewise(t,
                    [t <= 0.33, t > 0.33],
                    [lambda time: KPZ_var_theory(time), lambda time: f(time)])
    y = y * 2 **(-2/3) * t ** (2/3)
    return y

def KPZ_mean_fit(t):
    f = interp1d(KPZ_time, KPZ_mean, fill_value = KPZ_mean[-1], bounds_error=False)
    y = np.piecewise(t,
                    [t <= 0.33, t > 0.33],
                    [lambda time: KPZ_mean_theory(time), lambda time: f(time)])
    y = y*2**(-1/3) * t ** (1/3) - t/24
    return y

def gumbel_var(t, N):
    logN = np.log(N).astype(float)
    return np.piecewise(t,
                        [t <= logN, t > logN],
                        [lambda time: 0, lambda time: np.pi**2 / 6 * (time/logN-1)**2 / (2*time/logN-1)])


def log_moving_average(time, data, N, window_size=10):
    assert window_size > 1
    window_min = time[0]
    window_max = window_size * window_min
    new_times = []
    mean_data = []
    while window_min <= max(time):
        window_time = time[(time >= window_min) & (time < window_max)]
        window_data = data[(time >= window_min) & (time < window_max)]

        mean_data.append(np.mean(window_data))
        new_times.append(np.exp(np.mean(np.log(window_time))))

        window_min = window_max
        window_max = window_size * window_min

    return np.array(new_times), np.array(mean_data)

if __name__ == '__main__':
    num = 100
    times = np.geomspace(1, 10**3, num)
    data = np.random.random(len(times)) * times **3
    fig, ax = plt.subplots()
    log_times, mean_data = log_moving_average(times, data, window_size=10)
    ax.scatter(times, data)
    ax.scatter(log_times, mean_data)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time")
    fig.savefig("Test.png")
