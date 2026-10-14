"""A module for calculating various rocket nozzle properties and curves.

Functions
---------
get_nozzle_curve: Calculate the radius of the nozzle curve at each x value.
plot_nozzle_curve: Plot the nozzle curve.
"""
from enum import IntEnum, auto
from typing import overload

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as csp
from scipy.optimize import root

from supporting_funcs.coding.type_hints import BoolArray, FloatArray, MatPlotLibFigAxes
from supporting_funcs.pretty.matplotlib_tools import format_plot, plot_circle


def get_nozzle_curve(l_super: float,
                     l_sub: float,
                     r_throat: float,
                     r_conv_curve: float,
                     r_div_curve: float,
                     r_exit: float,
                     theta_n: float,
                     theta_e: float,
                     *,
                     n_pts: int = 10_000) -> FloatArray:
    """Calculate the radius of the nozzle curve at each x value.

    The nozzle curve is defined as a combination of
    a converging curve, a diverging curve, and two circular arcs.
    https://rrs.org/2023/01/28/making-correct-parabolic-nozzles/

    Parameters
    ----------
    l_super: [m]
        Length of the nozzle from the throat to the exit.
    l_sub: [m]
        Length of the nozzle from the subsonic start to the throat.
    r_throat: [m]
        Radius of the nozzle at the throat.
    r_conv_curve: [m]
        Radius of the converging curve of the nozzle (near the throat).
    r_div_curve: [m]
        Radius of the diverging curve of the nozzle (near the throat).
    r_exit: [m]
        Radius of the nozzle at the exit.
    theta_n: [rad]
        Angle of the nozzle at the start of the diverging curve.
    theta_e: [rad]
        Angle of the nozzle at the exit.
    n_pts: int, default=10_000
        Number of points to use for computation.

    Returns
    -------
        np.ndarray: array of corresponding radius values.
    """
    class xE(IntEnum): #noqa: N801
        m = 0
        psi = auto()
        t_n = auto()
        t_e = auto()
        h = auto()
        k = auto()

    def _compute_diverging_curve_params(x_parabola_start: float,
                                        r_parabola_start: float,
                                        x_exit: float,
                                        r_exit: float,
                                        theta_n: float,
                                        theta_e: float) -> np.ndarray:
        """Compute the parameters of the diverging curve of the nozzle.

        Parameters
        ----------
        x_parabola_start : [Length]
            x value at the start of the diverging curve
        r_parabola_start : [Length]
            r value at the start of the diverging curve
        x_exit : [Length]
            x value at the end of the diverging curve
        r_exit : [Length]
            r value at the end of the diverging curve
        theta_n : [rad]
            angle of the diverging curve at the start
        theta_e : [rad]
            angle of the diverging curve at the end

        Returns
        -------
        np.ndarray
            array of parameters [m, psi, t_n, t_e, h, k]
        """

        def _error_func(x: np.ndarray) -> np.ndarray:
            """Compute the error between the desired and actual values of the diverging curve."""
            nonlocal x_parabola_start, r_parabola_start, x_exit, r_exit, theta_n, theta_e, xE

            def x_func(t: float) -> float:
                nonlocal r_parabola_start, x, xE
                return t * (-x[xE.m] * np.sin(x[xE.psi]) * t + np.cos(x[xE.psi])) + x[xE.h]
            def r_func(t: float) -> float:
                nonlocal x_parabola_start, x, xE
                return t * (x[xE.m] * np.cos(x[xE.psi]) * t + np.sin(x[xE.psi])) + x[xE.k]
            def drdx(t: float) -> float:
                nonlocal x, xE
                return ((-2 * x[xE.m] * np.cos(x[xE.psi]) * t - np.sin(x[xE.psi]))
                        / (2 * x[xE.m] * np.sin(x[xE.psi]) * t - np.cos(x[xE.psi])))
            nonlocal x_parabola_start, r_parabola_start, x_exit, r_exit, theta_n, theta_e
            error = np.full_like(x, np.nan)
            error[0] = x_parabola_start - x_func(t=x[xE.t_n])
            error[1] = r_parabola_start - r_func(t=x[xE.t_n])
            error[2] = x_exit - x_func(t=x[xE.t_e])
            error[3] = r_exit - r_func(t=x[xE.t_e])
            error[4] = np.tan(theta_n) - drdx(t=x[xE.t_n]) # Is the start slope correct
            error[5] = np.tan(theta_e) - drdx(t=x[xE.t_e]) # Is the exit slope correct
            return error

        x0 = np.empty(len(xE))
        # Initial guess for the parameters, based on an open cross section of Raptor.
        x0[xE.m] = -9.049377
        x0[xE.psi] = -0.835233
        x0[xE.h] = 0.861202
        x0[xE.k] = 0.424397
        x0[xE.t_n] = -0.308790
        x0[xE.t_e] = -0.081470

        sol = root(_error_func, x0, method="hybr")
        return sol.x

    def _compute_diverging_curve(nozzle_params: np.ndarray, t: float | np.ndarray) -> np.ndarray:
        """Compute the diverging curve of the nozzle given the parameters and t values.

        The nozzle curve is defined as:
        x = t * (-m * sin(psi) * t + cos(psi)) + h
        r = t * (m * cos(psi) * t + sin(psi)) + k
        where m, psi, h, k are the parameters to be solved for.
        The curve is only valid for t_n <= t <= t_e,
        where t_n and t_e are also parameters to be solved for.

        Parameters
        ----------
        nozzle_params : (np.ndarray):
            array of parameters [m, psi, t_n, t_e, h, k]
        t : (float | np.ndarray): t values for which to compute the curve

        Returns
        -------
        curve_array: [x, r]
            array of shape (len(t), 2) containing the x and r values of the curve at each t value.
            Values are set to NaN for t values outside of the valid range.
        """
        m = nozzle_params[xE.m]
        psi = nozzle_params[xE.psi]
        t_n = nozzle_params[xE.t_n]
        t_e = nozzle_params[xE.t_e]
        h = nozzle_params[xE.h]
        k = nozzle_params[xE.k]

        invalid_t = (t < t_n) + (t > t_e)

        x = t * (-m * np.sin(psi) * t + np.cos(psi)) + h
        r = t * (m * np.cos(psi) * t + np.sin(psi)) + k

        return_arr = np.c_[x, r]
        return_arr[invalid_t] = np.array([np.nan, np.nan])
        return return_arr

    x = np.linspace(-l_sub, l_super, n_pts)
    r = np.full_like(x, np.nan)

    theta_start = 45 * csp.degree
    conv_center = np.array([0, r_throat + r_conv_curve]) # the center of the converging arc

    conv_join = np.full_like(conv_center, np.nan)
    conv_join[0] = np.sin(- theta_start) * r_conv_curve + conv_center[0]
    conv_join[1] =-np.cos(- theta_start) * r_conv_curve + conv_center[1]

    def r_conv_circle_func(x: float | np.ndarray) -> float | np.ndarray:
        return -np.sqrt(r_conv_curve ** 2 - (x - conv_center[0])**2) + conv_center[1]

    def r_conv_curve_func(x: float | np.ndarray) -> float | np.ndarray:
        return (2 / (1 + np.exp(2 * (x + x_offset))) + y_offset)

    @overload
    def r_div_circle_func(x: float) -> float: ...
    @overload
    def r_div_circle_func(x: np.ndarray) -> np.ndarray: ...
    def r_div_circle_func(x: float | np.ndarray) -> float | np.ndarray:
        return -np.sqrt(r_div_curve ** 2 - (x - 0)**2) + (r_throat + r_div_curve)

    def get_conv_curve_params(theta_start: float, conv_join_loc: np.ndarray) -> tuple[float, float]:
        """Get the parameters for the converging curve of the nozzle.

        Parameters
        ----------
        theta_start : [rad]
            angle of the converging curve at the start
        conv_join_loc: [m, m]
            array containing the x and r values of the converging circle join point
        """
        # R(x) = (2 / (1 + np.exp(2 * (x + X_Offset))) + Y_Offset
        # dRdx = - (4 * np.exp(2 * (x + X_Offset))) / ((1 + np.exp(2 * (x + X_Offset))) ** 2)
        # dRdx |_(x = Conv_join[0]) = -np.tan(thetaStart)
        # Solving for X_Offset and Y_Offset gives:
        x_c = conv_join_loc[0]
        y_c = conv_join_loc[1]
        k = np.tan(theta_start) / 4
        u = ((1 - 2*k) + np.sqrt(1 - 4 * k)) / (2 * k)
        x_offset = (np.log(u) - 2* x_c) / 2
        y_offset = y_c - (2 / (1 + u))
        return x_offset, y_offset
    x_offset, y_offset = get_conv_curve_params(theta_start = theta_start, conv_join_loc = conv_join)

    x_parabola_start: float = r_div_curve * np.sin(theta_n)
    r_parabola_start: float = r_div_circle_func(x_parabola_start)

    curve_params = _compute_diverging_curve_params(x_parabola_start=x_parabola_start,
                                                   r_parabola_start=r_parabola_start,
                                                   x_exit=l_super,
                                                   r_exit=r_exit,
                                                   theta_n=theta_n,
                                                   theta_e=theta_e)
    diverg_arr = _compute_diverging_curve(curve_params, t=np.linspace(curve_params[xE.t_n],
                                                                      curve_params[xE.t_e], n_pts))

    x_mask_conv_r_eqn = (x >= -l_sub) * (x < conv_join[0])
    x_mask_conv_circ = (x >= conv_join[0]) * (x < 0)
    x_mask_div_circ = (x >= 0) * (x < x_parabola_start)
    x_mask_div_parabola = (x >= x_parabola_start) * (x <= l_super)

    r[x_mask_conv_r_eqn] = r_conv_curve_func(x[x_mask_conv_r_eqn])
    r[x_mask_conv_circ] = r_conv_circle_func(x[x_mask_conv_circ])
    r[x_mask_div_circ] = r_div_circle_func(x[x_mask_div_circ])
    r[x_mask_div_parabola] = np.interp(x[x_mask_div_parabola], diverg_arr[:, 0], diverg_arr[:, 1])

    return np.c_[x, r]

def plot_nozzle_curve(self,
                            x: FloatArray,
                            r: FloatArray,
                            conv_join: FloatArray,
                            x_n: float,
                            x_mask_conv_r_eqn: BoolArray,
                            x_mask_conv_circ: BoolArray,
                            x_mask_div_circ: BoolArray,
                            x_mask_div_parabola: BoolArray,
                            ) -> MatPlotLibFigAxes:
    """Plot the nozzle curve for debugging purposes.

    Parameters
    ----------
    x: [m]
        Array of x values.
    r: [m]
        Array of corresponding radius values.
    conv_join: np.ndarray
        Array containing the x and r values of the converging circle join point.
    x_n: float
        x value of the diverging circle join point.
    """
    fig_dbg = plt.figure(figsize=(12, 6))
    gs = fig_dbg.add_gridspec(4, 1)
    ax_dbg = np.empty(2, dtype=object)
    ax_dbg[0] = fig_dbg.add_subplot(gs[0:3, 0])
    ax_dbg[1] = fig_dbg.add_subplot(gs[3, 0], sharex=ax_dbg[0])

    ax_dbg[0].axvline(-self.L_sub, color="#00B3FF", linestyle="--", label="Subsonic Start")
    ax_dbg[0].axvline(conv_join[0], color="k", linestyle="--", label="Conv Join")
    ax_dbg[0].axvline(0, color="r", linestyle="--", label="Throat")
    ax_dbg[0].axvline(x_n, color="b", linestyle="--", label="Div Circle Join")
    ax_dbg[0].axvline(self.L_super, color="g", linestyle="--", label="Nozzle End")
    ax_dbg[0].set_aspect("equal", adjustable="datalim")
    ax_dbg[0].set_ylim(0, self.r_exit * 1.2)
    ax_dbg[0].set_ylabel("r (inches)")

    ax_dbg[0].plot(x[x_mask_conv_r_eqn], r[x_mask_conv_r_eqn],
                   label="Conv Solver Func", color="k")
    ax_dbg[0].plot(x[x_mask_conv_circ], r[x_mask_conv_circ],
                   label="Conv Circle Func", color="r")
    plot_circle(ax_dbg[0], center=(0, (self.r_th + self.R_conv)), radius=self.R_conv,
                color="#FF880088", linestyle="--", label="Conv Circle")
    ax_dbg[0].plot(x[x_mask_div_circ], r[x_mask_div_circ],
                   label="Div Circle Func", color="b")
    plot_circle(ax_dbg[0], center=(0, (self.r_th + self.R_div)), radius=self.R_div,
                color="#0088FF88", linestyle="--", label="Div Circle")
    ax_dbg[0].plot(x[x_mask_div_parabola], r[x_mask_div_parabola],
                   label="Div Parabola Func", color="g")
    ax_dbg[0].legend()
    ax_dbg[0].set_title("Nozzle Curve Debugging")
    ax_dbg[0].set_aspect("equal", adjustable="datalim")
    ax_dbg[1].plot(x, np.gradient(r, x), label="dr/dx", color="m")
    ax_dbg[1].set_ylabel("dr/dx (unitless)")
    for ax in ax_dbg:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.minorticks_on()
        ax.tick_params(which='major', length=7 / 2, width=1.5 / 2, labelsize=12 * 0.8)
        ax.tick_params(which='minor', length=4 / 2, width=1 / 2, labelsize=12 * 0.8)
    ax_dbg[-1].set_xlabel("x (inches)")
    fig_dbg.tight_layout()

    fig_dbg, ax_dbg = format_plot(fig=fig_dbg, ax=ax_dbg, aspect_ratio = 3/5)
    fig_dbg.show()
    return fig_dbg, ax_dbg
