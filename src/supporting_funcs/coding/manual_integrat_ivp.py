"""# manual_integrat_ivp
An implementation of the 4th-order Runge-Kutta method.
Useful to allow actions to be taken at each full step of the integration, such as controller runs"""
# from dataclasses import dataclass
# from typing import Callable, Iterable, Sequence

# import numpy as np
raise NotImplementedError("All this was AI generated and has not been tested at all, " \
						  "but looked right from https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html")

# ArrayLike = np.ndarray | Sequence[float] | float

# @dataclass(slots=True)
# class RK4Result:
# 	t: np.ndarray
# 	y: np.ndarray
# 	success: bool = True
# 	message: str = ""
# 	nfev: int = 0


# def _as_state_vector(y0: ArrayLike) -> np.ndarray:
# 	y = np.asarray(y0, dtype=float)
# 	if y.ndim == 0:
# 		return y.reshape(1)
# 	return y.reshape(-1)


# def _rk4_step(
# 	fun: Callable[[float, np.ndarray], np.ndarray],
# 	t: float,
# 	y: np.ndarray,
# 	h: float,
# 	args: tuple,
# ) -> np.ndarray:
# 	k1 = np.asarray(fun(t, y, *args), dtype=float).reshape(-1)
# 	k2 = np.asarray(fun(t + 0.5 * h, y + 0.5 * h * k1, *args), dtype=float).reshape(-1)
# 	k3 = np.asarray(fun(t + 0.5 * h, y + 0.5 * h * k2, *args), dtype=float).reshape(-1)
# 	k4 = np.asarray(fun(t + h, y + h * k3, *args), dtype=float).reshape(-1)
# 	return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# def _build_default_output_times(t0: float, tf: float, step_size: float) -> np.ndarray:
# 	direction = np.sign(tf - t0)
# 	if direction == 0:
# 		return np.array([t0], dtype=float)

# 	step = abs(float(step_size)) * direction
# 	if step == 0:
# 		raise ValueError("step_size must be non-zero when t_span has non-zero length.")

# 	times = [float(t0)]
# 	current = float(t0)
# 	while (tf - current) * direction > 0:
# 		next_time = current + step
# 		if (tf - next_time) * direction < 0:
# 			next_time = float(tf)
# 		times.append(float(next_time))
# 		current = next_time
# 	return np.asarray(times, dtype=float)


# def _validate_output_times(t_eval: Iterable[float], t0: float, tf: float) -> np.ndarray:
# 	times = np.asarray(list(t_eval), dtype=float)
# 	if times.ndim != 1:
# 		raise ValueError("t_eval must be a one-dimensional sequence of times.")
# 	if times.size == 0:
# 		raise ValueError("t_eval must contain at least one time.")

# 	direction = np.sign(tf - t0)
# 	if direction == 0:
# 		if not np.allclose(times, t0):
# 			raise ValueError("t_eval must equal t_span when t_span has zero length.")
# 		return times

# 	if direction > 0:
# 		if np.any(np.diff(times) < 0):
# 			raise ValueError("t_eval must be monotonic increasing for an increasing t_span.")
# 	else:
# 		if np.any(np.diff(times) > 0):
# 			raise ValueError("t_eval must be monotonic decreasing for a decreasing t_span.")

# 	if not np.isclose(times[0], t0):
# 		raise ValueError("t_eval must start at the first value in t_span.")
# 	if not np.isclose(times[-1], tf):
# 		raise ValueError("t_eval must end at the second value in t_span.")
# 	return times


# def solve_ivp_rk4(
# 	fun: Callable[[float, np.ndarray], np.ndarray],
# 	t_span: tuple[float, float],
# 	y0: ArrayLike,
# 	*,
# 	step_size: float | None = None,
# 	max_step: float | None = None,
# 	t_eval: Iterable[float] | None = None,
# 	args: tuple = (),
# ) -> RK4Result:
# 	"""Integrate an initial value problem with the classical 4th-order Runge-Kutta method.

# 	Parameters
# 	----------
# 	fun:
# 		Callable of the form ``fun(t, y, *args)`` returning the derivative ``dy/dt``.
# 	t_span:
# 		Two-element tuple containing the start and end time.
# 	y0:
# 		Initial state, scalar or array-like.
# 	step_size:
# 		Fixed integration step. If omitted, ``max_step`` is used. If both are omitted,
# 		the solver uses 100 uniform steps across ``t_span``.
# 	max_step:
# 		Convenience alias for ``step_size``.
# 	t_eval:
# 		Optional output times. The solver will step internally with the requested
# 		fixed step size and return values at these times.
# 	args:
# 		Extra positional arguments forwarded to ``fun``.
# 	"""

# 	t0, tf = map(float, t_span)
# 	y0_vec = _as_state_vector(y0)

# 	if step_size is None:
# 		step_size = max_step
# 	if step_size is None:
# 		step_size = abs(tf - t0) / 100.0 if tf != t0 else 1.0
# 	if step_size <= 0:
# 		raise ValueError("step_size must be positive.")

# 	if t_eval is None:
# 		output_times = _build_default_output_times(t0, tf, step_size)
# 	else:
# 		output_times = _validate_output_times(t_eval, t0, tf)

# 	direction = np.sign(tf - t0) if tf != t0 else 1.0
# 	internal_step = abs(float(step_size)) * direction

# 	y_out = np.empty((y0_vec.size, output_times.size), dtype=float)
# 	y_current = y0_vec.astype(float, copy=True)
# 	t_current = t0
# 	y_out[:, 0] = y_current
# 	nfev = 0

# 	for index in range(1, output_times.size):
# 		t_target = float(output_times[index])
# 		while (t_target - t_current) * direction > 0:
# 			step = internal_step
# 			if (t_target - (t_current + step)) * direction < 0:
# 				step = t_target - t_current
# 			y_current = _rk4_step(fun, t_current, y_current, step, args)
# 			t_current += step
# 			nfev += 4
# 		y_out[:, index] = y_current
#
# 	return RK4Result(t=output_times,
# 				  y=y_out,
# 				  success=True,
# 				  message="RK4 integration completed.",
# 				  nfev=nfev)
