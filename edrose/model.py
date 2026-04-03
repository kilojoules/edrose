"""Core elliptical wind direction rose model.

Implements the generalised elliptical wind direction rose from:
    Hart, E. (2025). An elliptical parameterisation of the wind direction rose.
    Wind Energy Science, 10, 1821-1827.

The model parameterises a wind direction rose using ellipse geometry with
three parameters:
    a          - ellipse shape (a = 1/sqrt(pi) gives uniform distribution)
    f          - folding parameter in [0, 1] (0 = bi-directional, 1 = fully uni-directional)
    theta_prev - prevailing wind direction in degrees
"""

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Low-level math (module-private)
# ---------------------------------------------------------------------------

def _elliptical_cdf(theta_rad, a):
    """CDF of the base (un-folded, un-rotated) elliptical direction distribution.

    For a unit-area ellipse with semi-major axis *a* (along x) and
    semi-minor axis b = 1/(pi*a) (along y), the CDF at angle theta gives
    the probability that a direction falls in [0, theta].

    Parameters
    ----------
    theta_rad : array_like
        Angles in radians (any value; mapped internally to [0, 2pi)).
    a : float
        Ellipse shape parameter (a > 0).

    Returns
    -------
    numpy.ndarray
        CDF values in [0, 1].
    """
    theta = np.asarray(theta_rad, dtype=float)
    scalar_input = theta.ndim == 0
    theta = np.atleast_1d(theta)

    theta_mod = theta % (2 * np.pi)

    # Quadrant index: 0=[0,pi/2), 1=[pi/2,pi), 2=[pi,3pi/2), 3=[3pi/2,2pi)
    # clip to 3 so that theta_mod ≈ 2pi doesn't round up to quadrant 4
    quadrant = np.clip(np.floor(theta_mod / (np.pi / 2)).astype(int), 0, 3)

    # Map to equivalent first-quadrant angle
    q1 = np.where(
        quadrant == 0, theta_mod,
        np.where(
            quadrant == 1, np.pi - theta_mod,
            np.where(quadrant == 2, theta_mod - np.pi, 2 * np.pi - theta_mod),
        ),
    )
    q1 = np.clip(q1, 0.0, np.pi / 2)

    # F_q1(theta) = arctan(pi * a^2 * tan(theta)) / (2*pi)       [Eq. 2]
    # At theta = pi/2 the limit is 1/4.
    at_boundary = q1 >= np.pi / 2 - 1e-14
    safe_q1 = np.where(at_boundary, 0.0, q1)
    f_q1 = np.where(
        at_boundary,
        0.25,
        np.arctan(np.pi * a ** 2 * np.tan(safe_q1)) / (2 * np.pi),
    )

    # Assemble full CDF from 4-fold symmetry
    cdf = np.where(
        quadrant == 0, f_q1,
        np.where(
            quadrant == 1, 0.5 - f_q1,
            np.where(quadrant == 2, 0.5 + f_q1, 1.0 - f_q1),
        ),
    )

    return float(cdf) if scalar_input else cdf


def _elliptical_cdf_deriv_a(theta_rad, a):
    """Derivative of the elliptical CDF with respect to *a*.

    d/da F_q1 = a * tan(theta) / (1 + pi^2 * a^4 * tan^2(theta))
    """
    theta = np.asarray(theta_rad, dtype=float)
    theta = np.atleast_1d(theta)

    theta_mod = theta % (2 * np.pi)
    quadrant = np.clip(np.floor(theta_mod / (np.pi / 2)).astype(int), 0, 3)

    q1 = np.where(
        quadrant == 0, theta_mod,
        np.where(
            quadrant == 1, np.pi - theta_mod,
            np.where(quadrant == 2, theta_mod - np.pi, 2 * np.pi - theta_mod),
        ),
    )
    q1 = np.clip(q1, 0.0, np.pi / 2)

    at_boundary = q1 >= np.pi / 2 - 1e-14
    safe_q1 = np.where(at_boundary, 0.0, q1)
    tan_q1 = np.tan(safe_q1)
    g_prime = np.where(
        at_boundary,
        0.0,
        a * tan_q1 / (1 + np.pi ** 2 * a ** 4 * tan_q1 ** 2),
    )

    # Sign: +1 for Q0/Q2, -1 for Q1/Q3
    sign = np.where((quadrant == 0) | (quadrant == 2), 1.0, -1.0)
    return sign * g_prime


def _compute_frequencies(a, f, theta_prev_deg, wd_deg, n_sectors):
    """Compute sector frequencies for the generalised elliptical wind rose.

    Implements Eqs. 3-5 of Hart (2025).

    Parameters
    ----------
    a : float
        Ellipse shape parameter.
    f : float
        Folding parameter in [0, 1].
    theta_prev_deg : float
        Prevailing wind direction in degrees.
    wd_deg : array_like
        Sector centre directions in degrees.
    n_sectors : int
        Number of sectors.

    Returns
    -------
    numpy.ndarray
        Sector frequency probabilities (sum to 1).
    """
    wd_deg = np.asarray(wd_deg, dtype=float)
    delta_rad = np.pi / n_sectors  # half-bin width
    theta_prev_rad = np.deg2rad(theta_prev_deg)
    wd_rad = np.deg2rad(wd_deg)

    # Rotate: evaluate at (theta - theta_prev)  [Eq. 5]
    rotated = wd_rad - theta_prev_rad
    upper = rotated + delta_rad
    lower = rotated - delta_rad

    # Periodic CDF: CDF_periodic(x) = CDF(x mod 2pi) + floor(x / 2pi)
    cdf_upper = _elliptical_cdf(upper, a) + np.floor(upper / (2 * np.pi))
    cdf_lower = _elliptical_cdf(lower, a) + np.floor(lower / (2 * np.pi))
    probs = cdf_upper - cdf_lower

    # Apply folding  [Eq. 4]
    rotated_mod = rotated % (2 * np.pi)
    tol = 1e-10
    is_back = (rotated_mod > np.pi / 2 + tol) & (rotated_mod < 3 * np.pi / 2 - tol)
    is_boundary = (np.abs(rotated_mod - np.pi / 2) < tol) | (
        np.abs(rotated_mod - 3 * np.pi / 2) < tol
    )
    fold_mult = np.where(is_back, 1.0 - f, np.where(is_boundary, 1.0, 1.0 + f))
    probs = probs * fold_mult

    return probs


def _compute_frequencies_and_grad(a, f, phi_f, theta_prev_deg, wd_deg, n_sectors, measured):
    """Compute SSE and its gradient w.r.t. (a, phi_f).

    Returns (sse, grad_a, grad_phi_f).
    """
    wd_deg = np.asarray(wd_deg, dtype=float)
    measured = np.asarray(measured, dtype=float)
    delta_rad = np.pi / n_sectors
    theta_prev_rad = np.deg2rad(theta_prev_deg)
    wd_rad = np.deg2rad(wd_deg)

    rotated = wd_rad - theta_prev_rad
    upper = rotated + delta_rad
    lower = rotated - delta_rad

    # --- Forward pass ---
    cdf_upper = _elliptical_cdf(upper, a) + np.floor(upper / (2 * np.pi))
    cdf_lower = _elliptical_cdf(lower, a) + np.floor(lower / (2 * np.pi))
    p_el = cdf_upper - cdf_lower  # un-folded sector probs

    rotated_mod = rotated % (2 * np.pi)
    tol = 1e-10
    is_back = (rotated_mod > np.pi / 2 + tol) & (rotated_mod < 3 * np.pi / 2 - tol)
    is_boundary = (np.abs(rotated_mod - np.pi / 2) < tol) | (
        np.abs(rotated_mod - 3 * np.pi / 2) < tol
    )
    fold_mult = np.where(is_back, 1.0 - f, np.where(is_boundary, 1.0, 1.0 + f))

    probs = p_el * fold_mult
    residuals = probs - measured
    sse = np.sum(residuals ** 2)

    # --- Gradient w.r.t. a ---
    dcdf_upper_da = _elliptical_cdf_deriv_a(upper, a)
    dcdf_lower_da = _elliptical_cdf_deriv_a(lower, a)
    dp_el_da = dcdf_upper_da - dcdf_lower_da
    dp_da = fold_mult * dp_el_da
    grad_a = 2 * np.sum(residuals * dp_da)

    # --- Gradient w.r.t. phi_f ---
    # df/dphi = f * (1 - f)  (sigmoid derivative)
    df_dphi = f * (1 - f)
    # d(fold_mult)/df = -1 (back), +1 (front), 0 (boundary)
    dfold_df = np.where(is_back, -1.0, np.where(is_boundary, 0.0, 1.0))
    dp_dphi = p_el * dfold_df * df_dphi
    grad_phi = 2 * np.sum(residuals * dp_dphi)

    return sse, grad_a, grad_phi


def _fit_single(measured_freq, wd_deg, n_sectors, theta_prev):
    """Fit the model for a specific theta_prev using L-BFGS-B with analytic gradients."""
    measured_freq = np.asarray(measured_freq, dtype=float)

    def objective_and_grad(params):
        a_raw, phi_f = params
        a = np.abs(a_raw) + 1e-12
        f = 1.0 / (1.0 + np.exp(-np.clip(phi_f, -30, 30)))

        sse, g_a, g_phi = _compute_frequencies_and_grad(
            a, f, phi_f, theta_prev, wd_deg, n_sectors, measured_freq
        )
        # Chain rule for |a_raw|
        g_a_raw = g_a * np.sign(a_raw) if a_raw != 0 else g_a
        return sse, np.array([g_a_raw, g_phi])

    best = None
    for a0 in [0.3, 1 / np.sqrt(np.pi), 1.0, 2.0]:
        for phi0 in [-3.0, 0.0, 3.0]:
            try:
                result = minimize(
                    objective_and_grad,
                    [a0, phi0],
                    jac=True,
                    method="L-BFGS-B",
                    options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10},
                )
                if best is None or result.fun < best.fun:
                    best = result
            except Exception:
                continue

    if best is None:
        raise RuntimeError(f"Optimisation failed for theta_prev={theta_prev:.1f}")

    a_fit = np.abs(best.x[0]) + 1e-12
    f_fit = 1.0 / (1.0 + np.exp(-np.clip(best.x[1], -30, 30)))

    return {"a": a_fit, "f": f_fit, "theta_prev": theta_prev, "sse": best.fun}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class EllipticalWindRose:
    """Generalised elliptical wind direction rose.

    A parametric wind direction rose based on ellipse geometry, defined by
    three parameters:

    Parameters
    ----------
    a : float
        Ellipse shape parameter (a > 0).
        ``a = 1/sqrt(pi)`` gives a uniform (circular) distribution.
        Larger *a* concentrates probability along the prevailing direction axis.
    f : float, optional
        Folding parameter in [0, 1].  ``f = 0`` gives a symmetric
        (bi-directional) rose; ``f = 1`` folds all back-half probability onto
        the prevailing half.  Default 0.
    theta_prev : float, optional
        Prevailing wind direction in degrees (meteorological convention).
        Default 0.
    n_sectors : int, optional
        Number of equal-width direction sectors.  Default 12.

    Examples
    --------
    >>> wr = EllipticalWindRose(a=0.8, f=0.3, theta_prev=210, n_sectors=12)
    >>> wr.sector_frequencies.sum()
    1.0
    >>> wr.wind_directions
    array([  0.,  30.,  60.,  90., 120., 150., 180., 210., 240., 270., 300., 330.])
    """

    def __init__(self, a, f=0.0, theta_prev=0.0, n_sectors=12):
        if a <= 0:
            raise ValueError("a must be positive")
        if not 0 <= f <= 1:
            raise ValueError("f must be in [0, 1]")
        self.a = float(a)
        self.f = float(f)
        self.theta_prev = float(theta_prev % 360)
        self.n_sectors = int(n_sectors)
        self._fit_result = None

    # -- properties ---------------------------------------------------------

    @property
    def wind_directions(self):
        """Sector centre directions in degrees, shape ``(n_sectors,)``."""
        return np.linspace(0, 360, self.n_sectors, endpoint=False)

    @property
    def sector_width(self):
        """Full width of each sector in degrees."""
        return 360.0 / self.n_sectors

    @property
    def sector_frequencies(self):
        """Sector frequency probabilities, shape ``(n_sectors,)``.

        These sum to 1 and represent the probability of the wind blowing from
        each sector.  Directly usable as PyWake ``Sector_frequency``.
        """
        return _compute_frequencies(
            self.a, self.f, self.theta_prev, self.wind_directions, self.n_sectors
        )

    @property
    def r_squared(self):
        """R-squared from the last ``fit()`` call, or None."""
        return self._fit_result.get("r_squared") if self._fit_result else None

    @property
    def rmse(self):
        """RMSE from the last ``fit()`` call, or None."""
        return self._fit_result.get("rmse") if self._fit_result else None

    # -- fitting ------------------------------------------------------------

    @classmethod
    def fit(cls, measured_freq, wind_directions=None, theta_prev=None):
        """Fit an elliptical wind rose to measured sector frequencies.

        Parameters
        ----------
        measured_freq : array_like
            Measured sector frequency probabilities (should sum to ~1).
        wind_directions : array_like, optional
            Sector centre directions in degrees.  If *None*, sectors are
            evenly spaced starting at 0.
        theta_prev : float or None, optional
            Prevailing direction in degrees.  If *None*, both the circular
            mean and the mode direction are tried, and the fit with the
            lowest SSE is kept (following the heuristic in Hart, 2025).

        Returns
        -------
        EllipticalWindRose
            Fitted model with ``r_squared`` and ``rmse`` populated.
        """
        measured_freq = np.asarray(measured_freq, dtype=float)
        n_sectors = len(measured_freq)
        if wind_directions is None:
            wind_directions = np.linspace(0, 360, n_sectors, endpoint=False)
        wind_directions = np.asarray(wind_directions, dtype=float)

        # Determine theta_prev candidates
        if theta_prev is not None:
            candidates = [float(theta_prev)]
        else:
            wd_rad = np.deg2rad(wind_directions)
            circ_mean = float(
                np.rad2deg(
                    np.arctan2(
                        np.sum(measured_freq * np.sin(wd_rad)),
                        np.sum(measured_freq * np.cos(wd_rad)),
                    )
                )
                % 360
            )
            mode_dir = float(wind_directions[np.argmax(measured_freq)])
            candidates = [circ_mean, mode_dir]
            # Deduplicate
            if np.abs(circ_mean - mode_dir) < 1e-6:
                candidates = [circ_mean]

        best_result = None
        best_sse = np.inf
        for tp in candidates:
            result = _fit_single(measured_freq, wind_directions, n_sectors, tp)
            if result["sse"] < best_sse:
                best_sse = result["sse"]
                best_result = result

        wr = cls(
            a=best_result["a"],
            f=best_result["f"],
            theta_prev=best_result["theta_prev"],
            n_sectors=n_sectors,
        )

        # Goodness of fit
        gof = wr.goodness_of_fit(measured_freq)
        best_result.update(gof)
        wr._fit_result = best_result
        return wr

    # -- goodness of fit ----------------------------------------------------

    def goodness_of_fit(self, measured_freq):
        """Compute R-squared and RMSE against measured frequencies.

        Parameters
        ----------
        measured_freq : array_like
            Measured sector frequency probabilities.

        Returns
        -------
        dict
            ``{'r_squared': float, 'rmse': float}``
        """
        pred = self.sector_frequencies
        measured_freq = np.asarray(measured_freq, dtype=float)
        ss_res = np.sum((pred - measured_freq) ** 2)
        ss_tot = np.sum((measured_freq - np.mean(measured_freq)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean((pred - measured_freq) ** 2)))
        return {"r_squared": r2, "rmse": rmse}

    # -- PyWake integration -------------------------------------------------

    def to_pywake_site(self, A, k, ti=0.05):
        """Create a PyWake-compatible site object.

        Parameters
        ----------
        A : float or array_like
            Weibull scale parameter (m/s).  Scalar (uniform across sectors)
            or array of length ``n_sectors``.
        k : float or array_like
            Weibull shape parameter.  Scalar or per-sector array.
        ti : float, optional
            Turbulence intensity.  Default 0.05.

        Returns
        -------
        py_wake.site.XRSite
        """
        try:
            from py_wake.site import XRSite
            import xarray as xr
        except ImportError:
            raise ImportError(
                "PyWake is required for to_pywake_site(). "
                "Install with: pip install py_wake"
            )

        wd = self.wind_directions
        freq = self.sector_frequencies
        A_arr = np.broadcast_to(np.asarray(A, dtype=float), (self.n_sectors,)).copy()
        k_arr = np.broadcast_to(np.asarray(k, dtype=float), (self.n_sectors,)).copy()

        ds = xr.Dataset(
            data_vars={
                "Sector_frequency": ("wd", freq),
                "Weibull_A": ("wd", A_arr),
                "Weibull_k": ("wd", k_arr),
                "TI": ti,
            },
            coords={"wd": wd},
        )
        return XRSite(ds)

    def pywake_arrays(self, A=None, k=None):
        """Return raw arrays suitable for PyWake constructors.

        Parameters
        ----------
        A : float or array_like, optional
            Weibull scale parameter.  If *None*, not included.
        k : float or array_like, optional
            Weibull shape parameter.  If *None*, not included.

        Returns
        -------
        dict
            Always contains ``'wd'`` and ``'freq'``.  Optionally ``'A'``
            and ``'k'`` if provided.
        """
        out = {
            "wd": self.wind_directions,
            "freq": self.sector_frequencies,
        }
        if A is not None:
            out["A"] = np.broadcast_to(
                np.asarray(A, dtype=float), (self.n_sectors,)
            ).copy()
        if k is not None:
            out["k"] = np.broadcast_to(
                np.asarray(k, dtype=float), (self.n_sectors,)
            ).copy()
        return out

    # -- plotting -----------------------------------------------------------

    def plot(self, measured_freq=None, ax=None, show=True):
        """Plot the wind rose on a polar axis.

        Parameters
        ----------
        measured_freq : array_like, optional
            Measured sector frequencies to overlay.
        ax : matplotlib Axes, optional
            Existing polar axes.  Created if *None*.
        show : bool, optional
            Call ``plt.show()`` at the end.  Default *True*.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})

        wd_rad = np.deg2rad(self.wind_directions)
        width = np.deg2rad(self.sector_width)
        freqs = self.sector_frequencies

        if measured_freq is not None:
            measured_freq = np.asarray(measured_freq)
            ax.bar(
                wd_rad,
                measured_freq,
                width=width * 0.9,
                alpha=0.5,
                label="Measured",
                color="steelblue",
            )

        ax.bar(
            wd_rad,
            freqs,
            width=width * 0.5,
            alpha=0.7,
            label="Elliptical model",
            color="crimson",
        )

        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

        title_parts = [f"a={self.a:.2f}, f={self.f:.2f}"]
        title_parts.append(f"$\\theta_{{prev}}$={self.theta_prev:.0f}°")
        if self._fit_result:
            r2 = self._fit_result.get("r_squared")
            rmse_val = self._fit_result.get("rmse")
            if r2 is not None:
                title_parts.append(f"$R^2$={r2:.3f}")
            if rmse_val is not None:
                title_parts.append(f"RMSE={rmse_val:.4f}")
        ax.set_title(", ".join(title_parts), fontsize=9, pad=20)

        if show:
            plt.show()
        return ax

    # -- dunder -------------------------------------------------------------

    def __repr__(self):
        return (
            f"EllipticalWindRose(a={self.a:.4f}, f={self.f:.4f}, "
            f"theta_prev={self.theta_prev:.1f}, n_sectors={self.n_sectors})"
        )


class MixtureEllipticalWindRose:
    """Mixture of generalised elliptical wind direction roses.

    Linearly combines multiple :class:`EllipticalWindRose` components with
    specified weights to represent multi-modal direction distributions
    (see Hart 2025, Fig. 4).

    Parameters
    ----------
    components : list of EllipticalWindRose
        Component wind roses.  All must share the same ``n_sectors``.
    weights : array_like
        Non-negative mixing weights (will be normalised to sum to 1).

    Examples
    --------
    >>> c1 = EllipticalWindRose(a=0.8, f=0.1, theta_prev=180, n_sectors=36)
    >>> c2 = EllipticalWindRose(a=1.1, f=1.0, theta_prev=260.5, n_sectors=36)
    >>> mix = MixtureEllipticalWindRose([c1, c2], weights=[0.85, 0.15])
    >>> mix.sector_frequencies.sum()
    1.0
    """

    def __init__(self, components, weights):
        if len(components) != len(weights):
            raise ValueError("Number of components must match number of weights")
        if len(components) == 0:
            raise ValueError("At least one component is required")

        n = components[0].n_sectors
        if not all(c.n_sectors == n for c in components):
            raise ValueError("All components must have the same n_sectors")

        self.components = list(components)
        weights = np.asarray(weights, dtype=float)
        self.weights = weights / weights.sum()
        self.n_sectors = n

    @property
    def wind_directions(self):
        """Sector centre directions in degrees."""
        return self.components[0].wind_directions

    @property
    def sector_width(self):
        """Full width of each sector in degrees."""
        return 360.0 / self.n_sectors

    @property
    def sector_frequencies(self):
        """Weighted mixture of component sector frequencies."""
        return sum(
            w * c.sector_frequencies for w, c in zip(self.weights, self.components)
        )

    def to_pywake_site(self, A, k, ti=0.05):
        """Create a PyWake-compatible site from the mixture distribution.

        Parameters are identical to :meth:`EllipticalWindRose.to_pywake_site`.
        """
        try:
            from py_wake.site import XRSite
            import xarray as xr
        except ImportError:
            raise ImportError(
                "PyWake is required for to_pywake_site(). "
                "Install with: pip install py_wake"
            )

        wd = self.wind_directions
        freq = self.sector_frequencies
        A_arr = np.broadcast_to(np.asarray(A, dtype=float), (self.n_sectors,)).copy()
        k_arr = np.broadcast_to(np.asarray(k, dtype=float), (self.n_sectors,)).copy()

        ds = xr.Dataset(
            data_vars={
                "Sector_frequency": ("wd", freq),
                "Weibull_A": ("wd", A_arr),
                "Weibull_k": ("wd", k_arr),
                "TI": ti,
            },
            coords={"wd": wd},
        )
        return XRSite(ds)

    def pywake_arrays(self, A=None, k=None):
        """Return raw arrays for PyWake constructors."""
        out = {"wd": self.wind_directions, "freq": self.sector_frequencies}
        if A is not None:
            out["A"] = np.broadcast_to(
                np.asarray(A, dtype=float), (self.n_sectors,)
            ).copy()
        if k is not None:
            out["k"] = np.broadcast_to(
                np.asarray(k, dtype=float), (self.n_sectors,)
            ).copy()
        return out

    def plot(self, ax=None, show=True):
        """Plot the mixture wind rose."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})

        wd_rad = np.deg2rad(self.wind_directions)
        width = np.deg2rad(self.sector_width)

        ax.bar(
            wd_rad,
            self.sector_frequencies,
            width=width * 0.7,
            alpha=0.7,
            label="Mixture model",
            color="crimson",
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

        if show:
            plt.show()
        return ax

    def __repr__(self):
        parts = ", ".join(
            f"{w:.2f}*({c})" for w, c in zip(self.weights, self.components)
        )
        return f"MixtureEllipticalWindRose([{parts}])"
