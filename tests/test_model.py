"""Tests for the elliptical wind direction rose model."""

import numpy as np
import pytest

from edrose.model import (
    EllipticalWindRose,
    MixtureEllipticalWindRose,
    _compute_frequencies,
    _elliptical_cdf,
)

# Tolerance for floating-point comparisons
ATOL = 1e-10


# ---------------------------------------------------------------------------
# CDF tests
# ---------------------------------------------------------------------------

class TestEllipticalCDF:
    """Tests for the base elliptical CDF."""

    def test_cdf_at_boundaries(self):
        """CDF should be 0 at 0, 0.25 at pi/2, 0.5 at pi, etc."""
        a = 1.0
        assert abs(_elliptical_cdf(0.0, a)) < ATOL
        assert abs(_elliptical_cdf(np.pi / 2, a) - 0.25) < ATOL
        assert abs(_elliptical_cdf(np.pi, a) - 0.5) < ATOL
        assert abs(_elliptical_cdf(3 * np.pi / 2, a) - 0.75) < ATOL

    def test_cdf_monotonic(self):
        """CDF should be monotonically non-decreasing."""
        for a in [0.3, 0.56, 1.0, 2.0]:
            theta = np.linspace(0, 2 * np.pi - 1e-10, 500)
            cdf = _elliptical_cdf(theta, a)
            assert np.all(np.diff(cdf) >= -ATOL), f"CDF not monotonic for a={a}"

    def test_cdf_uniform(self):
        """When a = 1/sqrt(pi), CDF should be linear (uniform distribution)."""
        a = 1.0 / np.sqrt(np.pi)
        theta = np.linspace(0, 2 * np.pi - 1e-10, 100)
        cdf = _elliptical_cdf(theta, a)
        expected = theta / (2 * np.pi)
        np.testing.assert_allclose(cdf, expected, atol=1e-8)

    def test_cdf_vectorised(self):
        """CDF should work on arrays."""
        a = 1.0
        theta = np.array([0, np.pi / 4, np.pi / 2, np.pi])
        cdf = _elliptical_cdf(theta, a)
        assert cdf.shape == (4,)

    def test_cdf_periodicity(self):
        """CDF(theta + 2pi) mod 1 should equal CDF(theta)."""
        a = 0.8
        theta = np.linspace(0.1, 2 * np.pi - 0.1, 50)
        cdf1 = _elliptical_cdf(theta, a)
        cdf2 = _elliptical_cdf(theta + 2 * np.pi, a)
        np.testing.assert_allclose(cdf1, cdf2, atol=1e-12)


# ---------------------------------------------------------------------------
# Sector frequency tests
# ---------------------------------------------------------------------------

class TestSectorFrequencies:
    """Tests for sector frequency computation."""

    @pytest.mark.parametrize("a", [0.3, 0.56, 1.0, 1.5, 2.5])
    @pytest.mark.parametrize("n_sectors", [4, 12, 36, 72])
    def test_sum_to_one_no_folding(self, a, n_sectors):
        """Sector frequencies must sum to 1 (f=0)."""
        wd = np.linspace(0, 360, n_sectors, endpoint=False)
        freq = _compute_frequencies(a, 0.0, 0.0, wd, n_sectors)
        assert abs(freq.sum() - 1.0) < 1e-10, f"Sum={freq.sum()} for a={a}, n={n_sectors}"

    @pytest.mark.parametrize("f", [0.0, 0.3, 0.7, 1.0])
    def test_sum_to_one_with_folding(self, f):
        """Sector frequencies must sum to 1 even with folding."""
        a, n = 0.8, 12
        wd = np.linspace(0, 360, n, endpoint=False)
        freq = _compute_frequencies(a, f, 0.0, wd, n)
        assert abs(freq.sum() - 1.0) < 1e-10, f"Sum={freq.sum()} for f={f}"

    @pytest.mark.parametrize("theta_prev", [0, 45, 90, 180, 270, 350])
    def test_sum_to_one_with_rotation(self, theta_prev):
        """Rotation should not affect the total probability."""
        a, f, n = 0.8, 0.3, 12
        wd = np.linspace(0, 360, n, endpoint=False)
        freq = _compute_frequencies(a, f, theta_prev, wd, n)
        assert abs(freq.sum() - 1.0) < 1e-10

    def test_uniform_distribution(self):
        """a = 1/sqrt(pi), f=0 should give uniform sector frequencies."""
        a = 1.0 / np.sqrt(np.pi)
        n = 12
        wd = np.linspace(0, 360, n, endpoint=False)
        freq = _compute_frequencies(a, 0.0, 0.0, wd, n)
        expected = np.ones(n) / n
        np.testing.assert_allclose(freq, expected, atol=1e-8)

    def test_symmetry_no_folding(self):
        """With f=0, opposite sectors should have equal probability."""
        a, n = 0.8, 12
        wd = np.linspace(0, 360, n, endpoint=False)
        freq = _compute_frequencies(a, 0.0, 0.0, wd, n)
        half = n // 2
        np.testing.assert_allclose(freq[:half], freq[half:], atol=1e-12)

    def test_all_positive(self):
        """Sector frequencies should never be negative."""
        for a in [0.2, 0.5, 1.0, 3.0]:
            for f in [0.0, 0.5, 1.0]:
                wd = np.linspace(0, 360, 36, endpoint=False)
                freq = _compute_frequencies(a, f, 45.0, wd, 36)
                assert np.all(freq >= -1e-15), f"Negative freq for a={a}, f={f}"

    def test_folding_increases_prevailing(self):
        """Folding should increase probability in the prevailing half."""
        a, n = 0.8, 12
        wd = np.linspace(0, 360, n, endpoint=False)
        freq_nofold = _compute_frequencies(a, 0.0, 0.0, wd, n)
        freq_fold = _compute_frequencies(a, 0.5, 0.0, wd, n)
        # The prevailing sector (around 0 degrees) should increase
        assert freq_fold[0] > freq_nofold[0]


# ---------------------------------------------------------------------------
# EllipticalWindRose class tests
# ---------------------------------------------------------------------------

class TestEllipticalWindRose:
    """Tests for the main class."""

    def test_constructor(self):
        wr = EllipticalWindRose(a=0.8, f=0.1, theta_prev=180, n_sectors=12)
        assert wr.a == 0.8
        assert wr.f == 0.1
        assert wr.theta_prev == 180.0
        assert wr.n_sectors == 12

    def test_wind_directions(self):
        wr = EllipticalWindRose(a=1.0, n_sectors=12)
        wd = wr.wind_directions
        np.testing.assert_allclose(wd, np.arange(0, 360, 30))

    def test_sector_width(self):
        wr = EllipticalWindRose(a=1.0, n_sectors=36)
        assert wr.sector_width == 10.0

    def test_invalid_a(self):
        with pytest.raises(ValueError):
            EllipticalWindRose(a=-1)
        with pytest.raises(ValueError):
            EllipticalWindRose(a=0)

    def test_invalid_f(self):
        with pytest.raises(ValueError):
            EllipticalWindRose(a=1.0, f=-0.1)
        with pytest.raises(ValueError):
            EllipticalWindRose(a=1.0, f=1.1)

    def test_repr(self):
        wr = EllipticalWindRose(a=0.8, f=0.1, theta_prev=180)
        r = repr(wr)
        assert "0.8000" in r
        assert "0.1000" in r


# ---------------------------------------------------------------------------
# Fitting tests
# ---------------------------------------------------------------------------

class TestFitting:
    """Tests for fitting the model to data."""

    def test_fit_recovers_parameters(self):
        """Fitting synthetic data should recover the original parameters."""
        a_true, f_true, tp_true = 0.8, 0.3, 210.0
        n = 36
        wr_true = EllipticalWindRose(a=a_true, f=f_true, theta_prev=tp_true, n_sectors=n)
        measured = wr_true.sector_frequencies

        wr_fit = EllipticalWindRose.fit(measured, theta_prev=tp_true)

        assert abs(wr_fit.a - a_true) < 0.02, f"a: {wr_fit.a} vs {a_true}"
        assert abs(wr_fit.f - f_true) < 0.02, f"f: {wr_fit.f} vs {f_true}"
        assert wr_fit.r_squared > 0.999

    def test_fit_recovers_uniform(self):
        """Fitting a uniform rose should give a ~ 1/sqrt(pi), f ~ 0."""
        n = 12
        uniform = np.ones(n) / n
        wr_fit = EllipticalWindRose.fit(uniform)
        a_uniform = 1.0 / np.sqrt(np.pi)
        assert abs(wr_fit.a - a_uniform) < 0.05
        assert wr_fit.f < 0.05

    def test_fit_auto_theta_prev(self):
        """Fitting without specifying theta_prev should find a good fit."""
        a_true, f_true, tp_true = 1.2, 0.5, 90.0
        n = 24
        wr_true = EllipticalWindRose(a=a_true, f=f_true, theta_prev=tp_true, n_sectors=n)
        measured = wr_true.sector_frequencies

        wr_fit = EllipticalWindRose.fit(measured)
        assert wr_fit.r_squared > 0.99

    def test_goodness_of_fit(self):
        wr = EllipticalWindRose(a=0.8, f=0.2, theta_prev=0, n_sectors=12)
        freq = wr.sector_frequencies
        gof = wr.goodness_of_fit(freq)
        assert abs(gof["r_squared"] - 1.0) < 1e-10
        assert gof["rmse"] < 1e-10


# ---------------------------------------------------------------------------
# Mixture model tests
# ---------------------------------------------------------------------------

class TestMixtureModel:
    """Tests for the mixture wind rose."""

    def test_sum_to_one(self):
        c1 = EllipticalWindRose(a=0.8, f=0.1, theta_prev=180, n_sectors=36)
        c2 = EllipticalWindRose(a=1.1, f=1.0, theta_prev=260.5, n_sectors=36)
        mix = MixtureEllipticalWindRose([c1, c2], weights=[0.85, 0.15])
        assert abs(mix.sector_frequencies.sum() - 1.0) < 1e-10

    def test_single_component(self):
        """A mixture with one component should equal that component."""
        wr = EllipticalWindRose(a=0.8, f=0.3, theta_prev=90, n_sectors=12)
        mix = MixtureEllipticalWindRose([wr], weights=[1.0])
        np.testing.assert_allclose(
            mix.sector_frequencies, wr.sector_frequencies, atol=1e-12
        )

    def test_weight_normalisation(self):
        c1 = EllipticalWindRose(a=0.8, n_sectors=12)
        c2 = EllipticalWindRose(a=1.2, n_sectors=12)
        mix = MixtureEllipticalWindRose([c1, c2], weights=[2, 3])
        np.testing.assert_allclose(mix.weights, [0.4, 0.6])

    def test_mismatched_n_sectors(self):
        c1 = EllipticalWindRose(a=0.8, n_sectors=12)
        c2 = EllipticalWindRose(a=1.2, n_sectors=36)
        with pytest.raises(ValueError):
            MixtureEllipticalWindRose([c1, c2], weights=[0.5, 0.5])


# ---------------------------------------------------------------------------
# PyWake arrays test
# ---------------------------------------------------------------------------

class TestPyWakeArrays:
    """Tests for PyWake-compatible output."""

    def test_pywake_arrays_scalar(self):
        wr = EllipticalWindRose(a=0.8, f=0.3, theta_prev=180, n_sectors=12)
        out = wr.pywake_arrays(A=9.0, k=2.0)
        assert out["wd"].shape == (12,)
        assert out["freq"].shape == (12,)
        assert out["A"].shape == (12,)
        assert out["k"].shape == (12,)
        np.testing.assert_allclose(out["A"], 9.0)
        np.testing.assert_allclose(out["k"], 2.0)

    def test_pywake_arrays_per_sector(self):
        wr = EllipticalWindRose(a=0.8, n_sectors=4)
        A = [8, 9, 10, 11]
        k = [1.8, 2.0, 2.2, 2.4]
        out = wr.pywake_arrays(A=A, k=k)
        np.testing.assert_allclose(out["A"], A)
        np.testing.assert_allclose(out["k"], k)

    def test_pywake_arrays_no_weibull(self):
        wr = EllipticalWindRose(a=0.8, n_sectors=12)
        out = wr.pywake_arrays()
        assert "A" not in out
        assert "k" not in out
        assert "freq" in out
