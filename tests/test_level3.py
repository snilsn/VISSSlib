import numpy as np
import pytest
from VISSSlib.level3.combined_riming import (
    dry_density_air,
    dynamic_viscosity_air,
    mass_size,
)


class TestDynamicViscosityAir:
    """Unit tests for combined_riming.dynamic_viscosity_air (Sutherland's
    law), a pure function of temperature.
    """

    @pytest.mark.unit
    def test_equals_mu0_at_reference_temperature(self):
        # at T0=273K the (T0+C)/(T+C) and (T/T0)^1.5 factors are both 1
        assert np.isclose(dynamic_viscosity_air(273.0), 1.716e-5)

    @pytest.mark.unit
    def test_increases_with_temperature(self):
        assert dynamic_viscosity_air(300.0) > dynamic_viscosity_air(273.0)


class TestDryDensityAir:
    """Unit tests for combined_riming.dry_density_air (ideal gas law), a
    pure function of temperature and pressure.
    """

    @pytest.mark.unit
    def test_standard_atmosphere_sea_level(self):
        # ISA sea-level conditions: 15degC, 101325 Pa -> ~1.225 kg/m^3
        rho = dry_density_air(288.15, 101325)
        assert np.isclose(rho, 1.225, rtol=1e-3)

    @pytest.mark.unit
    def test_scales_linearly_with_pressure(self):
        rho1 = dry_density_air(273.0, 101325)
        rho2 = dry_density_air(273.0, 2 * 101325)
        assert np.isclose(rho2, 2 * rho1)


class TestMassSize:
    """Unit tests for combined_riming.mass_size, a cubic-spline
    interpolation over the Maherndl et al. (2023) lookup table.
    """

    @pytest.mark.unit
    def test_returns_exact_table_values_at_nodes(self):
        a, b = mass_size(0.0)
        assert np.isclose(a, 0.0324)
        assert np.isclose(b, 2.10)

        a, b = mass_size(0.129)
        assert np.isclose(a, 22.2)
        assert np.isclose(b, 2.85)

    @pytest.mark.unit
    def test_clips_scalar_above_table_max_to_last_entry(self):
        aClipped, bClipped = mass_size(100.0)
        aMax, bMax = mass_size(0.8155)
        assert np.isclose(aClipped, aMax)
        assert np.isclose(bClipped, bMax)

    @pytest.mark.unit
    def test_clips_array_values_above_table_max_in_place(self):
        # mass_size mutates array input in place when clipping to the
        # table's upper bound -- documenting this as a known gotcha
        M = np.array([0.0, 100.0])
        mass_size(M)
        assert M[1] == 0.8155
