import inspect
from unittest import mock

import numpy as np
import pytest
from VISSSlib import matching as matching_module
from VISSSlib.matching import *

from helpers import get_test_data_path, get_test_path, readTestSettings

nSample = 100
seed = 0


class TestRotation(object):
    def test_L2F(self):
        # make sure zero rotation doesn't do anything
        np.random.seed(seed)
        L_x = np.random.random(nSample) * 100
        L_y = np.random.random(nSample) * 100
        L_z = np.random.random(nSample) * 100

        F_xs, F_ys, F_zs = rotate_L2F(L_x, L_y, L_z, 0, 0, 0)

        assert np.allclose(L_x, F_xs)
        assert np.allclose(L_y, F_ys)
        assert np.allclose(L_z, F_zs)

    def test_F2L(self):
        # make sure zero rotation doesn't do anything
        np.random.seed(seed)
        F_xs = np.random.random(nSample) * 100
        F_ys = np.random.random(nSample) * 100
        F_zs = np.random.random(nSample) * 100

        L_x, L_y, L_z = rotate_F2L(F_xs, F_ys, F_zs, 0, 0, 0)

        assert np.allclose(F_xs, L_x)
        assert np.allclose(F_ys, L_y)
        assert np.allclose(F_zs, L_z)

    def test_F2L_L2F(self):
        # make sure reverse of rotation doesn't do anything
        np.random.seed(seed)
        F_xs = np.random.random(nSample) * 100
        F_ys = np.random.random(nSample) * 100
        F_zs = np.random.random(nSample) * 100
        phi = np.random.random(nSample)
        theta = np.random.random(nSample)
        psi = np.random.random(nSample)
        L_x, L_y, L_z = rotate_F2L(F_xs, F_ys, F_zs, phi, theta, psi)

        F_xs_2, F_ys_2, F_zs_2 = rotate_L2F(L_x, L_y, L_z, phi, theta, psi)

        assert np.allclose(F_xs_2, F_xs)
        assert np.allclose(F_ys_2, F_ys)
        assert np.allclose(F_zs_2, F_zs)

    def test_F2L_L2F_2(self):
        np.random.seed(seed)
        F_xs = np.random.random(nSample) * 100
        F_ys = np.random.random(nSample) * 100
        F_zs = np.random.random(nSample) * 100
        phi = np.random.random(nSample)
        theta = np.random.random(nSample)
        psi = np.random.random(nSample)
        Ol_x = np.random.random(nSample) * 100
        Of_y = np.random.random(nSample) * 100
        Of_z = np.random.random(nSample) * 100

        L_x, L_y, L_z = shiftRotate_F2L(
            F_xs, F_ys, F_zs, phi, theta, psi, Ol_x, Of_y, Of_z
        )

        F_xs_2, F_ys_2, F_zs_2 = shiftRotate_L2F(
            L_x, L_y, L_z, phi, theta, psi, Ol_x, Of_y, Of_z
        )

        assert np.allclose(F_xs_2, F_xs)
        assert np.allclose(F_ys_2, F_ys)
        assert np.allclose(F_zs_2, F_zs)

    def test_calc_L_z(self):
        # test calc_L_z
        np.random.seed(seed)
        F_xs = np.random.random(nSample) * 100
        F_ys = np.random.random(nSample) * 100
        F_zs = np.random.random(nSample) * 100
        phi = np.random.random(nSample)
        theta = np.random.random(nSample)
        psi = np.random.random(nSample)
        L_x, L_y, L_z = rotate_F2L(F_xs, F_ys, F_zs, phi, theta, psi)

        L_z_test = calc_L_z(L_x, F_ys, F_zs, phi, theta, psi)

        assert np.allclose(L_z_test, L_z)

    def test_calc_L_z_2(self):
        # test calc_L_z
        np.random.seed(seed)
        L_x = np.random.random(nSample) * 100
        L_y = np.random.random(nSample) * 100
        L_z = np.random.random(nSample) * 100
        phi = np.random.random(nSample)
        theta = np.random.random(nSample)
        psi = np.random.random(nSample)
        F_xs, F_ys, F_zs = rotate_L2F(L_x, L_y, L_z, phi, theta, psi)

        L_z_test = calc_L_z(L_x, F_ys, F_zs, phi, theta, psi)

        assert np.allclose(L_z_test, L_z)

    def test_calc_L_z_withOffsets(self):
        np.random.seed(seed)
        F_x = np.random.random(nSample) * 100
        F_y = np.random.random(nSample) * 100
        F_z = np.random.random(nSample) * 100
        phi = np.random.random(nSample)
        theta = np.random.random(nSample)
        psi = np.random.random(nSample)
        Ol_x = 1  # np.random.random(nSample)* 100
        Of_y = 2  # np.random.random(nSample)* 100
        Of_z = 3  # np.random.random(nSample)* 100

        L_x, L_y, L_z = shiftRotate_F2L(
            F_x, F_y, F_z, phi, theta, psi, Ol_x, Of_y, Of_z
        )
        L_z_test = calc_L_z_withOffsets(
            L_x,
            F_y,
            F_z,
            camera_phi=phi,
            camera_theta=theta,
            camera_psi=psi,
            camera_Ofy=Of_y,
            camera_Ofz=Of_z,
            camera_Olx=Ol_x,
        )

        assert np.allclose(L_z_test, L_z)


class TestMatch(object):
    @pytest.fixture(autouse=True)
    def setup_files(self):
        self.config = readTestSettings("test_0.6/test_0.6.yaml")
        self.testPath = get_test_data_path()
        yield

    def testMetaRotation(self):
        case = "20260110"
        metaRotation, fnameMetaRotation = createMetaRotation(
            case, self.config, skipExisting=False, writeNc=False, doPlots=False
        )
        assert np.all(
            np.isclose(
                metaRotation.isel(file_starttime=-1).camera_phi.values, [0.3144, 0.0161]
            )
        )

    def testL1Match(self):
        fname = f"{self.testPath}/test_0.6/products/level1detect/2026/01/10/level1detect_V1.2_test_visss11gb_visss_leader_S1145792_20260110-083000.nc"
        (
            _,
            matchedDats,
            rotate_final,
            rotate_err_final,
            nLeader,
            nFollower,
            nPairs,
            _,
        ) = matchParticles(fname, self.config, writeNc=False, skipExisting=False)

        assert nPairs == 1035
        np.isclose(rotate_final["camera_Ofz"], -20.31999969482422)

    def testManualRotation(self):
        case = "20260110-083000"
        res = manualRotationEstimate(case, self.config, minSamples4rot=10)
        assert res == {
            "20260110-083000": {
                "transformation": {
                    "camera_phi": 0.32747,
                    "camera_theta": 0.489329,
                    "camera_Ofz": -20.298707,
                },
                "transformation_err": {
                    "camera_phi": 0.017591,
                    "camera_theta": 0.015518,
                    "camera_Ofz": 0.283318,
                },
            }
        }

    def testMatchSegmentsIsolated(self):
        # _matchSegments is the file-I/O-free matching/rotation-retrieval
        # core of matchParticles. Spy on the real call matchParticles makes
        # (so the already-opened leader/follower/event datasets don't have
        # to be hand-built here) and then re-invoke it standalone with a
        # fresh copy of the mutable `errors` argument: it must reproduce
        # exactly the same result, proving it depends only on its
        # arguments and is independently callable/testable.
        fname = f"{self.testPath}/test_0.6/products/level1detect/2026/01/10/level1detect_V1.2_test_visss11gb_visss_leader_S1145792_20260110-083000.nc"

        captured = {}
        original = matching_module._matchSegments

        def spy(*args, **kwargs):
            result = original(*args, **kwargs)
            captured["args"] = args
            captured["kwargs"] = kwargs
            captured["result"] = result
            return result

        with mock.patch.object(matching_module, "_matchSegments", side_effect=spy):
            (
                _,
                matchedDats,
                rotate_final,
                rotate_err_final,
                nLeader,
                nFollower,
                nPairs,
                _,
            ) = matchParticles(fname, self.config, writeNc=False, skipExisting=False)

        assert "args" in captured, "_matchSegments was not called"
        assert nPairs == 1035

        callArgs = dict(
            inspect.signature(matching_module._matchSegments)
            .bind(*captured["args"], **captured["kwargs"])
            .arguments
        )
        callArgs["errors"] = callArgs["errors"].copy()

        (
            matchedDats2,
            matchedDat2,
            errorStrs2,
            nSamples2,
            matchedDat4Rot2,
            rotate_result2,
            rotate_err_result2,
            rotate_final2,
            rotate_err_final2,
            nLeader2,
            nFollower2,
        ) = matching_module._matchSegments(**callArgs)

        assert nLeader2 == nLeader
        assert nFollower2 == nFollower
        assert sum(len(d.pair_id) for d in matchedDats2) == nPairs
        assert dict(rotate_final2) == pytest.approx(dict(rotate_final))
        assert dict(rotate_err_final2) == pytest.approx(dict(rotate_err_final))

    def testMatchParticlesOffsetsOnly(self):
        # offsetsOnly short-circuits out of the middle of the segment loop
        # with a differently-shaped 2-tuple; this is now implemented via
        # _MatchEarlyReturn inside _matchSegments and must still surface
        # through matchParticles unchanged.
        fname = f"{self.testPath}/test_0.6/products/level1detect/2026/01/10/level1detect_V1.2_test_visss11gb_visss_leader_S1145792_20260110-083000.nc"
        result = matchParticles(
            fname, self.config, writeNc=False, skipExisting=False, offsetsOnly=True
        )
        assert len(result) == 2
        captureIdOffset, nMatched = result
        assert nMatched > 0
