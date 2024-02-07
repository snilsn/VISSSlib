from VISSSlib.matching import *
import numpy as np

nSample = 100
seed = 0

class TestRotation(object):


    def test_L2F(self):
        #make sure zero rotation doesn't do anything
        np.random.seed(seed)
        L_x = np.random.random(nSample) * 100
        L_y = np.random.random(nSample) * 100
        L_z = np.random.random(nSample) * 100

        F_xs, F_ys, F_zs = rotate_L2F(L_x, L_y, L_z, 0, 0, 0)

        assert np.allclose(L_x, F_xs)
        assert np.allclose(L_y, F_ys)
        assert np.allclose(L_z, F_zs)

    def test_F2L(self):

        #make sure zero rotation doesn't do anything
        np.random.seed(seed)
        F_xs = np.random.random(nSample) * 100
        F_ys = np.random.random(nSample) * 100
        F_zs = np.random.random(nSample) * 100


        L_x, L_y, L_z = rotate_F2L(F_xs, F_ys, F_zs, 0, 0, 0)

        assert np.allclose(F_xs, L_x)
        assert np.allclose(F_ys, L_y)
        assert np.allclose(F_zs, L_z)


    def test_F2L_L2F(self):

        #make sure reverse of rotation doesn't do anything
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
        Ol_x = np.random.random(nSample)* 100
        Of_y = np.random.random(nSample)* 100
        Of_z = np.random.random(nSample)* 100


        L_x, L_y, L_z = shiftRotate_F2L(F_xs, F_ys, F_zs, phi, theta, psi,Ol_x, Of_y, Of_z)

        F_xs_2, F_ys_2, F_zs_2 = shiftRotate_L2F(L_x, L_y, L_z, phi, theta, psi,Ol_x, Of_y, Of_z)

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
        Ol_x = 1#np.random.random(nSample)* 100
        Of_y = 2#np.random.random(nSample)* 100
        Of_z = 3#np.random.random(nSample)* 100

        L_x, L_y, L_z = shiftRotate_F2L(F_x, F_y, F_z, phi, theta, psi, Ol_x, Of_y, Of_z)
        L_z_test = calc_L_z_withOffsets(L_x, F_y, F_z, camera_phi=phi, camera_theta=theta, camera_psi=psi, camera_Ofy=Of_y, camera_Ofz=Of_z, camera_Olx=Ol_x)

        print(L_z_test- L_z)

        assert np.allclose(L_z_test, L_z)



