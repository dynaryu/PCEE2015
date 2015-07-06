import unittest
import numpy as np

from run_nonlinear_sdof import nonlinear_response_sdof, linear_response_sdof
from model_bilinear import model_bilinear
from gmotion import gmotion

el_centro_array = np.load('/Users/hyeuk/Project/PCEE2015/data/El_Centro_Chopra.npy')
el_centro = gmotion(el_centro_array, 0.02)
g_const = 386.0

class TestLinearResponseSdofFunction(unittest.TestCase):
    """Test for Linear Response Sdof Function"""

    def test_linear_elcentro(self):

        """ Figure 7.4.2 (p268, Chopra book)
            m = w/g_const, xi = 0, period = 0.5(sec),
        """

        # Figure 6.6.1 (a) (p209)
        result = []
        epp = model_bilinear(mass=1.0, period=0.5, fy=10.0, alpha=0.0,
            damp_ratio=0.02)
        response = linear_response_sdof(model=epp,
            grecord=el_centro, grecord_factor=-1.0*g_const, dt_comp=0.005,
            flag_newmark='AA')
        spec_values = np.abs(response).max()
        result.append(spec_values['dis'])

        # Figure 6.6.2 (b) (p209)
        epp = model_bilinear(mass=1.0, period=1.0, fy=10.0, alpha=0.0,
            damp_ratio=0.02)
        response = linear_response_sdof(model=epp,
            grecord=el_centro, grecord_factor=-1.0*g_const, dt_comp=0.005,
            flag_newmark='AA')
        spec_values = np.abs(response).max()
        result.append(spec_values['dis'])

        # Figure 6.6.2 (c) (p209)
        epp = model_bilinear(mass=1.0, period=2.0, fy=10.0, alpha=0.0,
            damp_ratio=0.02)
        response = linear_response_sdof(model=epp,
            grecord=el_centro, grecord_factor=-1.0*g_const, dt_comp=0.005,
            flag_newmark='AA')
        spec_values = np.abs(response).max()
        result.append(spec_values['dis'])

        # Figure 6.8.4 (p226)
        epp = model_bilinear(mass=1.0, period=0.02, fy=10.0, alpha=0.0,
            damp_ratio=0.02)
        response = linear_response_sdof(model=epp,
            grecord=el_centro, grecord_factor=-1.0*g_const, dt_comp=0.001,
            flag_newmark='AA')
        spec_values = np.abs(response).max()
        result.append(spec_values['tacc']/g_const)

        # Figure 6.8.5 (p227)
        epp = model_bilinear(mass=1.0, period=30.0, fy=10.0, alpha=0.0,
            damp_ratio=0.02)
        response = linear_response_sdof(model=epp,
            grecord=el_centro, grecord_factor=-1.0*g_const, dt_comp=0.005,
            flag_newmark='AA')
        spec_values = np.abs(response).max()
        result.append(spec_values['dis'])

        result = np.array(result)
        expected_result = np.array([2.67, 5.97, 7.47, 0.321, 8.23])

        message = 'Expecting %s, but it returns %s' % (
            expected_result, result)
        self.assertTrue(np.allclose(
            expected_result, result, rtol=1e-02), message)

class TestNonlinearResponseSdofFunction(unittest.TestCase):
    """Test for Nonlinear Response Sdof Function"""

    def test_simple(self):

        """ Table E5.6 (p192, Chopra book)
            m = 0.2533(kip-sec2/in), k0 = 10.0(kips/in), c = 0.1592
            period = 1.0(sec)
        """
        half_sine = gmotion(np.array(
            [[0.0, 5.0, 8.6602, 10.0, 8.6603, 5.0, 0.0, 0.0, 0.0, 0.0]]).T,
            0.1)

        epp = model_bilinear(mass=0.2533, period=1.0, fy=7.5, alpha=0.00,
            damp_ratio=0.05)

        response = nonlinear_response_sdof(model=epp,
            grecord=half_sine, grecord_factor=1.0, dt_comp = 0.1, flag_newmark='AA')

        # displacement
        expected_result = np.array(
            [0.000, 0.0437, 0.2326, 0.6121, 1.1143, 1.6213, 1.9889,
            2.0947, 1.9233, 1.5593])
        message = 'Expecting %s, but it returns %s' % (
            expected_result, response.dis.values)
        self.assertTrue(np.allclose(
            expected_result, response.dis.values, rtol=1e-03), message)

    def test_elcentro(self):

        # Figure 7.4.3 (a) (p269)
        result = []
        epp = model_bilinear(mass=1.0, period=0.5, fy=10.0, alpha=0.0,
            damp_ratio=0.05)
        response = linear_response_sdof(model=epp,
            grecord=el_centro, grecord_factor=-1.0*g_const, dt_comp=0.001,
            flag_newmark='AA')
        spec_values = np.abs(response).max()
        result.append(spec_values['dis'])
        force0 = epp.omega**2.0*epp.mass*spec_values['dis']

        # Figure 7.4.3 (b) (p269)
        epp = model_bilinear(mass=1.0, period=0.5, fy=0.5*force0, alpha=0.0,
            damp_ratio=0.05)
        response = nonlinear_response_sdof(model=epp,
            grecord=el_centro, grecord_factor=-1.0*g_const, dt_comp=0.001,
            flag_newmark='AA')
        spec_values = np.abs(response).max()
        result.append(spec_values['dis'])

        # Figure 7.4.3 (c) (p269)
        epp = model_bilinear(mass=1.0, period=0.5, fy=0.25*force0, alpha=0.0,
            damp_ratio=0.05)
        response = nonlinear_response_sdof(model=epp,
            grecord=el_centro, grecord_factor=-1.0*g_const, dt_comp=0.001,
            flag_newmark='AA')
        spec_values = np.abs(response).max()
        result.append(spec_values['dis'])

        # Figure 7.4.3 (d) (p269)
        epp = model_bilinear(mass=1.0, period=0.5, fy=0.125*force0, alpha=0.0,
            damp_ratio=0.05)
        response = nonlinear_response_sdof(model=epp,
            grecord=el_centro, grecord_factor=-1.0*g_const, dt_comp=0.001,
            flag_newmark='AA')
        spec_values = np.abs(response).max()
        result.append(spec_values['dis'])

        result = np.array(result)
        expected_result = np.array([2.25, 1.62, 1.75, 2.07])

        message = 'Expecting %s, but it returns %s' % (
            expected_result, result)
        self.assertTrue(np.allclose(
            expected_result, result, rtol=1e-02), message)

if __name__ == '__main__':
    unittest.main()
