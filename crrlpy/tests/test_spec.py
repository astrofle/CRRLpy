#!/usr/bin/env python

import unittest
import numpy as np
from crrlpy.crrls import dv2df
from crrlpy.spec import Spectrum
import crrlpy.spec as cs

class spec_tests(unittest.TestCase):
    
    def setUp(self):
        x = np.arange(0,10)
        y = np.arange(0,10)
        
        self.spec = Spectrum(x, y)
 
    def test_mask_edges_redge_only(self):
        
        x = np.arange(0,10)
        y = np.arange(0,10)
        
        spec = Spectrum(x, y)
        spec.mask_edges(2)
        
        self.assertEqual(7, spec.x.compressed()[-1])
        
    def test_mask_edges_both(self):
        
        x = np.arange(0,10)
        y = np.arange(0,10)
        
        spec = Spectrum(x, y)
        spec.mask_edges(2, 3)
        
        self.assertEqual(3, spec.x.compressed()[0])
        self.assertEqual(7, spec.x.compressed()[-1])
        
    def test_mask_edges_nonint(self):
        
        x = np.arange(0,10)
        y = np.arange(0,10)
        
        spec = Spectrum(x, y)
        self.assertRaises(ValueError, spec.mask_edges, 'two', 'three')
        
    def test_mask_ranges_one(self):
        
        self.spec.mask_ranges([(0,8)])
        self.assertEqual(8, self.spec.x.compressed()[0])
        
    def test_mask_ranges(self):
        
        self.spec.mask_ranges([(0,2), (8,10)])
        self.assertEqual(2, self.spec.x.compressed()[0])
        self.assertEqual(7, self.spec.x.compressed()[-1])
        
    def test_bpcorr(self):
        
        self.spec.bandpass_corr(1)
        np.testing.assert_allclose(0, self.spec.y_bpcorr.compressed()[-1], rtol=1)
        
    def test_find_good_lines_one(self):
        
        data = np.loadtxt('example_spec.ascii')
        spec = Spectrum(data[:,0], data[:,1])
        spec.find_good_lines('RRL_CIalpha', ['RRL_CIalpha', 'RRL_CIbeta', 'RRL_CIgamma' ,'RRL_CIdelta'], z=0, separation=10e-6)
        self.assertEqual([584.], spec.good_lines['RRL_CIalpha'])
        
    def test_find_good_lines_none(self):
        
        data = np.loadtxt('example_spec.ascii')
        spec = Spectrum(data[:,0], data[:,1])
        spec.find_good_lines('RRL_CIalpha', ['RRL_CIalpha', 'RRL_CIbeta', 'RRL_CIgamma' ,'RRL_CIdelta'], z=0, separation=1)
        self.assertEqual({'RRL_CIalpha': []}, spec.good_lines)
        
    def test_find_good_lines_many(self):
        
        data = np.loadtxt('example_spec.ascii')
        spec = Spectrum(data[:,0], data[:,1])
        spec.find_good_lines('RRL_CIalpha', ['RRL_CIalpha', 'RRL_CIbeta', 'RRL_CIgamma' ,'RRL_CIdelta'], z=0, separation=10e-6)
        spec.find_good_lines('RRL_CIbeta', ['RRL_CIalpha', 'RRL_CIbeta', 'RRL_CIgamma' ,'RRL_CIdelta'], z=0, separation=10e-6)
        self.assertEqual({'RRL_CIalpha': [584.0], 'RRL_CIbeta': [735.0]}, spec.good_lines)
        
    def test_find_good_lines_many_realistic(self):
        
        data = np.loadtxt('example_spec.ascii')
        spec = Spectrum(data[:,0], data[:,1])
        sep = dv2df(spec.x.mean()*1e6, 30e3)/1e6
        spec.find_good_lines('RRL_CIalpha', ['RRL_CIalpha', 'RRL_CIbeta', 'RRL_CIgamma' ,'RRL_CIdelta'], z=0, separation=sep)
        spec.find_good_lines('RRL_CIbeta', ['RRL_CIalpha', 'RRL_CIbeta', 'RRL_CIgamma' ,'RRL_CIdelta'], z=0, separation=sep)
        self.assertEqual({'RRL_CIalpha': [584.0], 'RRL_CIbeta': []}, spec.good_lines)
        
    def test_find_good_lines_many_noedge(self):
        
        data = np.loadtxt('example_spec.ascii')
        spec = Spectrum(data[:,0], data[:,1])
        sep = dv2df(spec.x.mean()*1e6, 30e3)/1e6
        spec.find_good_lines('RRL_CIalpha', ['RRL_CIalpha', 'RRL_CIbeta', 'RRL_CIgamma' ,'RRL_CIdelta'], z=0, separation=sep)
        spec.find_good_lines('RRL_CIdelta', ['RRL_CIalpha', 'RRL_CIbeta', 'RRL_CIgamma' ,'RRL_CIdelta'], z=0, separation=sep, redge=0)
        self.assertEqual({'RRL_CIalpha': [584.0], 'RRL_CIdelta': [925.0, 926.0]}, spec.good_lines)
        
    def test_find_good_lines_many_edge(self):
        
        data = np.loadtxt('example_spec.ascii')
        spec = Spectrum(data[:,0], data[:,1])
        sep = dv2df(spec.x.mean()*1e6, 30e3)/1e6
        spec.find_good_lines('RRL_CIalpha', ['RRL_CIalpha', 'RRL_CIbeta', 'RRL_CIgamma' ,'RRL_CIdelta'], z=0, separation=sep)
        spec.find_good_lines('RRL_CIdelta', ['RRL_CIalpha', 'RRL_CIbeta', 'RRL_CIgamma' ,'RRL_CIdelta'], z=0, separation=sep, redge=0.1)
        self.assertEqual({'RRL_CIalpha': [584.0], 'RRL_CIdelta': [925.0]}, spec.good_lines)
        
class spec_utils_test(unittest.TestCase):
    
    def setUp(self):
        self.ls = np.load('linelist.npy')
        
    def test_distribute_lines(self):
        lgs = cs.distribute_lines(self.ls, 18)
        self.assertEqual(lgs[10][1], 623.0)

if __name__ == '__main__':
    unittest.main()
