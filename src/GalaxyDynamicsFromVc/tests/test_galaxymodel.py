# -*- coding: utf-8 -*-
import pytest
import numpy as np

from .. import galaxymodel

#%%

def test_DM_halo(Mvir = 10.0, cvir = 11.0, Delta_vir = 222.0, h = 0.7):

    halo = galaxymodel.DM_halo(Mvir, cvir, Delta_vir=Delta_vir, h_cosmo=h)    
    
    assert halo.Mvir_in_1e11Msun == Mvir
    assert halo.Mvir_in_Msun == Mvir*1.0e11
    assert halo.cvir == cvir
    assert halo.Delta_vir == Delta_vir
    assert halo.h_cosmo == h
    assert halo.rvir_kpc == (Mvir*1.0e11/4.0/np.pi*3.0/Delta_vir/halo._rho_critical_Msun_kpc3)**(1./3.)
    assert halo.rs_kpc == halo.rvir_kpc/cvir


@pytest.fixture
def DM_halo_default():
    return galaxymodel.DM_halo(10.0, 12.0)

@pytest.mark.parametrize("Mvir",[(10.1),(6.0)])
def test_DM_halo_change_of_Mvir(DM_halo_default,Mvir):

    halo = DM_halo_default
    halo.update_Mvir_in_1e11Msun(Mvir)
    
    assert halo.Mvir_in_1e11Msun == Mvir
    assert halo.Mvir_in_Msun == Mvir*1.0e11
    assert halo.rvir_kpc == (Mvir*1.0e11/4.0/np.pi*3.0/halo.Delta_vir/halo._rho_critical_Msun_kpc3)**(1./3.)
    assert halo.rs_kpc == halo.rvir_kpc/halo.cvir
    
@pytest.mark.parametrize("cvir",[(7.0),(13.17)])
def test_DM_halo_change_of_cvir(DM_halo_default,cvir):
    
    halo = DM_halo_default
    halo.update_cvir(cvir)
    
    assert halo.cvir == cvir
    assert halo.rvir_kpc == (halo.Mvir_in_Msun/4.0/np.pi*3.0/halo.Delta_vir/halo._rho_critical_Msun_kpc3)**(1./3.)
    assert halo.rs_kpc == halo.rvir_kpc/halo.cvir
    
@pytest.mark.parametrize("r",[(5.0),(8.0),(30.0)])
def test_NFW_squared_velocity_equals_velocity_squared(r):
    Mvir = 10.5
    cvir = 12.8
    Delta_vir = 220.2
    h = 0.77
    halo = galaxymodel.NFW(Mvir, cvir, Delta_vir=Delta_vir, h_cosmo=h)
    assert halo.circular_velocity_km_s(r)**2 == pytest.approx(halo.squared_circular_velocity_km2_s2(r),1.0e-8)

@pytest.mark.parametrize("R,z",[(5.0,0.0),(8.0,0.2),(30.0,-1.0)])
def test_Miyamoto_Nagai_squared_velocity_equals_velocity_squared(R,z):
    M = 1.7e10
    a = 2.1
    b = 0.27
    pot = galaxymodel.Miyamoto_Nagai_disk(total_mass_Msun=M, a_kpc=a, b_kpc=b)
    assert pot.circular_velocity_km_s(R,z)**2 == pytest.approx(pot.squared_circular_velocity_km2_s2(R,z),1.0e-8)

@pytest.mark.parametrize("r",[(5.0),(8.0),(30.0)])
def test_Plummer_squared_velocity_equals_velocity_squared(r):
    M = 1.3e10
    b = 0.27
    pot = galaxymodel.Plummer(total_mass_Msun=M, b_kpc=b)
    assert pot.circular_velocity_km_s(r)**2 == pytest.approx(pot.squared_circular_velocity_km2_s2(r),1.0e-8)
    
def test_inputs_Galactic_model():
    halo = galaxymodel.NFW()
    bulge = galaxymodel.Plummer()
    disk = galaxymodel.Miyamoto_Nagai_disk()
    galaxymodel.Galactic_model(halo, bulge, disk)
    with pytest.raises(AssertionError):
        galaxymodel.Galactic_model(1,[])
    with pytest.raises(AssertionError):
        galaxymodel.Galactic_model(1,7)
    with pytest.raises(AssertionError):
        galaxymodel.Galactic_model(1,1.0)

@pytest.mark.parametrize("r",[(5.0),(8.0),(30.0)])
def test_Galactic_model_circular_velocity_equals_sum_from_components(r):
    halo = galaxymodel.NFW()
    disk_1 = galaxymodel.Miyamoto_Nagai_disk()
    bulge = galaxymodel.Plummer()
    
    potentials = [halo, disk_1, bulge]
    
    GM = galaxymodel.Galactic_model(*potentials)
    
    Vc_tot = np.sqrt(sum([pot.squared_circular_velocity_km2_s2(r) for pot in potentials]))
    
    assert GM.circular_velocity_km_s(r) == pytest.approx(Vc_tot,1.0e-8)
    
    
    
    
    
