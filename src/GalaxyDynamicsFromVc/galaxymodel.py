# -*- coding: utf-8 -*-

import numpy as np

from GalaxyDynamicsFromVc import units


#%%

class Potential:
    pass

class DM_halo(Potential):
    def __init__(self, Mvir_in_1e11Msun, cvir, Delta_vir=200.0, h_cosmo=0.678):
        self._Mvir_in_1e11Msun = Mvir_in_1e11Msun
        self._Mvir_in_Msun = self.Mvir_in_1e11Msun*1.0e11
        self._cvir = cvir
        self._Delta_vir = Delta_vir
        self._h_cosmo = h_cosmo
        self._rho_critical_Msun_kpc3 = 2.77536627e2*h_cosmo**2
        self._rvir_kpc = self._set_rvir_kpc()
        self._rs_kpc = self._set_rs_kpc()

    @property
    def Mvir_in_1e11Msun(self):
        return self._Mvir_in_1e11Msun
    
    @property
    def Mvir_in_Msun(self):
        return self._Mvir_in_Msun

    @property
    def cvir(self):
        return self._cvir

    @property
    def Delta_vir(self):
        return self._Delta_vir
    
    @property
    def h_cosmo(self):
        return self._h_cosmo

    @property
    def rvir_kpc(self):
        return self._rvir_kpc

    @property
    def rs_kpc(self):
        return self._rs_kpc

    def _set_rvir_kpc(self):
        return (self.Mvir_in_Msun/4.0/np.pi*3.0/self.Delta_vir/self._rho_critical_Msun_kpc3)**(1./3.)

    def _set_rs_kpc(self):
        return self.rvir_kpc/self.cvir
    
    def update_Mvir_in_1e11Msun(self, value):
        self._Mvir_in_1e11Msun = value
        self._Mvir_in_Msun = self.Mvir_in_1e11Msun*1.0e11
        self._rvir_kpc = self._set_rvir_kpc()
        self._rs_kpc = self._set_rs_kpc()
        
    def update_cvir(self, value):
        self._cvir = value
        self._rs_kpc = self._set_rs_kpc()

class NFW(DM_halo):
    def __init__(self, Mvir_in_1e11Msun=11.2, cvir=12.8, Delta_vir=200.0, h_cosmo=0.678):
        super().__init__(Mvir_in_1e11Msun, cvir, Delta_vir=Delta_vir, h_cosmo=h_cosmo)
        
    def density_Msun_kpc3(self, r_kpc):
        normalisation = self.Mvir_in_Msun/(4.0*np.pi*((self.rs_kpc)**3)*(1.0/(self.cvir+1.0)+np.log(self.cvir+1.0)-1.0))
        rs_over_r = self.rs_kpc/r_kpc
        return normalisation*(rs_over_r)/(1.0+1.0/rs_over_r)**2

    def density_GeV_cm3(self, r_kpc):
        return units._Msun_kpc3_to_GeV_cm3_factor*self.density_Msun_kpc3(r_kpc)
    
    def enclosed_mass_in_Msun(self, r_kpc):
        normalisation = self.Mvir_in_Msun/((1.0/(self.cvir+1.0)+np.log(self.cvir+1.0)-1.0))
        r_over_rs = r_kpc/self.rs_kpc
        return normalisation*(1.0/(r_over_rs+1.0)+np.log(r_over_rs+1.0)-1.0)
    
    def circular_velocity_km_s(self, r_kpc):
        Mv = self.Mvir_in_Msun/2.32e7 # 2.32e7 Msun
        Grav_constant = 1.0 # gravitational cte
        rs_3 = self.rs_kpc**3
        normalisation = Mv/(4.0*np.pi*(rs_3)*(1.0/(self.cvir+1.0)+np.log(self.cvir+1.0)-1.0))
        inner_factor = 4.0*np.pi*Grav_constant*normalisation
        convFactor = 10.0
        r_over_rs = r_kpc/self.rs_kpc
        return convFactor*np.sqrt(inner_factor*(rs_3)*(1.0/(r_over_rs+1.0)+np.log(1.0+r_over_rs)-1.0)/r_kpc)

    def squared_circular_velocity_km2_s2(self, r_kpc):
        Mv = self.Mvir_in_Msun/2.32e7 # to transform it in units of 2.32e7 Msun
        Grav_constant = 1.0 # gravitational cte
        rs_3 = self.rs_kpc**3
        normalisation = Mv/(4.0*np.pi*(rs_3)*(1.0/(self.cvir+1.0)+np.log(self.cvir+1.0)-1.0))
        inner_factor = 4.0*np.pi*Grav_constant*normalisation
        convFactor = 100.0
        r_over_rs = r_kpc/self.rs_kpc
        return convFactor*(inner_factor*(rs_3)*(1.0/(r_over_rs+1.0)+np.log(1.0+r_over_rs)-1.0)/r_kpc)        
        
class Miyamoto_Nagai_disk(Potential):
    def __init__(self, total_mass_Msun=3.944e10, a_kpc=5.3, b_kpc=0.25):
        self.total_mass_Msun = total_mass_Msun
        self.a_kpc = a_kpc
        self.b_kpc = b_kpc
        
    @property
    def total_mass_Msun(self):
        return self._total_mass_Msun
    @total_mass_Msun.setter
    def total_mass_Msun(self, value):
        self._total_mass_Msun = value
        
    @property
    def a_kpc(self):
        return self._a_kpc
    @a_kpc.setter
    def a_kpc(self, value):
        self._a_kpc = value
        
    @property
    def b_kpc(self):
        return self._b_kpc
    @b_kpc.setter
    def b_kpc(self, value):
        self._b_kpc = value
        
    def density_Msun_kpc3(self, R_kpc, z_kpc=0):
        sqrzb = np.sqrt(z_kpc**2 + self.b_kpc**2)
        internal = (self.a_kpc+3.0*sqrzb)*(self.a_kpc+sqrzb)**2
        denominator = ((R_kpc**2+(self.a_kpc+sqrzb)**2)**(5.0/2.0)) * (sqrzb**(3))
        rho = self.b_kpc**2*self.total_mass_Msun/(4.0*np.pi)
        return rho*(self.a_kpc*R_kpc**2 + internal)/denominator
    
    def density_GeV_cm3(self, R_kpc, z_kpc=0):
        return units._Msun_kpc3_to_GeV_cm3_factor*self.density_Msun_kpc3(R_kpc, z_kpc)
    
    def circular_velocity_km_s(self, R_kpc, z_kpc=0):
        Grav_constant = 1 # gravitational cte.
        M = self.total_mass_Msun/2.32e7
        convFactor = 10.0
        return convFactor*R_kpc*np.sqrt(Grav_constant*M)/(R_kpc**2 + (self.a_kpc + np.sqrt(z_kpc**2 + self.b_kpc**2))**2)**(3./4.)

    def squared_circular_velocity_km2_s2(self, R_kpc, z_kpc=0):
        Grav_constant = 1 # gravitational cte.
        M = self.total_mass_Msun/2.32e7
        convFactor = 100.0
        R_2 = R_kpc**2
        return convFactor*R_2*Grav_constant*M/(R_2 + (self.a_kpc + np.sqrt(z_kpc**2 + self.b_kpc**2))**2)**(3./2.)

class Plummer(Potential):
    def __init__(self, total_mass_Msun=1.0672e10, b_kpc=0.3):
        self.total_mass_Msun = total_mass_Msun
        self.b_kpc = b_kpc
        
    @property
    def total_mass_Msun(self):
        return self._total_mass_Msun
    @total_mass_Msun.setter
    def total_mass_Msun(self, value):
        self._total_mass_Msun = value
    
    @property
    def b_kpc(self):
        return self._b_kpc
    @b_kpc.setter
    def b_kpc(self, value):
        self._b_kpc = value
        
    def enclosed_mass_in_Msun(self, r_kpc):
        return self.total_mass_Msun*r_kpc**3*(self.b_kpc**2 + r_kpc**2)**(-3.0/2.0)
    
    def circular_velocity_km_s(self, r_kpc):
        Grav_constant = 1 # gravitational cte.
        M = self.total_mass_Msun/2.32e7
        convFactor = 10.0
        return convFactor*r_kpc*np.sqrt(Grav_constant*M)/(r_kpc**2 + self.b_kpc**2)**(3./4.)
    
    def squared_circular_velocity_km2_s2(self, r_kpc):
        Grav_constant = 1
        M = self.total_mass_Msun/2.32e7
        convFactor = 100.0
        r_2 = r_kpc**2
        return convFactor*r_2*Grav_constant*M/(r_2+self.b_kpc**2)**(3./2.)
    
class Galactic_model:
    def __init__(self, *potentials):
        self._potentials = potentials
        for pot in potentials:
            assert isinstance(pot, Potential), "input *potentials must be any derived class of the class Potential"

    def circular_velocity_km_s(self, R_kpc):
        Vc2 = 0.0
        for pot in self._potentials:
            Vc2 += pot.squared_circular_velocity_km2_s2(R_kpc)
        return np.sqrt(Vc2)
        
#%%
if __name__ == '__main__':
    pass
        
        