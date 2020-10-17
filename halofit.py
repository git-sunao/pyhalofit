import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy import integrate
from scipy.optimize import fsolve

# Original paper : Takahashi et al. (2012)
# https://arxiv.org/abs/1208.2701
# DOI : 10.1088/0004-637X/761/2/152

class halofit:
    def __init__(self):
        self.cosmo = cosmo_util()
    
    def set_cosmology(self, cosmo_dict):
        """
        input dict of
            Omega_de0
            Omega_K0
            w0
            wa
            h
        Omega_m0 is computed internally as
            Omega_m0 = 1.0 - Omega_de0 - Omega_K0
        """
        self.cosmo.set_cosmology(cosmo_dict)

    def set_pklin(self, k, pklin, z, unit='h/Mpc'):
        if unit == 'h/Mpc':
            self.k, self.pklin = k * self.cosmo.h, pklin / self.cosmo.h**3
        elif unit == '/Mpc':
            self.k, self.pklin = k, pklin
        self.unit = unit
        self.z = z
        self.Delta_L = self.pklin*self.k**3/(2.*np.pi**2)
    
    def sigma(self, R):
        return integrate.simps(self.Delta_L*np.exp(-(self.k*R)**2), np.log(self.k))

    def _compute_R_sigma(self, z):
        # init guess : [1/Mpc]   Takada & Jain (2004) fiducial model
        k_sigma = 10.**(-1.065+4.332e-1*(1.+z)-2.516e-2*pow(1.+z,2)+9.069e-4*pow(1.+z,3))
        
        def eq(R):
            return self.sigma(R) - 1.0
        
        self.R_sigma = fsolve(eq, [1./k_sigma])
    
    def _compute_neff_C(self, h=1e-4):
        lnR_0 = np.log(self.R_sigma)
        lnR_p = lnR_0*(1.+h)
        lnR_m = lnR_0*(1.-h)
        
        lnsigma_0 = np.log(self.sigma(np.exp(lnR_0)))
        lnsigma_p = np.log(self.sigma(np.exp(lnR_p)))
        lnsigma_m = np.log(self.sigma(np.exp(lnR_m)))
        
        self.neff = -3. - (lnsigma_p-lnsigma_m)/(lnR_p-lnR_m)
        self.C = - (lnsigma_p - 2.*lnsigma_0 + lnsigma_m)/(lnR_p-lnR_0)**2
    
    def _compute_coeffs(self, z):
        neff, C, Omega_de, w = self.neff, self.C, self.cosmo.Omega_de(z), self.cosmo.w0
        self.an = 10.**( 1.5222 + 2.8553*neff + 2.3706*neff**2 + 0.9903*neff**3 + 0.2250*neff**4 
                - 0.6038*C + 0.1749*Omega_de*(1.+w) )
        self.bn = 10.**(-0.5642 + 0.5864*neff + 0.5716*neff**2 
                - 1.5474*C + 0.2279*Omega_de*(1.+w))
        self.cn = 10.**( 0.3698 + 2.0404*neff + 0.8161*neff**2 
                + 0.5869*C)
        self.gamman = 0.1971 - 0.0843*neff + 0.8460*C
        self.alphan = abs( 6.0835 + 1.3373*neff - 0.1959*neff**2 - 5.5274*C)
        self.betan  = 2.0379 - 0.7354*neff + 0.3157*neff**2 + 1.2490*neff**3 + 0.3980*neff**4 - 0.1682*C
        self.mun    = 0.
        self.nun    = 10.**(5.2105 + 3.6902*neff)

    def get_pkhalo(self):
        self._compute_R_sigma(self.z)
        self._compute_neff_C()
        self._compute_coeffs(self.z)

        y = self.k * self.R_sigma
        f = y/4. + y**2/8.
        Omz = self.cosmo.Omega_m(self.z)
        f1, f2, f3 = Omz**-0.0307, Omz**-0.0585, Omz**0.0743
        Delta_Q = self.Delta_L * ((1.+self.Delta_L)**self.betan)/(1.+self.alphan*self.Delta_L) * np.exp(-f)
        Delta_H = self.an*y**(3.*f1) / (1.+self.bn*y**f2 + (self.cn*y*f3)**(3.-self.gamman))
        Delta_H = Delta_H / (1. + self.mun/y + self.nun/y**2)
        pkhalo = (Delta_Q + Delta_H) * (2.*np.pi**2) / self.k**3 
        if self.unit == 'h/Mpc':
            return pkhalo * self.cosmo.h**3
        elif self.unit == '/Mpc':
            return pkhalo

class cosmo_util:
    def __init__(self, cosmo_dict=None):
        if cosmo_dict is None:
            cosmo_dict = {"Omega_de0":0.6844, "Omega_K0":0.0,"w0":-1.0,"wa":0.0,'h':0.6774}
        self.set_cosmology(cosmo_dict)

    def set_cosmology(self, cosmo_dict):
        """
        input dict of
            Omega_de0
            Omega_K0
            w0
            wa
            h
        Omega_m0 is computed internally as
            Omega_m0 = 1.0 - Omega_de0 - Omega_K0
        """
        self.Omega_de0 = cosmo_dict['Omega_de0']
        self.Omega_K0 = cosmo_dict['Omega_K0']
        self.w0 = cosmo_dict['w0']
        self.wa = cosmo_dict['wa']
        self.h = cosmo_dict['h']
        self.Omega_m0 = 1.0 - self.Omega_de0 - self.Omega_K0

    def Omega_de(self, z):
        de = self.de_func(z)
        a = 1./(1.+z)
        return self.Omega_de0*de / (self.Omega_m0/a**3 + self.Omega_K0/a**2 + self.Omega_de0*de)
    
    def Omega_m(self, z):
        de = self.de_func(z)
        a = 1./(1.+z)
        return self.Omega_m0/a**3 / (self.Omega_m0/a**3 + self.Omega_K0/a**2 + self.Omega_de0*de)

    def Omega_K(self, z):
        de = self.de_func(z)
        a = 1./(1.+z)
        return self.Omega_K0/a**2  / (self.Omega_m0/a**3 + self.Omega_K0/a**2 + self.Omega_de0*de)

    def de_func(self, z):
        w0, wa = self.w0, self.wa
        a = 1./(1.+z)
        return a**(-3.0*(1.+w0+wa)) * np.exp(-3.0*wa*(1.-a))
