import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy import integrate
from scipy.optimize import fsolve
from astropy.cosmology import w0waCDM

# Original paper : Takahashi et al. (2012)
# https://arxiv.org/abs/1208.2701
# DOI : 10.1088/0004-637X/761/2/152

class halofit:
    """halofit class

    This is the class of halofit, which is the fitting formula developed in Takahashi et al. (2012)

    """
    def __init__(self):
        self.cosmo = w0waCDM(70.0, 0.3, 0.7)

    def get_cosmology(self):
        """
        Get current cosmological paramters.

        Returns
        -------
        cosmo_dict : dict
            ditionary of cosmological parameters.
        """
        return {"Omega_de0":self.cosmo.Ode0, "Omega_K0":self.cosmo.Ok0, "w0":self.cosmo.w0, "wa":self.cosmo.wa, "h":self.cosmo.h}
    
    def set_cosmology(self, cosmo_dict):
        """Settig cosmological parameters

        Parameters
        ----------
            cosmo_dict : dict
                dictionary of cosmological parameters.
                cosmo_dict needs to include, Omega_de0, Omega_K0, w0, wa, h.
                Omega_m0 is computed internally by Omega_m0 = 1.0 - Omega_de0 - Omega_K0
        """
        Omega_de0 = cosmo_dict['Omega_de0']
        Omega_m0  = 1.0 - cosmo_dict['Omega_de0'] - cosmo_dict['Omega_K0']
        w0 = cosmo_dict['w0']
        wa = cosmo_dict['wa']
        H0 = 100.0*cosmo_dict['h']
        self.cosmo = w0waCDM(H0, Omega_m0, Omega_de0, w0=w0, wa=wa)

    def set_pklin(self, k, pklin, z, unit='h/Mpc'):
        """Setting Fourier mode and linear matter power spectrum at desired redshift.

        Parameters
        ----------
            k : ndarray
                Array of Fourier modes
            pklin : ndarray
                Array of linear matter power spectrum at redshift z
            z : float
                Cosmological redshift
            unit : str
                Unit of Fourier mode. '/Mpc' and 'h/Mpc' are available.
                Default is 'h/Mpc'.
        """
        if unit == 'h/Mpc':
            self.k, self.pklin = k * self.cosmo.h, pklin / self.cosmo.h**3
        elif unit == '/Mpc':
            self.k, self.pklin = k, pklin
        self.unit = unit
        self.z = z
        self.Delta_L = self.pklin*self.k**3/(2.*np.pi**2)
    
    def sigma(self, R):
        """Computing the variance of linear matter power spectrum.

        .. math:
            \\sigma(R) \\equiv \\int_0^\\infty {\\rm d}\\ln k \\Delta_{\\rm L}^2(k)e^{-k^2R^2}

        Parameters
        ----------
        R : float
            Smoothing scale to compute the variance of linear power spectrum.

        Returns
        -------
        sigma : float
            variance of linear power smoothed at scale :math:`R`.

        """
        return integrate.simps(self.Delta_L*np.exp(-(self.k*R)**2), np.log(self.k))

    def _compute_R_sigma(self, z):
        """Computing R satisfying 

        .. math:
            \\sigma(R) \\equiv \\int_0^\\infty {\\rm d}\\ln k \\Delta_{\\rm L}^2(k)e^{-k^2R^2}=1.

        Parameters
        ----------
        z : float
            cosmological redshift.

        Returns
        -------
        R : float
            :math:`R` satisfying :math:`\\sigma(R)=1`.
        """
        if self.sigma(0.0) < 1:
            print('Warning pyhalofit: sigma(0)<0 for z=%f. Set R_sigma=%f.'%(self.z, 0.01/self.k.max()))
            self.R_sigma = 0.01/self.k.max()
        else:
            # init guess : [1/Mpc]   Takada & Jain (2004) fiducial model
            k_sigma = 10.**(-1.065+4.332e-1*(1.+z)-2.516e-2*pow(1.+z,2)+9.069e-4*pow(1.+z,3))
            
            def eq(R):
                return self.sigma(R) - 1.0
            
            self.R_sigma = abs(fsolve(eq, [1./k_sigma]))
    
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
        neff, C, Omega_de, w = self.neff, self.C, self.cosmo.Ode(z), self.cosmo.w0
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
        """Getting halofit power spectrum.

        Returns
        -------
        pkhalo : ndarray
            halofit power spectrum.

        """
        self._compute_R_sigma(self.z)
        self._compute_neff_C()
        self._compute_coeffs(self.z)
        Omz = self.cosmo.Om(self.z)
        pkhalo = _pkhalo(self.k, self.Delta_L, self.z, self.R_sigma, Omz, self.an, self.bn, self.cn, self.gamman, self.alphan, self.betan, self.mun, self.nun)
        if self.unit == 'h/Mpc':
            return pkhalo * self.cosmo.h**3
        elif self.unit == '/Mpc':
            return pkhalo

    def _coeff_interps(self, k, pklintable, z):
        R_sigma = np.empty(z.size)
        an      = np.empty(z.size)
        bn      = np.empty(z.size)
        cn      = np.empty(z.size)
        gamman  = np.empty(z.size)
        alphan  = np.empty(z.size)
        betan   = np.empty(z.size)
        mun     = np.empty(z.size)
        nun     = np.empty(z.size)
        Omz     = np.empty(z.size)

        for i in range(z.size):
            self.set_pklin(k, pklintable[:,i], z[i])
            self._compute_R_sigma(self.z)
            self._compute_neff_C()
            self._compute_coeffs(self.z)

            R_sigma[i] = self.R_sigma
            Omz[i]     = self.cosmo.Om(self.z)
            an[i]      = self.an
            bn[i]      = self.bn
            cn[i]      = self.cn
            gamman[i]  = self.gamman
            alphan[i]  = self.alphan
            betan[i]   = self.betan
            mun[i]     = self.mun
            nun[i]     = self.nun

        interps = [ius(z, coeff) for coeff in [R_sigma, Omz, an, bn, cn, gamman, alphan, betan, mun, nun]]
        return interps

    def pklintable2pkhalotable(self, k, z, pklintable, dz_min=0.1):
        di = int(dz_min/np.diff(z)[0])
        i_sub = np.arange(0, z.size, di)
        pklintable_sub = pklintable[:, i_sub]
        z_sub = z[i_sub]
        interps = self._coeff_interps(k, pklintable_sub, z_sub)
        
        pkhalotable = np.empty(pklintable.shape)
        for i in range(z.size):
            Delta_L = pklintable[:,i]*k**3/(2.*np.pi**2)
            R_sigma = interps[0](z[i])
            Omz     = interps[1](z[i])
            an      = interps[2](z[i])
            bn      = interps[3](z[i])
            cn      = interps[4](z[i])
            gamman  = interps[5](z[i])
            alphan  = interps[6](z[i])
            betan   = interps[7](z[i])
            mun     = interps[8](z[i])
            nun     = interps[9](z[i])
            pkhalo = _pkhalo(k, Delta_L, z[i], R_sigma, Omz, an, bn, cn, gamman, alphan, betan, mun, nun)
            pkhalotable[:,i] = pkhalo

        return pkhalotable

def _pkhalo(k, Delta_L, z, R_sigma, Omz, an, bn, cn, gamman, alphan, betan, mun, nun):
    y = k * R_sigma
    f = y/4. + y**2/8.
    f1, f2, f3 = Omz**-0.0307, Omz**-0.0585, Omz**0.0743
    Delta_Q = Delta_L * ((1.+Delta_L)**betan)/(1.+alphan*Delta_L) * np.exp(-f)
    Delta_H = an*y**(3.*f1) / (1.+bn*y**f2 + (cn*y*f3)**(3.-gamman))
    Delta_H = Delta_H / (1. + mun/y + nun/y**2)
    pkhalo = (Delta_Q + Delta_H) * (2.*np.pi**2) / k**3 
    return pkhalo
