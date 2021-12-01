import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

folder = "/export/data1/dbonan/CMIP5/piControl/"

# Constants 
ps = 98000     # surface pressure (kg m-1 s-2)
cp = 1005      # specific heat capacity at constant pressure (J kg-1 K-1)
RH = 0.8       # relative humidity (0-1)
Lv = 2257000   # latent heat of vaporization (J kg-1)

def get_data(var, institution, model, ID):
    ds = xr.open_dataset(f"{folder}{institution}{model}/{var}_Amon_{ID}_piControl_r1i1p1_300yrs.nc")
    return ds

def qsat(t, p):
    """
        qsat = qsat(t, p)

    Computes saturation specific humidity (qsat), given inputs temperature (t) in K and
    pressure (p) in hPa.
                            
    Buck (1981, J. Appl. Meteorol.)
    """
    tc=t-273.16
    tice=-23
    t0=0
    Rd=287.04
    Rv=461.5
    epsilon=Rd/Rv
    ewat=(1.0007+(3.46e-6*p))*6.1121*np.exp(17.502*tc/(240.97+tc))
    eice=(1.0003+(4.18e-6*p))*6.1115*np.exp(22.452*tc/(272.55+tc))
    eint=eice+(ewat-eice)*((tc-tice)/(t0-tice))**2
    esat=eint
    esat[np.where(tc<tice)]=eice[np.where(tc<tice)]
    esat[np.where(tc>t0)]=ewat[np.where(tc>t0)]
    qsat = epsilon*esat/(p-esat*(1-epsilon))
    return qsat

def get_h_Q(institution, model, ID):
    # get temp and radiation data
    ds = get_data("tas", institution, model, ID)
    lat = ds.lat
    tas = ds.tas.mean("time").mean("lon")
    
    ds = get_data("rlds", institution, model, ID)
    rlds = ds.rlds.mean("time").mean("lon")
    
    ds = get_data("rlus", institution, model, ID)
    rlus = ds.rlus.mean("time").mean("lon")
    
    ds = get_data("rlut", institution, model, ID)
    rlut = ds.rlut.mean("time").mean("lon")
    
    ds = get_data("rsds", institution, model, ID)
    rsds = ds.rsds.mean("time").mean("lon")
    
    ds = get_data("rsdt", institution, model, ID)
    rsdt = ds.rsdt.mean("time").mean("lon")
    
    ds = get_data("rsus", institution, model, ID)
    rsus = ds.rsus.mean("time").mean("lon")
    
    ds = get_data("rsut", institution, model, ID)
    rsut = ds.rsut.mean("time").mean("lon")

    ds = get_data("hfls", institution, model, ID)
    hfls = ds.hfls.mean("time").mean("lon")

    ds = get_data("hfss", institution, model, ID)
    hfss = ds.hfss.mean("time").mean("lon")
    
    # x, h, Q
    x = np.sin(lat*np.pi/180)
    h = cp*tas + Lv*RH*qsat(tas, ps/100)
    R_t = (rsdt - rsut) - rlut                        # top of atmosphere
    Q_s = (rsds - rsus) + (rlds - rlus) - hfls - hfss # net surface heat flux
    Q = R_t - Q_s                                     # net heating of atmosphere

    # Q without TOA LW
    Q_no_TOA_LW = Q + rlut
    
    # # plot if you want
    # plt.plot(x, Q)
    # plt.savefig(f"Q_{institution}-{model}.png")
    # plt.close()
    # plt.plot(x, h)
    # plt.savefig(f"h_{institution}-{model}.png")
    # plt.close()
    
    np.savez(f"h_Q_{institution}-{model}.npz", x=x, h=h, Q=Q, Q_no_TOA_LW=Q_no_TOA_LW)

institutions = ["ACCESS1",   "ACCESS1",   "CAN",     "CNRM",     "GFDL",     "GFDL",       "GFDL",       "INM",    "IPSL",         "IPSL",         "IPSL",         "MIROC",  "MIROC",     "MPI",        "MRI",       "NOR"]
models =       ["0",         "3",         "ESM2",    "CM5",      "CM3",      "ESM2G",      "ESM2M",      "CM4",    "CM5ALR",       "CM5AMR",       "CM5BLR",       "5",      "ESM",       "ESMLR",      "CGCM3",     "ESM1M"]
IDs =          ["ACCESS1-0", "ACCESS1-3", "CanESM2", "CNRM-CM5", "GFDL-CM3", "GFDL-ESM2G", "GFDL-ESM2M", "inmcm4", "IPSL-CM5A-LR", "IPSL-CM5A-MR", "IPSL-CM5B-LR", "MIROC5", "MIROC-ESM", "MPI-ESM-LR", "MRI-CGCM3", "NorESM1-M"]
for institution, model, ID in zip(institutions, models, IDs):
    print(ID)
    get_h_Q(institution, model, ID)
