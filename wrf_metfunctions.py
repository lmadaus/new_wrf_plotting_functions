#!/usr/bin/env python

import numpy as np
from scipy.ndimage.filters import gaussian_filter

def ms_to_kts_barb(U, V, XLAT=None, XLONG=None,m=None):
    """
    Converts wrfout u and v winds to wind barbs in knots
    If a basemap object (m) is supplied with lats and lons, will rotate
    the wind barbs to be correctly aligned with the map projection
    (I'm not entirely convinced this works correctly)
    
    Requires:
        U   => u-wind component (m/s, 2d)
        V   => v-wind component (m/s, 2d)
        XLAT (optional) => latitudes (2d)
        XLONG (optional)  => longitudes (2d)
        m (optional) => Basemap object
    Returns:
        Two 2d arrays of u and v components
        of the winds in kts (optionally rotated)

    """

    # Rotate winds with projection
    if m is not None:
        urot, vrot = m.rotate_vector(U,V,XLONG,XLAT)
    else:
        urot = U
        vrot = V
    return [[urot*1.94384, vrot*1.94384]]


def precipitable_water(QVAPOR, PB, P):
    """
    Computes total column precipitable water
    
    Requires:
        QVAPOR => water vapor mixing ratio (kg/kg, 3d)
        PB => WRF base-state pressure (Pa, 3d)
        P => WRF perturbation pressure (Pa, 3d)

    Returns:
        total column precipitable water (mm, 3d)
    """
    # Get total pressure
    ptot = PB + P
    # Need to calculate the change in pressure and average qv
    delP = np.diff(ptot, axis=0)
    nz,ny,nx = QVAPOR.shape
    avgQ = (QVAPOR[0:nz-1] + QVAPOR[1:nz]) / 2.0
    # Divide P by g
    delP = delP / 9.81
    # Now layer precipitable water is product
    layerPW  = -avgQ*delP
    # Sum along height axis to get precipitable water in mm
    return [np.sum(layerPW, axis=0)]

def plev_interp(VAR, PB, P, plev):
    """
    Function to interpolate arbitrary 3d variable
    to a given pressure level
    
    If the field is below the ground level or
    off the model top, will return np.nan there

    Requires:
        VAR => Arbitrary 3d variable
        PB => WRF base-state pressure (Pa, 3d)
        P  => WRF perturbation pressure (Pa, 3d)
        plev => pressure level to interpolate to (hPa, float)
    Returns:
        2d array of "var" interpolated to desired pressure level

    """
    
    # Total pressure
    ptot  = PB + P
    
    # Approximate the height of the given plev
    # By finding the index in the z dimension we first
    # cross it in each column
    above = np.argmax(ptot < plev*100., axis=0)
    # Get the index below that too
    below = above - 1
    # If we are below ground, set to zero
    below[below < 0] = 0

    nz,ny,nx = ptot.shape
    # Get the P values at these two above and below levels
    aboveP = np.log(ptot.reshape(nz,ny*nx)[above.flatten(),xrange(ny*nx)].reshape(ny,nx))
    belowP = np.log(ptot.reshape(nz,ny*nx)[below.flatten(),xrange(ny*nx)].reshape(ny,nx))
    # For a linear interpolation, weight by distance from plev
    distAbove = np.abs(aboveP - np.log(plev*100))
    distBelow = np.abs(belowP - np.log(plev*100))
    total_dist = distAbove + distBelow
    weightAbove = 1.0 - (distAbove/total_dist)
    weightBelow = 1.0 - weightAbove
   
    # Now grab var at these two levels, weight with our
    # calculated weights, and add to get interpolated value
    nz,ny,nx = VAR.shape
    varflat = VAR.reshape(nz, ny*nx)
    aboveV = varflat[above.flatten(),xrange(ny*nx)].reshape(ny,nx)
    belowV = varflat[below.flatten(),xrange(ny*nx)].reshape(ny,nx)
    final = aboveV * weightAbove + belowV * weightBelow
    # Anywhere we were below ground or above model top is np.nan
    final[above==0] = np.nan

    # Optionally, to verify that the interpolation works, uncomment
    # the following two lines which will print the average pressure
    # level from the interpolation as compared to what was requested
    #finalP = aboveP * weightAbove + belowP * weightBelow
    #print(" Requested P:", plev*100., "Mean Interp. P:", np.nanmean(np.exp(finalP)))
    
    return final


def melting_level(T, PB, P, PHB, PH, TBASE=300.0):
    """ 
    Find the melting (0C) level in feet above ground
    Specifically, will find the HIGHEST point at which
    the temperature profile crosses below 0C (could be
    warm levels below this)

    Requires:
        T => potential temperature (K, 3d)
        PB => WRF Base-state pressure (Pa, 3d)
        P => Perturbation pressure (Pa, 3d)
        PHB => Base-state geopotential (m^2/s^2, 3d)
        PH => Perturbation geopotential (m^2/s^2, 3d)
        TBASE (optional) => If using a different base-
                            state potential temperature
                            than 300K, specify here (K, float)

    Returns:
        2d array of 0C level in feet.  Areas below ground will be 0.0
    """

    # Like interpolating height to a certain temperature level
    # First, convert theta to raw t
    T = (TBASE + T) * ((PB+P)/100000.) ** (287.04/(7.*287.04/2.))
    # And geopotential to geopotential height
    phtot = (PH + PHB) / 9.81
    # Now need the level of 273 K
    # Approximate the height of this level
    # Reverse the first axis here to search from the top
    nz,ny,nx = T.shape
    below = np.argmax(T[::-1,:,:] > 273., axis=0)
    # Now convert this to a bottom-top index
    below = nz - below
    below -= 1
    below[below < 0] = 0
    below[below >= nz-1] = 0
    above = below + 1
    above[above >= nz] = nz-1
    #nz,ny,nx = T.shape
    # Get the PH values at these levels
    aboveT = T.reshape(nz,ny*nx)[above.flatten(),xrange(ny*nx)].reshape(ny,nx)
    belowT = T.reshape(nz,ny*nx)[below.flatten(),xrange(ny*nx)].reshape(ny,nx)
    # For a linear interpolation, weight by distance from 273K
    distAbove = np.abs(aboveT - 273.)
    distBelow = np.abs(belowT - 273.)
    total_dist = distAbove + distBelow
    weightAbove = 1.0 - (distAbove/total_dist)
    weightBelow = 1.0 - weightAbove
    # Now interpolate phtot to this level
    nz,ny,nx = phtot.shape
    varflat = phtot.reshape(nz, ny*nx)
    aboveV = varflat[above.flatten(),xrange(ny*nx)].reshape(ny,nx)
    belowV = varflat[below.flatten(),xrange(ny*nx)].reshape(ny,nx)
    final = aboveV * weightAbove + belowV * weightBelow
    # Areas below ground are set to 0
    final[np.bitwise_or(above==0, below==0)] = 0.
    #print(np.min(final,axis=None), np.max(final, axis=None))
    # Return in feet
    return [final * 3.28084]




def plev_vorticity(U, V, XLAT, PB, P, plev=500.0, dx=36000.0):
    """
    Computes absolute vorticity from interpolations
    to specified pressure level
    
    Requires:
        U => u-wind component (m/s, 3d)
        V => v-wind component (m/s, 3d)
        XLAT => Latitudes (deg, 2d)
        PB => base-state pressure (Pa, 3d)
        P => perturbation pressure (Pa, 3d)
        plev => desired pressure level (hPa, float)
        dx => model grid spacing (m, float)

    Returns:
        absolute vorticity on pressure level (s^-1, 2d)
        Areas below ground will be np.nan
    """
    # Unstagger winds
    u_unstaggered = 0.5 * (U[:,:,:-1] + U[:,:,1:])
    v_unstaggered = 0.5 * (V[:,:-1,:] + V[:,1:,:])
    # Interpolate to pressure level
    u_interp = plev_interp(u_unstaggered, PB, P, plev)
    v_interp = plev_interp(v_unstaggered, PB, P, plev)
    # Compute coriolis component
    fc = 1.458E-4 * np.sin(XLAT*np.pi/180.)
    # Do our gradients
    dvdx = np.gradient(v_interp,dx,dx)[1]
    dudy = np.gradient(u_interp,dx,dx)[0]
    # Compute vorticity
    avort = dvdx - dudy + fc
    return [avort]

def plev_temp(T, PB, P, plev, TBASE=300.0):
    """
    Interpolates temperature field to a 
    specified pressure level

    Requires:
        T => perturbation potential temperature (K, 3d)
        PB => base-state pressure (Pa, 3d)
        P => perturbation pressure (Pa, 3d)
        TBASE (optional) => base-state potential temp if not
                            300K

    Returns:
        Actual temperature interpolated to desired P level in Celsius
        Areas below ground are np.nan
    """
    # Back out standard temperature from potential temp
    temp = (TBASE + T) * ((PB+P)/100000.) ** (287./1004.)
    out = plev_interp(temp, PB, P, plev) - 273.
    return [out]

def plev_rh(T, PB, P, QVAPOR, plev, TBASE=300.0):
    """ 
    Compute relative humidity on the given pressure level

    Requires:
        T => perturbation potential temperature (K, 3d)
        PB => Base-state pressure (Pa, 3d)
        P => perturbation pressure (Pa, 3d)
        QVAPOR => water vapor mixing ratio (kg/kg, 3d)
        plev => Desired pressure level (hPa, float)
        TBASE (optional) => base-state potential temperature
                            if different from 300K

    """
    # Get the temperature on this plevel
    temp_lev = plev_temp(T, PB, P, plev, TBASE)
    # Back to Kelvin
    t = temp_lev[0] + 273.0

    # Get the actual pressure in the interpolation
    # To take into account small errors
    press_lev = plev_interp((PB+P)/100.0,PB,P,plev)
    press_lev = press_lev

    # Interpolate moisture to this level
    q_lev = plev_interp(QVAPOR, PB, P, plev)
    qv = q_lev

    # Now compute the saturation mixing ratio
    es = 6.11 * np.exp(5423.0 * (1.0/273.15 - 1.0/T))
    qs = 0.622 * es / (plev - es)
    
    # Remove supersaturation
    qv[qv>qs] = qs[qv>qs]

    # RH is just the fraction of saturation
    RH = qv / qs * 100.

    return[RH]
    




def plev_wind(U, V, PB, P, plev):
    """
    Interpolate wind to given pressure level
    Returning wind SPEED
    Requires:
        U => u-wind component (m/s, 3d)
        V => v-wind component (m/s, 3d)
        PB => Base-state pressure (Pa, 3d)
        P => perturbation pressure (Pa, 3d)
        plev => desired pressure level (hPa, float)
    Returns:
        WIND MAGNITUDE in knots
        Below ground values are np.nan
    
    """
    # Unstagger winds
    u_unstaggered = 0.5 * (U[:,:,:-1] + U[:,:,1:])
    v_unstaggered = 0.5 * (V[:,:-1,:] + V[:,1:,:])
    # Interpolate to pressure level 
    u_interp = plev_interp(u_unstaggered, PB, P, plev)
    v_interp = plev_interp(v_unstaggered, PB, P, plev)
    # Compute wind magnitude
    out = np.sqrt(u_interp**2 + v_interp**2) * 1.94384
    return [out]

def plev_wind_barb(U, V, PB, P, plev, XLAT=None, XLONG=None, m=None):
    """
    Interpolate wind to given pressure level
    Returning wind COMPONENTS
    If XLAT, XLONG and m are given, will rotate to map projection
    (again, not sure if this is working correctly)
    Requires:
        U => u-wind component (m/s, 3d)
        V => v-wind component (m/s, 3d)
        PB => Base-state pressure (Pa, 3d)
        P => perturbation pressure (Pa, 3d)
        plev => desired pressure level (hPa, float)
        XLAT (optional) => latitudes (deg, 2d)
        XLONG (optional)  => longitudes (deg, 2d)
        m (optional) => Basemap object

    Returns:
        Two 2d arrays of u and v wind in knots
        Below ground values are np.nan
    
    """
    # Unstagger winds
    u_unstaggered = 0.5 * (U[:,:,:-1] + U[:,:,1:])
    v_unstaggered = 0.5 * (V[:,:-1,:] + V[:,1:,:])
    # Interpolate to pressure level
    u_interp = plev_interp(u_unstaggered, PB, P, plev)
    v_interp = plev_interp(v_unstaggered, PB, P, plev)
    if m is not None:
        urot,vrot = m.rotate_vector(u_interp, v_interp, XLONG, XLAT)
    else:
        urot = u_interp
        vrot = v_interp
    return [[urot * 1.94384, vrot * 1.94384]]

def plev_height(PHB, PH, PB, P, PSFC, HGT, T, QVAPOR, plev, TBASE=300.0):
    """
    Interpolate geopotential height of a given pressure level
    This is the one interpolation where it will attempt to
    fill in values below ground following WRF-POST algorithm

    Requires:
        PHB => Base-state geopotential (m^2/s^2, 3d)
        PH  => Perturbation geopotential (m^2/s^2, 3d)
        PB  => Base-state pressure (Pa, 3d)
        P   => Perturbation pressure (Pa, 3d)
        PSFC => Surface pressure (Pa, 2d)
        HGT => Surface elevation (m, 2d)
        T => Perturbation potential temperature (K, 3d)
        QVAPOR => water vapor mixing ratio (kg/kg, 3d)
        plev => Requested pressure level (hPa, float)
        TBASE (optional) => Base-state potential temp if not 300K

    Returns:
        Geopotential height of requested pressure level, interpolated
        below ground if requested and smoothed with Gaussian filter

    """
    # Convert to geopotential height 
    ght = (PH + PHB) / 9.81
    nz,ny,nx = ght.shape
    # Unstagger geopotential height
    ght[1:nz-1,:,:] = 0.5 * (ght[1:nz-1,:,:] + ght[2:nz,:,:])
    # Interpolate geopotential height to pressure level
    out = plev_interp(ght, PB, P, plev)

    # Now fill underground
    pinterp = plev*100.
    ptot = pb+p
    ghtbase = ght[0,:,:]
    pbase = ptot[0,:,:]
    # First find the model level that is closest to a target pressure
    # level, where the target is delta-p less than the local
    # value of a horizontally smoothed surface pressure field.  We
    # use delta-p = 150hPa here.  A standard lapse rate temperature
    # profile passing through the temperature at this model level will
    # be used to define the temperature profile below ground
    # This follows algorithm from WRF-POST
    locs = np.isnan(out)
    if np.sum(locs,axis=None) != 0:
        #print "numnan:", np.sum(locs,axis=None)
        #print "psfc:", psfc.shape
        #print "locs:", locs.shape
        #print "psfc[locs]:", psfc[locs].shape
        ptarget = psfc - 15000.
        # Find pressure value that is first above this
        vert_levs = np.argmax(ptot < ptarget, axis=0)
        # Convert to actual temperature
        T = (300+t) * (ptot/100000.) ** (287.04/1004.)
        nz,ny,nx = T.shape
        Tupper = T.reshape(nz,ny*nx)[vert_levs.flatten(),xrange(ny*nx)].reshape(ny,nx)[locs]
        Pupper = ptot.reshape(nz,ny*nx)[vert_levs.flatten(),xrange(ny*nx)].reshape(ny,nx)[locs]
        pbot = np.maximum(ptot[0],psfc)[locs]
        zbot = np.minimum(ght[0],hgt)[locs]
        expon = 287.04*0.0065/9.81
        tbotextrap = Tupper * (pbot/Pupper)**expon
        tvbotextrap = virtual(tbotextrap,qv[0][locs])
        out[locs] = (zbot + tvbotextrap/0.0065*(1.-(plev*100./pbot)**expon))

    # Return a smoothed field (change smoothing by changing sigma, 3 is reasonable)
    return [gaussian_filter(out, sigma=3)]

def virtual(TEMP,QVAPOR):
    """ 
    Returns virtual temperature given
    actual temperature and mixing ratio
    
    Requires:
        TEMP => Temperature (C)
        QVAPOR => water vapor mixing ratio (kg/kg)
    Returns:
        2d array of virtual temperature
    """

    return TEMP * (0.622+QVAPOR) / (0.622 * (1.0 + QVAPOR))

def wind_speed_kts(U,V):
    """
    Returns wind speed in knots given u and v wind components
    
    Requires:
        U => u-wind component (m/s)
        V => v-wind component (m/s)
    Returns:
        wind speed magniutde in knots
    """
    return [np.sqrt(np.power(U,2) + np.power(V,2)) * 1.94384]

def olr_to_temp(OLR):
    """
    Converts OLR to radiating temperature in Celisus
    by the Stefan-Boltzmann law

    Requires:
        OLR => Outgoing longwave radiation (W/m^2)
    Returns:
        temperature in Celsius
    """
    return [np.power(OLR / 5.67E-8, 0.25) - 273.]
 
def K_to_F(TEMP):
    """
    Given temperature in Kelvin, return in Fahrenheit

    Requires:
        TEMP =>  Temperature (K)
    Returns:
        Temperature in Fahrenheit
    """

    return [(TEMP - 273.) * 9./5. + 32.]

def mm_to_in(vars):
    """
    Sums precipitation fields in mm and converts to inches

    Requires:
        vars => list of precipitation variables in mm
    Returns:
        sum of all components of vars in inches

    """
    return [np.sum(np.array(vars) * 0.03937, axis=0)] 


def merged_precip(vars):
    """
    Given a list of precipitation variables, add them together
    Requires:
        vars => list of precipitation variables
    Returns:
        sum of all components of vars
    """

    return [np.sum(np.array(vars), axis=0)]


def altimeter(PSFC, HGT):
    """
    Compute altimeter setting
    Requires:
        PSFC => surface pressure (Pa, 2d)
        HGT => surface elevation (m, 2d)
    Returns:
        Smoothed altimeter setting field (hPa)

    """
    PSFC = PSFC / 100.
    alt = ((PSFC - 0.3) ** (0.190284) + 8.4228807E-5 * HGT) ** (1.0/0.190284)
    # Smooth this field
    return [gaussian_filter(alt, sigma=6)]

def slp(PB, P, PHB, PH, T, QVAPOR, TBASE=300.0):
    """
    Compute sea-level pressure following the WRF-POST algorithm
    
    Requires:
        PB  => Base-state pressure (Pa, 3d)
        P   => Perturbation pressure (Pa, 3d)
        PHB => Base-state geopotential (m^2/s^2, 3d)
        PH  => Perturbation geopotential (m^2/s^2, 3d)
        T   => Perturbation potential temperature (K, 3d)
        QVAPOR => Water vapor mixing ratio (kg/kg, 3d)
        TBASE (optional) => Base-state potential temperature
                            if different from 300K
    Returns:
        smoothed sea-level pressure (hPa, 2d)

    """
    # Quick conversions to full pressure and actual temperature
    ptot = P + PB
    t = (300+T) * (ptot/100000.) ** (287.04/1004.)
    # compute geopotential
    ph = (PHB + PH)/9.81
    nz,ny,nx = ph.shape
    # Unstagger geopotential
    ph[1:nz-1,:,:] = 0.5 * (ph[1:nz-1,:,:] + ph[2:nz,:,:])

    # These constants are from WRF-POST
    TC = 273.16+17.5
    PCONST = 10000.
    # Find lowest level that is at least PCONST above the surface
    klo = np.argmax(ptot < ptot[0]-PCONST, axis=0) - 1
    
    # Get the temperature, pressure and moisture at this level
    # and the level above
    klo[(klo-1 < 0)] = 0
    khi = klo+1
    nz,ny,nx = t.shape
    Tlo = t.reshape(nz,ny*nx)[klo.flatten(),xrange(ny*nx)].reshape(ny,nx)
    Plo = ptot.reshape(nz,ny*nx)[klo.flatten(),xrange(ny*nx)].reshape(ny,nx)
    Qlo = QVAPOR.reshape(nz,ny*nx)[klo.flatten(),xrange(ny*nx)].reshape(ny,nx)
    Thi = t.reshape(nz,ny*nx)[khi.flatten(),xrange(ny*nx)].reshape(ny,nx)
    Phi = ptot.reshape(nz,ny*nx)[khi.flatten(),xrange(ny*nx)].reshape(ny,nx)
    Qhi = QVAPOR.reshape(nz,ny*nx)[khi.flatten(),xrange(ny*nx)].reshape(ny,nx)
    nz,ny,nx = ph.shape
    Zhi = ph.reshape(nz,ny*nx)[khi.flatten(),xrange(ny*nx)].reshape(ny,nx)
    Zlo = ph.reshape(nz,ny*nx)[klo.flatten(),xrange(ny*nx)].reshape(ny,nx)
    # Virtual temperature correction
    Tlo = Tlo * (1.0+0.608 * Qlo)
    Thi = Thi * (1.0+0.608 * Qhi)

    p_at_pconst = ptot[0] - PCONST
    t_at_pconst = Thi-(Thi-Tlo)*np.log(p_at_pconst/Phi)*np.log(Plo/Phi)
    z_at_pconst = Zhi-(Zhi-Zlo)*np.log(p_at_pconst/Phi)*np.log(Plo/Phi)

    t_surf = t_at_pconst * (ptot[0]/p_at_pconst) ** (0.0065*287.04/9.81)
    t_sea_level = t_at_pconst + 0.0065*z_at_pconst

    # They call this the "Ridiculous MM5 test" for too hot temperatures
    l1 = t_sea_level < TC
    l2 = t_surf <= TC
    locs = np.bitwise_and(l2,~l1)
    t_sea_level = TC - 0.005*(t_surf-TC)**2
    t_sea_level[locs] = TC

    # Now final computation
    z_half_lowest = ph[0]
    out = ptot[0] * np.exp((2.*9.81*z_half_lowest)/\
                        (287.04*(t_sea_level+t_surf)))
    #print np.max(out/100.,axis=None), np.min(out/100.,axis=None)
    # Smooth the output field and return in hPa
    return [gaussian_filter(out/100., sigma=3)]



def composite_reflectivity(QRAIN, QSNOW, TEMP, PSFC):
    """ Vars come in order QRAIN, QSNOW, TEMP, PSFC """
    rhor = 1000.
    rhos = 100.
    rhog = 400.
    rhoi = 917.

    # Define fixed intercepts
    Norain = 8.0E6
    Nosnow = 2.0E6 * np.exp(-0.12 * (T - 273.))
    Nograu = 4.0E6
    # Define surface density
    density = np.divide(PSFC, (287.*T))

    # Find maximum QR and QS
    Qra = np.max(QR, axis=0)
    Qsn = np.max(QS, axis=0)

    # Caclulate slope factor
    lambr = np.divide((np.pi * Nosnow * rhor), np.multiply(density, Qra)) ** 0.25
    lambs = np.exp(-0.0536 * (T - 273))

    # Calculate equivalent reflectivity factor
    Zer = (720.0 * Norain *(lambr ** -7.0)) * 1E18
    #Zes = (0.224 * 720.0 * Nosnow * (lambr ** -7.0) * (rhos/rhoi) ** 2) * 1E18
    Zes_int = np.divide((lambs * Qsn * density), Nosnow)
    Zes = ((0.224 * 720. * 1E18)/(np.pi * rhor) ** 2) * Zes_int ** 2

    Ze = np.add(Zer, Zes)
    # Convert to dBZ
    dBZ = 10* np.log10(Ze)
    dBZ = np.nan_to_num(dBZ)
    return [dBZ]



