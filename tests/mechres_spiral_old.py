import numpy as np
from optiks.old.optiks import optiks
from optiks.old.loss_functions import *
from optiks.old.utils import spiralTraj
from optiks.old.options import *

"""
This script designs a 3mm, 24cm FOV, R=2 spiral for the GE 3T UHP system minimizing power deposited in known mechanical
resonance bands, with a maximum duration of 16.5ms.
"""

# gpu device idx
# device = get_free_gpu()
device_idx = 4
device = torch.device(device_idx)

# Designing desired trajectory (Spiral)=================================================================================
# C = spiralTraj(22/3, 0.09)
C = spiralTraj(12, 0.3)

C_m = np.hstack((np.real(C), np.imag(C))).astype(float)

sys = "MAGNUS"

# Setting hardware options==============================================================================================
hw = HardwareOpts(
    g0 = 0,
    # gfin = 0,
    gmax = 5,
    smax = 20,
)

# Setting design options================================================================================================
Pthresh = 100*0.95
if sys == "UHP":
    r = 26.5
    c = 359e-6
    alpha = 0.37
    fedges = [[0.51, 0.575], [0.96, 1.06], [1.14, 1.26], [1.4, 1.56], [1.72, 1.9]]  # UHP
    hw.smax = 19.7
elif sys == "PREMIER":
    fedges = [[0.560, 0.620], [0.96, 1.310], [1.860, 1.950]]  # Premier
    fedges = [[0.550, 0.630], [0.96, 1.310], [1.850, 1.960], [4, np.inf]]  # Premier with band-limiting
    r = 23.4
    c = 334e-6
    alpha = 0.333
    hw.smax = 15
elif sys == "MAGNUS":
    fedges = [[0.590, 0.972], [1.1, 1.4], [1.6, 1.8]] #, [5, np.inf]]  # Premier with band-limiting
    r = 52.2
    c = 611e-6
    alpha = 0.324
    hw.gmax = 10
    hw.smax = 60
else:
    raise NotImplementedError(f"System {sys} not known.")
params = {
    'terms': [time_bound, slew_lim, freq_min],
    'bound': 9,
    'pns': [Pthresh, r, c, alpha],
    'frequency': fedges,
}
weights = {
    'time': 1e0,
    'slew': 1e1,
    'frequency': 4e3,
}

des = DesignOpts(params=params, weights=weights)


# Setting solver options================================================================================================
sv = SolverOpts(ds=5e-5, maxiter=10000, count=50, device=device) # TODO: 20k step

# Designing gradient waveforms==========================================================================================
output = optiks(C_m, hwopts=hw, dsopts=des, svopts=sv, plot=True)
C_v = output.Cnew
t_sf = output.t
g_sf = output.g
s_sf = output.s
g_usf = output.ginit
s_usf = output.sinit
g_last = output.g_last

# save output
torch.save(
    dict(
        C=C_v,
        t=t_sf,
        g=g_sf,
        s=s_sf,
        g_usf=g_usf,
        s_usf=s_usf,
        g_last=g_last
    ), 
    "data/mechres_spiral_old.pt",
)
print("done.")