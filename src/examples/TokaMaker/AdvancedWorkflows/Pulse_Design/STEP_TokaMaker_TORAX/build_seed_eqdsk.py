"""Solve the seed free-boundary equilibrium for the STEP EB-CC flat top.

Shape targets (separatrix and X-points) come from the OpenSTEP EB-CC
free-boundary equilibrium's own psi map; Ip/pax targets and FF'/p' profile
shapes come from the ftop equilibrium IDS. Writes `STEP_seed.eqdsk`.
"""
import os
import warnings

import imas
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, ndimage

import OpenFUSIONToolkit
from OpenFUSIONToolkit import TokaMaker
from OpenFUSIONToolkit.TokaMaker import meshing

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, 'data')
FTOP_FILE = os.path.join(DATA_DIR, 'STEP_SPP_001_EBCC_ftop.nc')
FREEBOUNDARY_FILE = os.path.join(DATA_DIR, 'STEP_SPP_001_EBCC_freeboundary.nc')
WALL_FILE = os.path.join(DATA_DIR, 'STEP_SPP_001_wall.nc')
MESH_FILE = os.path.join(HERE, 'STEP_mesh.h5')
EQDSK_FILE = os.path.join(HERE, 'STEP_seed.eqdsk')

# Constraint weights, matching the TokaMaker_TORAX re-solves.
SADDLE_WEIGHT = 1000.0
ISOFLUX_WEIGHT = 300.0
PSI_PIN_WEIGHT = 10.0 * ISOFLUX_WEIGHT


def load_freeboundary_separatrix(spacing=0.25):
    """Returns diverted separatrix targets from the free-boundary psi map.

    X-points are the saddle points (min |grad psi|) of the 2D psi map; the
    separatrix is the psi = psi_X contour clipped to the first wall, which
    includes the true curved divertor legs. Targets are exactly up-down
    symmetrized (the lower half of each group mirrors the upper half) to
    match the symmetric-coil-current constraint used in the solves.

    Returns 'x_points' (2, 2) plus arclength-downsampled point groups:
    'core' (|Z| below the tmtx re-solve isoflux trim), 'shoulders' (above
    it, up to the X-points), and 'legs' (beyond the X-points).
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        db = imas.DBEntry(FREEBOUNDARY_FILE, 'r')
        eq = db.get('equilibrium')
        p2d = eq.time_slice[0].profiles_2d[0]
        grid_r = np.array(p2d.grid.dim1)
        grid_z = np.array(p2d.grid.dim2)
        psi2d = np.array(p2d.psi)
        db.close()
        db = imas.DBEntry(WALL_FILE, 'r')
        unit = db.get('wall').description_2d[0].limiter.unit[0]
        wall_path = matplotlib.path.Path(
            np.column_stack([np.array(unit.outline.r), np.array(unit.outline.z)]))
        db.close()

    spline = interpolate.RectBivariateSpline(grid_r, grid_z, psi2d)
    x_points = []
    for guess in ((2.4, 5.9), (2.5, -6.0)):
        rg = np.linspace(guess[0] - 0.8, guess[0] + 0.8, 600)
        zg = np.linspace(guess[1] - 0.8, guess[1] + 0.8, 600)
        rr, zz = np.meshgrid(rg, zg)
        grad2 = spline.ev(rr, zz, dx=1) ** 2 + spline.ev(rr, zz, dy=1) ** 2
        k = np.unravel_index(np.argmin(grad2), grad2.shape)
        x_points.append((rr[k], zz[k]))
    x_points = np.array(x_points)
    psi_x = 0.5 * (spline.ev(*x_points[0]) + spline.ev(*x_points[1]))

    fig, ax = plt.subplots()
    contours = ax.contour(grid_r, grid_z, psi2d.T, levels=[float(psi_x)])
    segments = []
    for path in contours.get_paths():
        v, c = path.vertices, path.codes
        starts = ([0, len(v)] if c is None else
                  np.where(c == matplotlib.path.Path.MOVETO)[0].tolist() + [len(v)])
        for s, e in zip(starts[:-1], starts[1:]):
            if e - s > 5:
                segments.append(v[s:e])
    plt.close(fig)

    def downsample(points):
        if len(points) == 0:
            return np.empty((0, 2))
        keep = [0]
        acc = 0.0
        for j in range(1, len(points)):
            acc += np.hypot(*(points[j] - points[j - 1]))
            if acc >= spacing:
                keep.append(j)
                acc = 0.0
        return points[keep]

    x_r = x_points[:, 0].mean()
    x_z = np.abs(x_points[:, 1]).mean()
    x_points = np.array([[x_r, x_z], [x_r, -x_z]])

    def mirror(points):
        upper = points[points[:, 1] >= 0.0]
        lower = upper[upper[:, 1] > 1e-6] * np.array([1.0, -1.0])
        lower = lower[wall_path.contains_points(lower)]
        return np.vstack([upper, lower])

    z_x = np.abs(x_points[:, 1]).min()
    z_trim = 0.6 * np.abs(x_points[:, 1]).max()
    core, shoulders, legs = [], [], []
    for seg in segments:
        seg = seg[wall_path.contains_points(seg)]
        if len(seg) == 0:
            continue
        absz = np.abs(seg[:, 1])
        core.append(downsample(seg[absz <= z_trim]))
        shoulders.append(downsample(seg[(absz > z_trim) & (absz <= z_x)]))
        legs.append(downsample(seg[absz > z_x]))
    return {
        'x_points': x_points,
        'core': mirror(np.vstack(core)),
        'shoulders': mirror(np.vstack(shoulders)),
        'legs': mirror(np.vstack(legs)),
    }


def load_targets():
    """Returns Ip/pax/F0, boundary geometry, and FF'/p' shapes from the IDS data."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        db = imas.DBEntry(FTOP_FILE, 'r')
        eq = db.get('equilibrium')
        ts = eq.time_slice[0]
        b = ts.boundary
        p1d = ts.profiles_1d
        psi_prof = np.array(p1d.psi)
        vtf = eq.vacuum_toroidal_field
        target = {
            'R0': float(b.geometric_axis.r),
            'Z0': float(b.geometric_axis.z),
            'a': float(b.minor_radius),
            'kappa': float(b.elongation),
            'delta': float(0.5 * (b.triangularity_upper + b.triangularity_lower)),
            'Ip': float(ts.global_quantities.ip),
            'pax': float(np.array(p1d.pressure)[0]),
            'psi_norm': (psi_prof - psi_prof[0]) / (psi_prof[-1] - psi_prof[0]),
            'pprime': np.array(p1d.dpressure_dpsi),
            'ffprime': np.array(p1d.f_df_dpsi),
            'F0': float(vtf.r0) * float(np.array(vtf.b0)[0]),
        }
        db.close()
        db = imas.DBEntry(FREEBOUNDARY_FILE, 'r')
        gq = db.get('equilibrium').time_slice[0].global_quantities
        # IMAS psi [Wb] -> TokaMaker-native [Wb/rad], opposite sign convention.
        target['psi_lcfs_tm'] = -float(gq.psi_boundary) / (2.0 * np.pi)
        db.close()
    return target


def build_seed_equilibrium():
    """Solves the seed equilibrium and writes EQDSK_FILE."""
    target = load_targets()
    sep = load_freeboundary_separatrix()

    myOFT = OpenFUSIONToolkit.OFT_env(nthreads=4)
    mygs = TokaMaker.TokaMaker(myOFT)
    mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict = meshing.load_gs_mesh(MESH_FILE)
    mygs.setup_mesh(mesh_pts, mesh_lc, mesh_reg)
    mygs.setup_regions(cond_dict=cond_dict, coil_dict=coil_dict)
    mygs.settings.maxits = 400
    mygs.setup(order=2, F0=target['F0'])

    # Virtual vertical-stability coil (antisymmetric outboard P3 pair): the
    # kappa~3 equilibrium is vertically unstable, and with up-down-symmetric
    # coil currents the regular coil DOFs cannot provide the antisymmetric
    # radial field the nonlinear solve needs for vertical feedback.
    mygs.set_coil_vsc({'P3_COIL_U': 1.0, 'P3_COIL_L': -1.0})

    # Bounds are a total-current limit divided by turns (coil regions carry
    # up to a few hundred turns); not the binding constraint on the shape.
    coil_bounds = {}
    for name in mygs.coil_sets:
        nturns = coil_dict[name]['nturns'] if name in coil_dict else 1
        per_turn_limit = 150.0E6 / max(nturns, 1)
        coil_bounds[name] = [-per_turn_limit, per_turn_limit]
    mygs.set_coil_bounds(coil_bounds)

    # Weak min-norm regularization per coil, exact up-down symmetry per U/L
    # pair (naming assigned in build_STEP_mesh.py), VSC pinned to zero.
    reg_terms = []
    for name in mygs.coil_sets:
        if name == '#VSC':
            continue
        reg_terms.append(mygs.coil_reg_term({name: 1.0}, target=0.0, weight=1.0E-3))
        if name.endswith('_U') and name[:-2] + '_L' in mygs.coil_sets:
            reg_terms.append(mygs.coil_reg_term({name: 1.0, name[:-2] + '_L': -1.0},
                                                target=0.0, weight=1.0E3))
    reg_terms.append(mygs.coil_reg_term({'#VSC': 1.0}, target=0.0, weight=1.0E3))
    mygs.set_coil_reg(reg_terms=reg_terms)

    # Shape constraints: X-point saddles above isoflux in weight, plus an
    # absolute psi pin at the outboard midplane target point. The pin
    # anchors the flux value (from the free-boundary IDS) and, with the
    # strong weights, makes the solve converge from a cold ellipse start.
    iso_pts = np.concatenate([sep['core'], sep['shoulders'], sep['legs']])
    mygs.set_saddle_constraints(sep['x_points'],
                                weights=np.full(len(sep['x_points']), SADDLE_WEIGHT))
    mygs.set_isoflux_constraints(iso_pts, weights=np.full(len(iso_pts), ISOFLUX_WEIGHT))
    omp = iso_pts[np.argmax(iso_pts[:, 0] * np.exp(-0.5 * (iso_pts[:, 1] / (0.3 * target['a']))**2))]
    mygs.set_psi_constraints(omp.reshape(1, 2), targets=np.array([target['psi_lcfs_tm']]),
                             weights=np.array([PSI_PIN_WEIGHT]))

    # FF'/p' shapes from the ftop IDS (amplitudes are set by the Ip/pax
    # targets; this IMAS file's psi increases axis->edge, opposite to
    # TokaMaker, so d/dpsi profiles are negated). The IDS FF' is hollow --
    # its net current opposes p', which leaves TokaMaker's Ip scale factor
    # without usable leverage -- so FF' is clipped to >= 0 and p' is given
    # an equivalent core taper (to 40% at the axis over psi_N < 0.4; depth
    # chosen to best match the IDS q profile, q_min 2.56, within the range
    # the Ip/pax target system converges for), reproducing the scenario's
    # broad, bootstrap-dominated current.
    psi_u = np.linspace(0.0, 1.0, 100)
    pp_src = -target['pprime']
    pp_src[0] = pp_src[1]  # axis point is a grid artifact in the IDS
    pp_u = ndimage.gaussian_filter1d(np.interp(psi_u, target['psi_norm'], pp_src), 2, mode='nearest')
    ffp_u = ndimage.gaussian_filter1d(np.interp(psi_u, target['psi_norm'], -target['ffprime']), 2, mode='nearest')
    ffp_u = np.clip(ffp_u, 0.0, None)
    taper = 0.4 + 0.3 * (1.0 - np.cos(np.pi * np.clip(psi_u / 0.4, 0.0, 1.0)))
    pp_u = pp_u * taper
    mygs.set_profiles(ffp_prof={'type': 'linterp', 'x': psi_u, 'y': ffp_u / ffp_u.max()},
                      pp_prof={'type': 'linterp', 'x': psi_u, 'y': pp_u / pp_u.max()})
    mygs.set_resistivity(eta_prof={'type': 'linterp', 'x': psi_u, 'y': np.zeros_like(psi_u)})

    mygs.set_targets(Ip=target['Ip'], pax=target['pax'])
    mygs.init_psi(target['R0'], target['Z0'], target['a'], target['kappa'], target['delta'])
    mygs.solve()

    stats = mygs.get_stats()
    print(f"Solved: kappa={stats['kappa']:.3f} (target {target['kappa']:.3f}), "
          f"Ip={stats['Ip']/1e6:.2f} MA, pax={stats['P_ax']/1e3:.0f} kPa, "
          f"q_0={stats['q_0']:.2f}, q_95={stats['q_95']:.2f}, l_i={stats['l_i']:.2f}")

    mygs.save_eqdsk(EQDSK_FILE, cocos=2, nr=500, nz=500)
    print(f'Saved {EQDSK_FILE}')
    return mygs, target


if __name__ == '__main__':
    build_seed_equilibrium()
