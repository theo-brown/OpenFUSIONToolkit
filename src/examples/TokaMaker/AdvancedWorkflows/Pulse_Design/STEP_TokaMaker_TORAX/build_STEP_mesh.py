"""Build a TokaMaker free-boundary mesh for STEP.

First-wall contour (`wall` IDS) and PF/CS coil winding packs (`pf_active`
IDS, one bounding rectangle per coil) come from the OpenSTEP SPP-001 EB-CC
free-boundary dataset (https://github.com/ukaea/OpenSTEP,
DOI: 10.14468/07jt-s540).
"""
import os
import warnings

import imas
import numpy as np

from OpenFUSIONToolkit.TokaMaker import meshing

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
WALL_FILE = os.path.join(DATA_DIR, 'STEP_SPP_001_wall.nc')
FREEBOUNDARY_FILE = os.path.join(DATA_DIR, 'STEP_SPP_001_EBCC_freeboundary.nc')
MESH_FILE = os.path.join(os.path.dirname(__file__), 'STEP_mesh.h5')

# Mesh resolution targets [m]
PLASMA_RESOLUTION = 0.12
COIL_RESOLUTION = 0.15
VAC_RESOLUTION = 0.7


def load_wall_contour():
    """Returns the first-wall outline from the wall IDS."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        db = imas.DBEntry(WALL_FILE, 'r')
        unit = db.get('wall').description_2d[0].limiter.unit[0]
        r = np.array(unit.outline.r)
        z = np.array(unit.outline.z)
        db.close()
    return np.column_stack([r, z])


def load_coil_geometry():
    """Returns {coil_name: {'rc','zc','w','h','nturns'}} from the pf_active IDS.

    Each coil is collapsed to the bounding rectangle of its filament
    elements, with one turn per filament.
    """
    coils = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        db = imas.DBEntry(FREEBOUNDARY_FILE, 'r')
        pf = db.get('pf_active')
        for i in range(len(pf.coil)):
            c = pf.coil[i]
            name = str(c.name)
            rs = np.array([e.geometry.rectangle.r for e in c.element])
            zs = np.array([e.geometry.rectangle.z for e in c.element])
            nturns = len(c.element)
            rc = 0.5 * (rs.min() + rs.max())
            zc = 0.5 * (zs.min() + zs.max())
            # Guard against degenerate (zero-width or zero-height) winding
            # packs, e.g. single-column solenoid coils.
            w = max(rs.max() - rs.min(), COIL_RESOLUTION)
            h = max(zs.max() - zs.min(), COIL_RESOLUTION)
            coils[name] = {'rc': rc, 'zc': zc, 'w': w, 'h': h, 'nturns': nturns}
        db.close()

    # Rename each up-down coil pair (pf_active suffixes the mirrored partner
    # with '_r') to TokaMaker_TORAX's U/L convention, assigning U/L from the
    # coil center height, so set_TokaMaker_coil_reg(updownsym=True) can
    # enforce up-down-symmetric coil currents.
    renamed = {}
    lower_names = {k.lower() for k in coils}
    for name, coil in coils.items():
        is_reflected = name.lower().endswith('_r')
        base = name[:-2] if is_reflected else name
        partner = base.lower() if is_reflected else base.lower() + '_r'
        if partner in lower_names:
            suffix = '_U' if coil['zc'] > 0 else '_L'
            renamed[base + suffix] = coil
        else:
            renamed[name] = coil
    assert len(renamed) == len(coils), sorted(renamed)
    return renamed


def build_mesh():
    """Builds and saves the TokaMaker mesh; returns the mesh objects."""
    wall_contour = load_wall_contour()
    coils = load_coil_geometry()

    gs_mesh = meshing.gs_Domain()
    gs_mesh.define_region('air', VAC_RESOLUTION, 'boundary')
    gs_mesh.define_region('plasma', PLASMA_RESOLUTION, 'plasma')
    for name, coil in coils.items():
        gs_mesh.define_region(name, COIL_RESOLUTION, 'coil', nTurns=coil['nturns'])

    gs_mesh.add_polygon(wall_contour, 'plasma', parent_name='air')
    for name, coil in coils.items():
        gs_mesh.add_rectangle(coil['rc'], coil['zc'], coil['w'], coil['h'], name, parent_name='air')

    mesh_pts, mesh_lc, mesh_reg = gs_mesh.build_mesh()
    coil_dict = gs_mesh.get_coils()
    cond_dict = gs_mesh.get_conductors()

    meshing.save_gs_mesh(mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict, MESH_FILE)
    print(f'Saved mesh to {MESH_FILE}')
    return mesh_pts, mesh_lc, mesh_reg, coil_dict, cond_dict, gs_mesh


if __name__ == '__main__':
    build_mesh()
