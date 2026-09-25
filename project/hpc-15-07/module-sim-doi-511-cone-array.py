import argparse
from pathlib import Path

import opengate as gate
import opengate_core as g4
from opengate.utility import g4_units as u
from opengate.utility import read_mac_file_to_commands
import math
import numpy as np

# -------------------------
# Command-line arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, default=10)
parser.add_argument("--events", type=int, default=100)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument(
    "--source-mode",
    choices=["gamma_spectrum_cone", "gamma_cone"],
    default="gamma_spectrum_cone",
    help=(
        "gamma_spectrum_cone: cone source with a 511 keV line plus an approximate patient-scattered spectrum; "
        "gamma_cone: monoenergetic 511 keV cone source."
    ),
)
parser.add_argument("--source-radius-mm", type=float, default=13.0)
parser.add_argument("--source-distance-cm", type=float, default=20.0)
parser.add_argument("--cone-margin", type=float, default=0.3)
parser.add_argument("--scatter-fraction", type=float, default=0.35)
parser.add_argument("--scatter-energy-min-kev", type=float, default=180.0)
parser.add_argument("--scatter-energy-max-kev", type=float, default=505.0)
parser.add_argument("--scatter-energy-bins", type=int, default=48)
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"threads    = {args.threads}")
print(f"events     = {args.events}")
print(f"seed       = {args.seed}")
print(f"output_dir = {output_dir}")
print(f"source     = {args.source_mode}")

# define simulation object and set basic options
sim = gate.Simulation()
sim.visu = False
sim.visu_type = "qt"
sim.number_of_threads = args.threads

# sim.force_multithread_mode = False

sim.random_seed = args.seed
sim.progress_bar = True
sim.check_volumes_overlap = False
sim.g4_verbose = False

# Simulation world
sim.world.size = [100 * u.cm, 100 * u.cm, 100 * u.cm]
sim.world.material = "G4_AIR"

# Physics
sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
sim.physics_manager.set_production_cut("world", "gamma", 0.1 * u.mm)
sim.physics_manager.special_physics_constructors.G4OpticalPhysics = True

# Optical physics for scintillation + photon transport
sim.physics_manager.set_production_cut("world", "electron", 0.1 * u.mm)
sim.physics_manager.physics_list_name = "G4EmStandardPhysics"
sim.physics_manager.energy_range_min = 10 * u.eV
sim.physics_manager.energy_range_max = 1 * u.MeV

# Do not use Geant4 user limits for this geometry: the source-to-module flight
# path can exceed pixel-level limits and create fake UserSpecialCut gamma hits.

if sim.number_of_threads > 1:
    sim.physics_manager.energy_range_min = 10 * u.eV
    sim.physics_manager.energy_range_max = 1 * u.MeV

sim.physics_manager.optical_properties_file = "optical_properties_fast.xml"
sim.physics_manager.surface_properties_file = "surface_properties_fast.xml"

# -------------------------
# Materials (LYSO + ESR)
# -------------------------

m = sim.volume_manager.material_database

def _has_material(db, name):
    if hasattr(db, "get_material_names"):
        return name in db.get_material_names()
    if hasattr(db, "materials"):
        return name in db.materials
    if hasattr(db, "material_names"):
        return name in db.material_names
    if hasattr(db, "materials_dict"):
        return name in db.materials_dict
    if hasattr(db, "get_material"):
        try:
            db.get_material(name)
            return True
        except Exception:
            return False
    return False

def _add_material_nb_atoms(db, name, density, elements):
    # Convert dict -> ordered lists for APIs that expect (name, [symbols], [n], density)
    elem_symbols = elements
    elem_counts = None
    if isinstance(elements, dict):
        elem_symbols = sorted(elements.keys())
        elem_counts = [elements[k] for k in elem_symbols]
    try:
        if elem_counts is not None:
            return db.add_material_nb_atoms(name, elem_symbols, elem_counts, density)
        return db.add_material_nb_atoms(name, elem_symbols, density)
    except TypeError:
        pass
    try:
        return db.add_material_nb_atoms(name=name, density=density, elements=elements)
    except TypeError:
        pass
    try:
        return db.add_material_nb_atoms(name, density, elements)
    except TypeError:
        pass
    try:
        return db.add_material_nb_atoms(name, elements, density)
    except TypeError as exc:
        raise TypeError(
            "add_material_nb_atoms signature did not match known variants. "
            "Try inspecting MaterialDatabase methods for the expected order."
        ) from exc

if not _has_material(m, "LYSO"):
    print("manually adding LYSO material ")
    _add_material_nb_atoms(
        m,
        "LYSO",
        7.1 * u.g / (u.cm**3),
        {"Lu": 2, "Si": 1, "O": 5},
    )

if not _has_material(m, "ESR"):
    print("manually adding ESR material ")
    _add_material_nb_atoms(
        m,
        "ESR",
        1.38 * u.g / (u.cm**3),
        {"C": 10, "H": 8, "O": 4},
    )

# --------------------------------------------------------------
# Geometry: 8x8 array of 3x3x15 mm pixels with 200 µm ESR gaps
# --------------------------------------------------------------
pix_size = [3 * u.mm, 3 * u.mm, 15 * u.mm]
pix_material = "LYSO"
n = 8

esr_material = "ESR"
foil_thickness = 0.2 * u.mm

array_extent = n * pix_size[0] + (n - 1) * foil_thickness
block_xy = array_extent + 2 * foil_thickness

pitch = pix_size[0] + foil_thickness
offset = (n - 1) * pitch / 2.0

pixel_names = []
pixel_volumes = []

for ix in range(n):
    for iy in range(n):
        name = f"pixel_{ix}_{iy}"
        v = sim.add_volume("Box", name)
        v.mother = "world"
        v.size = pix_size
        v.material = pix_material
        v.color = [1.0, 0.0, 0.0, 1.0]
        v.translation = [ix * pitch - offset, iy * pitch - offset, 0.0]
        pixel_names.append(name)
        pixel_volumes.append(v)

# --------------------------------------------------------------
# Front diffuser + reflector for DOI
# --------------------------------------------------------------
air_gap = -0.0 * u.mm
glass_thickness = 4.0 * u.mm
glass_material = "G4_SILICON_DIOXIDE"

front_glass = sim.add_volume("Box", "front_glass")
front_glass.mother = "world"
front_glass.size = [block_xy, block_xy, glass_thickness]
front_glass.material = glass_material
front_glass.color = [0.0, 1.0, 0.0, 1.0]
front_glass.translation = [0, 0, -pix_size[2] / 2.0 - air_gap - glass_thickness / 2.0]

# --------------------------------------------------------------
# Back surface SiPM array
# --------------------------------------------------------------
back_glass_thickness = 100 * u.um
back_glass = sim.add_volume("Box", "back_glass")
back_glass.mother = "world"
back_glass.size = [block_xy, block_xy, back_glass_thickness]
back_glass.material = glass_material
back_glass.color = [1.0, 0.0, 1.0, 1.0]
back_glass.translation = [0, 0, pix_size[2] / 2.0 + air_gap + back_glass_thickness / 2.0]

sipm_material = "G4_Si"
sipm_thickness = 0.5 * u.mm
sipm_block = sim.add_volume("Box", "sipm_block")
sipm_block.mother = "world"
sipm_block.size = [block_xy, block_xy, sipm_thickness]
sipm_block.material = sipm_material
pixel_back_z = pix_size[2] / 2.0
sipm_block.translation = [0, 0, (pixel_back_z + air_gap + air_gap + back_glass_thickness) + (sipm_thickness / 2.0)]
sipm_block.color = [0.0, 0.0, 1.0, 1.0]
sipm_pixel_names = []

# -------------------------
# Sources
# -------------------------

source_radius = args.source_radius_mm * u.mm
source_distance = args.source_distance_cm * u.cm
source_center = [0, 0, -source_distance]

# The cone is a variance-reduction choice: it sends most photons toward the
# module instead of spending CPU on gammas emitted away from the detector.
acceptance_radius = args.cone_margin * (block_xy / 2.0 + source_radius)
theta_max = math.atan(acceptance_radius / source_distance) 

source_vis_mother = "world"
source_vis_translation = source_center
source_vis_material = "G4_AIR"

# Visible sphere to show the active source region in the Geant4 viewer.
source_vis = sim.add_volume("Sphere", "source_vis")
source_vis.mother = source_vis_mother
source_vis.rmin = 0
source_vis.rmax = source_radius
source_vis.sphi = 0 * u.deg
source_vis.dphi = 360 * u.deg
source_vis.stheta = 0 * u.deg
source_vis.dtheta = 180 * u.deg
source_vis.material = source_vis_material
source_vis.translation = source_vis_translation
source_vis.color = [1.0, 0.2, 0.2, 1.0]
source_vis.color = [1.0, 1.0, 1.0, 1.0]


def _configure_cone_source(source, n_events):
    source.particle = "gamma"
    source.position.type = "sphere"
    source.position.radius = source_radius
    source.position.translation = source_center
    source.direction.type = "iso"
    source.direction.momentum = [0, 0, 0]
    source.direction.theta = [180.0 * u.deg, (180.0 - math.degrees(theta_max)) * u.deg]
    source.direction.phi = [0, 360 * u.deg]
    source.n = int(n_events)


def _patient_scatter_spectrum():
    """Approximate single-Compton scattered 511 keV photon spectrum.

    This is a variance-reduced surrogate for patient scatter: it changes only
    the incoming gamma energy, while keeping the same cone angular distribution.
    The weights are based on the Klein-Nishina shape transformed from scatter
    angle to scattered photon energy.
    """
    e0 = 511.0
    e_min = max(float(args.scatter_energy_min_kev), e0 / 3.0 + 1e-3)
    e_max = min(float(args.scatter_energy_max_kev), e0 - 1e-3)
    n_bins = max(int(args.scatter_energy_bins), 2)
    energies = np.linspace(e_min, e_max, n_bins)

    cos_theta = 2.0 - e0 / energies
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sin2_theta = 1.0 - cos_theta ** 2
    ratio = energies / e0
    weights = ratio + 1.0 / ratio - sin2_theta
    weights = np.clip(weights, 0.0, None)
    if not np.any(weights > 0):
        weights = np.ones_like(energies)
    return list(energies * u.keV), weights.tolist()


sources = []

if args.source_mode == "gamma_cone":
    src = sim.add_source("GenericSource", "gamma511")
    _configure_cone_source(src, args.events)
    src.energy.type = "mono"
    src.energy.mono = 511 * u.keV
    sources.append(src)
else:
    scatter_fraction = float(np.clip(args.scatter_fraction, 0.0, 1.0))
    n_scatter = int(round(args.events * scatter_fraction))
    n_primary = int(args.events) - n_scatter

    if n_primary > 0:
        src_primary = sim.add_source("GenericSource", "gamma511_primary")
        _configure_cone_source(src_primary, n_primary)
        src_primary.energy.type = "mono"
        src_primary.energy.mono = 511 * u.keV
        sources.append(src_primary)

    if n_scatter > 0:
        src_scatter = sim.add_source("GenericSource", "gamma511_patient_scatter")
        _configure_cone_source(src_scatter, n_scatter)
        scatter_energies, scatter_weights = _patient_scatter_spectrum()
        src_scatter.energy.type = "spectrum_discrete"
        src_scatter.energy.spectrum_energies = scatter_energies
        src_scatter.energy.spectrum_weights = scatter_weights
        sources.append(src_scatter)

print(f"source radius          = {source_radius / u.mm:.3g} mm")
print(f"source distance        = {source_distance / u.cm:.3g} cm")
print(f"source cone half-angle = {math.degrees(theta_max):.3g} deg")
if args.source_mode == "gamma_spectrum_cone":
    print(f"scatter fraction       = {float(np.clip(args.scatter_fraction, 0.0, 1.0)):.3g}")
    print(f"scatter energy range   = {args.scatter_energy_min_kev:.3g} to {args.scatter_energy_max_kev:.3g} keV")

if sim.visu is True:
    sim.number_of_threads = 1
    for s in sources:
        s.n = 1

    #n_vis_total = sum(int(s.n) for s in sources)
    #for s in sources:
    #    s.n = max(1, int(round(2500 * int(s.n) / max(n_vis_total, 1))))
    
    #sim.physics_manager.special_physics_constructors.G4OpticalPhysics = False

#--------------------------------------------------

# -------------------------
# Optical surfaces
# -------------------------
esr_surface_name = "ESR_fast_reflector"
det_surface_name = "Detector_fast"
diffuse_surface_name = "Diffuse_front"

for name in pixel_names:
    sim.physics_manager.add_optical_surface(name, "world", esr_surface_name)
    sim.physics_manager.add_optical_surface("world", name, esr_surface_name)
    sim.physics_manager.add_optical_surface(name, "front_glass", diffuse_surface_name)
    sim.physics_manager.add_optical_surface("front_glass", name, diffuse_surface_name)

sim.physics_manager.add_optical_surface("front_glass", "world", esr_surface_name)

optical_max_time = 100 * u.ns

# -------------------------
# HIT scoring
# -------------------------
def _filter_digi_attributes(attrs):
    try:
        avail = set(g4.GateDigiAttributeManager.GetInstance().GetAvailableDigiAttributeNames())
        print("attributes " + str(avail))
        return [a for a in attrs if a in avail]
    except Exception:
        return attrs

avail = set(g4.GateDigiAttributeManager.GetInstance().GetAvailableDigiAttributeNames())

sipm_photons = sim.add_actor("PhaseSpaceActor", "sipm_photons")
sipm_photons.attached_to = sipm_block
sipm_photons.output_filename = str(output_dir / "sipm_hits.root")
sipm_photons.steps_to_store = "entering"
sipm_photons.attributes = _filter_digi_attributes(
    [
        "EventID",
        "TrackID",
        "ParentID",
        "ParticleName",
        "GlobalTime",
        "Position",
        "PrePosition",
        "PostPosition",
        "PostStepVolumeCopyNo",
        "PostStepUniqueVolumeID",
        "TrackVolumeName",
        "TotalEnergyDeposit",
        "KineticEnergy",
        "EnergyDeposit",
        "Edep",
    ]
)

opt_filter = sim.add_filter("ParticleFilter", "optical_only")
opt_filter.particle = "opticalphoton"
sipm_photons.filters = [opt_filter]

optical_time_filter = sim.add_filter("ThresholdAttributeFilter", "late_optical_time_filter")
optical_time_filter.attribute = "GlobalTime"
optical_time_filter.value_min = optical_max_time

optical_kill = sim.add_actor("KillActor", "kill_late_optical_photons")
optical_kill.attached_to = pixel_volumes
optical_kill.filters = [opt_filter, optical_time_filter]

print("Actors.")
gamma_hits = sim.add_actor("DigitizerHitsCollectionActor", "gamma_hits")
gamma_hits.attached_to = pixel_volumes
gamma_hits.output_filename = str(output_dir / "gamma_steps.root")
gamma_hits.attributes = _filter_digi_attributes(
    [
        "EventID",
        "TrackID",
        "ParentID",
        "ParticleName",
        "ProcessDefinedStep",
        "TrackCreatorProcess",
        "GlobalTime",
        "PrePosition",
        "PostPosition",
        "Position",
        "PreKineticEnergy",
        "PostKineticEnergy",
        "KineticEnergy",
        "StepLength",
        "TrackVolumeName",
        "TotalEnergyDeposit",
    ]
)

gamma_filter = sim.add_filter("ParticleFilter", "gamma_only")
gamma_filter.particle = "gamma"

gamma_edep_filter = sim.add_filter("ThresholdAttributeFilter", "gamma_edep_filter")
gamma_edep_filter.attribute = "TotalEnergyDeposit"
gamma_edep_filter.value_min = 1e-9 * u.keV
gamma_hits.filters = [gamma_filter, gamma_edep_filter]

# -------------------------
# Run
# -------------------------
sim.run()

print("Done.")
