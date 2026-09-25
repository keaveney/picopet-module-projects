import opengate as gate
import opengate_core as g4
from opengate.utility import g4_units as u
from pathlib import Path
from opengate.utility import read_mac_file_to_commands

#define simulation object and set basic options
sim = gate.Simulation()
sim.visu = False
sim.visu_type = "qt"
sim.number_of_threads = 8 

#sim.force_multithread_mode = False

sim.random_seed = 32845
sim.progress_bar = True
sim.check_volumes_overlap = False
sim.g4_verbose = False

# Simulation world
sim.world.size = [40 * u.cm, 40 * u.cm, 40 * u.cm]
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

#allow user to put limits on particle propoagation to avoid nonconverging simulations.
sim.physics_manager.set_user_limits_particles(["all"])

if sim.number_of_threads > 1:
    sim.physics_manager.energy_range_min = 10 * u.eV
    sim.physics_manager.energy_range_max = 1 * u.MeV

sim.physics_manager.optical_properties_file = "optical_properties_fast.xml"
sim.physics_manager.surface_properties_file = "surface_properties_fast.xml"

# -------------------------
# Materials (LYSO + ESR)
#--------------------------

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
        # keep deterministic ordering
        elem_symbols = sorted(elements.keys())
        elem_counts = [elements[k] for k in elem_symbols]
    try:
        if elem_counts is not None:
            return db.add_material_nb_atoms(name, elem_symbols, elem_counts, density)
        return db.add_material_nb_atoms(name, elem_symbols, density)
    except TypeError:
        pass
    # Fallbacks for other possible signatures
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
    # A commonly used approximation: Lu2SiO5:Ce with density ~7.1 g/cm3
    print("manually adding LYSO material " )
    _add_material_nb_atoms(
        m,
        "LYSO",
        7.1 * u.g / (u.cm**3),
        {"Lu": 2, "Si": 1, "O": 5},
    )

# ESR foil approximation: define ESR material so we can assign optical properties
if not _has_material(m, "ESR"):
    print("manually adding ESR material " )
    _add_material_nb_atoms(
        m,
        "ESR",
        1.38 * u.g / (u.cm**3),
        {"C": 10, "H": 8, "O": 4},
    )

# --------------------------------------------------------------
# Geometry: 8x8 array of 3x3x15 mm pixels with 200 µm ESR gaps
# --------------------------------------------------------------
# Pixel definition 
pix_size = [3 * u.mm, 3 * u.mm, 15 * u.mm]
pix_material = "LYSO"
n = 8

#foil 
esr_material = "ESR"
foil_thickness = 0.2 * u.mm

# Overall array size
array_extent = n * pix_size[0] + (n - 1) * foil_thickness
block_xy = array_extent + 2 * foil_thickness  # outer extent including ESR frame

# Place pixels in an 8x8 grid (pitch = pixel size + ESR foil)
pitch = pix_size[0] + foil_thickness
offset = (n - 1) * pitch / 2.0

# build/arrange the array of pixels
pixel_names = []
pixel_volumes = []

for ix in range(n):
    for iy in range(n):
        name = f"pixel_{ix}_{iy}"
        v = sim.add_volume("Box", name)
        v.mother = "world"
        v.size = pix_size
        v.material = pix_material
        v.translation = [ix * pitch - offset, iy * pitch - offset, 0.0]
        pixel_names.append(name)
        pixel_volumes.append(v)

# --------------------------------------------------------------
# Front diffuser + reflector for DOI
# --------------------------------------------------------------
air_gap = -0.0 * u.mm
glass_thickness = 1.0 * u.mm
glass_material = "G4_SILICON_DIOXIDE"

#front_glass = sim.add_volume("Box", "front_glass")
#front_glass.mother = "world"
#front_glass.size = [block_xy, block_xy, glass_thickness]
#front_glass.material = glass_material
#front_glass.color = [0.0, 1.0, 0.0, 1.0]
#front_glass.translation = [0, 0, -pix_size[2] / 2.0 - air_gap - glass_thickness / 2.0]

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

# Back surface: SiPM plane segmented into 8x8 pixels
sipm_material = "G4_Si"
sipm_thickness = 0.5 * u.mm
sipm_block = sim.add_volume("Box", "sipm_block")
sipm_block.mother = "world"
sipm_block.size = [block_xy, block_xy, sipm_thickness]
sipm_block.material = sipm_material
pixel_back_z = pix_size[2] / 2.0
sipm_block.translation = [0, 0, (pixel_back_z + air_gap + air_gap + back_glass_thickness ) + (sipm_thickness / 2.0)]
sipm_block.color = [0.0, 0.0, 1.0, 1.0]
sipm_pixel_names = []

# -------------------------
# Sources
# -------------------------
# 511 keV gammas striking the array
src = sim.add_source("GenericSource", "gamma511")
src.particle = "gamma"
src.energy.type = "mono"
src.energy.mono = 511 * u.keV

# Shoot a pencil beam along +z, starting in front of the array
src.position.type = "point"
src.position.translation = [7.5 * u.mm, 7.5 * u.mm, -50 * u.mm]
src.direction.type = "momentum"
src.direction.momentum = [0, 0, 1]
src.n = 1000

if sim.visu == True:
    sim.number_of_threads = 1  # visualization is more stable in single-thread mode
    src.n = 1   
# -------------------------
# Optical surfaces (ESR reflector + SiPM detector surface)
# -------------------------
esr_surface_name = "ESR_fast_reflector"
det_surface_name = "Detector_fast" 
diffuse_surface_name = "Diffuse_front" 

# Approximate ESR in gaps by coating pixel lateral faces 
for name in pixel_names:
    sim.physics_manager.add_optical_surface(name, "world", esr_surface_name)
    sim.physics_manager.add_optical_surface("world", name, esr_surface_name)
    #sim.physics_manager.add_optical_surface(name, "front_glass", diffuse_surface_name)
    #sim.physics_manager.add_optical_surface("front_glass", name, diffuse_surface_name)

#sim.physics_manager.add_optical_surface("front_glass", "world", esr_surface_name)


max_time = 10 * u.ns
max_track_length = 15 * u.cm
for v in pixel_volumes:
    v.set_max_time(max_time)
    v.set_max_track_length(max_track_length)

# -------------------------
# HIT scoring (this is the key)
# -------------------------
# Actors are the intended tool to collect per-step/per-hit data. :contentReference[oaicite:1]{index=1}
#
# The exact actor class name can differ slightly by OpenGATE version.
# In this install, use "DigitizerHitsCollectionActor".

def _filter_digi_attributes(attrs):
    try:
        avail = set(g4.GateDigiAttributeManager.GetInstance().GetAvailableDigiAttributeNames())
        print("attributes" + str(avail))
        return [a for a in attrs if a in avail]
    except Exception:
        return attrs

avail = set(g4.GateDigiAttributeManager.GetInstance().GetAvailableDigiAttributeNames())

# SiPM photons: store optical photons entering the back plane

sipm_photons = sim.add_actor("PhaseSpaceActor", "sipm_photons")
sipm_photons.attached_to = sipm_block
sipm_photons.output_filename = "sipm_hits.root"
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
        "Edep"
    ]
)

opt_filter = sim.add_filter("ParticleFilter", "optical_only")
opt_filter.particle = "opticalphoton"
sipm_photons.filters = [opt_filter]

print("Actors.")
# Primary gamma interaction hits (robust alternative to PhaseSpaceActor)
gamma_hits = sim.add_actor("DigitizerHitsCollectionActor", "gamma_hits")
gamma_hits.attached_to = pixel_volumes
gamma_hits.output_filename = "gamma_steps.root"
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
