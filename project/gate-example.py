#imports
import opengate as gate
import opengate_core as g4
from opengate.utility import g4_units as u

import pathlib
import opengate as gate

#define simulation object and set basic options
sim = gate.Simulation()

m = sim.volume_manager.material_database
db_path = pathlib.Path(gate.__file__).parent / "contrib" / "spect" / "spect_ge_nm670_materials.db"
m.read_from_file(str(db_path))

print(sorted(m.material_builders.keys()))


sim.visu = True
sim.visu_type = "qt"
sim.number_of_threads = 1 
sim.random_seed = 12345
sim.progress_bar = True
sim.check_volumes_overlap = False
sim.g4_verbose = False
sim.world.size = [10 * u.cm, 10 * u.cm, 15 * u.cm]

sim.physics_manager.optical_properties_file = "optical_properties_fast.xml"

optical_system = sim.add_volume("Box", "optical_system")
optical_system.size = [10 * u.cm, 10 * u.cm, 14 * u.cm]
optical_system.material = "G4_AIR"
optical_system.translation = [0 * u.cm, 0 * u.cm, 0 * u.cm]

crystal = sim.add_volume("Box", "crystal")
crystal.mother = optical_system.name
crystal.size = [3 * u.mm, 3 * u.mm, 20 * u.mm]
crystal.translation = [0 * u.mm, 0 * u.mm, 10 * u.mm]
crystal.material = "BGO"

grease = sim.add_volume("Box", "grease")
grease.mother = optical_system.name
grease.size = [3 * u.mm, 3 * u.mm, 0.015 * u.mm]
grease.material = "Epoxy"
grease.translation = [0 * u.mm, 0 * u.mm, 20.0075 * u.mm]

pixel = sim.add_volume("Box", "pixel")
pixel.mother = optical_system.name
pixel.size = [3 * u.mm, 3 * u.mm, 0.1 * u.mm]
pixel.material = "SiO2"
pixel.translation = [0 * u.mm, 0 * u.mm, 20.065 * u.mm]

sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"

# This also includes Scintillation and Cerenkov processes.
sim.physics_manager.special_physics_constructors.G4OpticalPhysics = True
sim.physics_manager.set_production_cut("world", "electron", 10 * u.mm)
sim.physics_manager.set_production_cut("world", "positron", 10 * u.um)
sim.physics_manager.set_production_cut("crystal", "electron", 10 * u.um)
sim.physics_manager.set_production_cut("crystal", "positron", 10 * u.um)

# In Gate 10, enery range limits should be set like this for scintillation. # Reason for this is unknown.
sim.physics_manager.energy_range_min = 10 * u.eV
sim.physics_manager.energy_range_max = 1 * u.MeV

opt_surf_optical_system_to_crystal = sim.physics_manager.add_optical_surface(volume_from="optical_system",volume_to="crystal", g4_surface_name="Customized3_LUT")
opt_surf_crystal_to_optical_system = sim.physics_manager.add_optical_surface("crystal", "optical_system", "Customized3_LUT")
opt_surf_grease_to_crystal = sim.physics_manager.add_optical_surface("grease", "crystal", "Customized2_LUT")
opt_surf_crystal_to_grease = sim.physics_manager.add_optical_surface("crystal", "grease", "Customized2_LUT")
opt_surface_pixel_to_grease = sim.physics_manager.add_optical_surface("pixel", "grease", "Customized4_LUT")
opt_surf_grease_to_pixel = sim.physics_manager.add_optical_surface("grease", "pixel", "Customized4_LUT")

source = sim.add_source("GenericSource", "my_source")
source.particle = "e-"
source.energy.type = "mono"
source.energy.mono = 420 * u.keV
source.position.type = "sphere"
source.position.radius = 0 * u.mm
source.activity = 1000 * u.Bq
source.direction.type = "iso"
source.direction.theta = [163 * u.deg, 165 * u.deg]
source.direction.phi = [100 * u.deg, 110 * u.deg]
source.position.translation = [0 * u.mm, 0 * u.mm, 19 * u.mm]


phase = sim.add_actor("PhaseSpaceActor", "Phase")
phase.attached_to = pixel.name
phase.output_filename = "test075_optigan_create_dataset_first_phase_space_with_track_volume.root"
phase.attributes = [
    "EventID",
    "ParticleName",
    "Position",
    "TrackID",
    "ParentID",
    "Direction",
    "KineticEnergy",
    "PreKineticEnergy",
    "PostKineticEnergy",
    "TotalEnergyDeposit",
    "LocalTime",
    "GlobalTime",
    "TimeFromBeginOfEvent",
    "StepLength",
    "TrackCreatorProcess",
    "TrackLength",
    "TrackVolumeName",
    "PDGCode"
]

sim.run()

print("Done.")