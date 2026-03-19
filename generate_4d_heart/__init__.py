from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent

# The shape of SSM, for now, it should always be (144, 144, 128)
SSM_SHAPE = (144, 144, 128)
SSM_SPACING = (-1., 1., 1.)  # in mm

# SSM is trained on digital phantoms whose spacing is (-1, 1, 1) mm, in LAS orientation, so the direction is also (-1, 1, 1)

# The direction is calculated from the affine matrix of the nii images that train SSM.
# SSM_DIRECTION = np.diag(affine_ssm_train[:3, :3])
SSM_DIRECTION = (-1, 1, 1)

# The number of total phase in SSM
NUM_TOTAL_PHASE = 20

# The number of total points in SSM
NUM_TOTAL_POINTS = 500

# The labels
from enum import IntEnum

class CavityLabel(IntEnum):
    LV_MYO = 1
    LV = 2
    RV = 3
    LA = 4
    RA = 5

class VesselLabel(IntEnum):
    PV = 6
    AORTA = 7


MU_WATER = 0.02     # mm^-1
MU_IDODINE = 0.25   # mm^-1

from .moving_dvf.dvf import generate_4d_dvf
