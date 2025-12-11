# The shape of SSM, for now, it should always be (144, 144, 128)
SSM_SHAPE = (144, 144, 128)

# The direction is calculated from the affine matrix of the nii images that train SSM.
# SSM_DIRECTION = np.diag(affine_ssm_train[:3, :3])
SSM_DIRECTION = (-1, 1, 1)

# The number of total phase in SSM
NUM_TOTAL_PHASE = 20

# The number of cavity label in SSM
NUM_TOTAL_CAVITY_LABEL = 5

# The number of total points in SSM
NUM_TOTAL_POINTS = 500

# The labels
LV_MYO_LABEL = 1
LV_LABEL = 2
RV_LABEL = 3
LA_LABEL = 4
RA_LABEL = 5

ALL_CAVITY_LABEL = [LV_MYO_LABEL, LV_LABEL, RV_LABEL, LA_LABEL, RA_LABEL]
assert len(ALL_CAVITY_LABEL) == NUM_TOTAL_CAVITY_LABEL

PV_LABEL = 6    # pulmonary vein
AORTA_LABEL = 7

MU_WATER = 0.02     # mm^-1
MU_IDODINE = 0.25   # mm^-1

from .moving_dvf.dvf import generate_4d_dvf
