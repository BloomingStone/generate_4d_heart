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

# The label of LV
LV_LABEL = 2

from .dvf import generate_4d_dvf
from .dsa import generate_rotate_dsa