# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt

# termination criteria


def calibration(img):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    objpoints = []
    imgpoints = []

    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)
    # Draw and display the corners
    # cv2.drawChessboardCorners(img, (7, 7), corners2, ret)
    # plt.imshow(img)
    # plt.show()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return ret, mtx, dist, rvecs, tvecs


def undistortion(_img, _cmat, _coef):
    w, h = _img.shape[:2]
    newcmat, roi = cv2.getOptimalNewCameraMatrix(
        _cmat, _coef, (w, h), 1, (w, h)
    )
    undistort = cv2.undistort(_img, _cmat, _coef, None, newcmat)

    return undistort, newcmat


# %%
# org camera params
img = cv2.imread("checkerboard/checkerboard.png")
ret, org_mtx, org_dist, rvecs, tvecs = calibration(img)


# %%
# distortion
dist = np.copy(org_dist)
dist[0, 2] = 1100

img = cv2.imread("checkerboard/checkerboard.png")
distorted, newcmat = undistortion(img, org_mtx, dist)
plt.imshow(distorted)
plt.show()
cv2.imwrite("checkerboard/distorted.png", img)
# %%

ret, pred_mtx, pred_dist, rvecs, tvecs = calibration(distorted)

undistorted, newcmat = undistortion(distorted, pred_mtx, pred_dist)
plt.subplot(1, 2, 1)
plt.imshow(distorted)
plt.subplot(1, 2, 2)
plt.imshow(undistorted)
plt.savefig("checkerboard/calibration_results.png")
plt.show()

# %%
