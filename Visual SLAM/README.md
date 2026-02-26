# Visual SLAM

Monocular visual odometry pipeline built on ORB features, Essential matrix bootstrapping, and PnP RANSAC tracking. No g2o, no Ceres — raw OpenCV + Eigen3.

---

## What is Visual SLAM?

**Simultaneous Localization and Mapping** is the problem of estimating a sensor's 6-DoF trajectory through an unknown environment while concurrently building a consistent map of that environment — using only the sensor's own observations. In the monocular case, the only input is a sequence of 2D images $\{I_t\}_{t=0}^{T}$.

The state to be estimated at time $t$ is the camera pose expressed as a rigid body transformation in $SE(3)$:

```math
\mathbf{T}_t = \begin{bmatrix} \mathbf{R}_t & \mathbf{t}_t \\ \mathbf{0}^\top & 1 \end{bmatrix} \in SE(3), \quad \mathbf{R}_t \in SO(3),\ \mathbf{t}_t \in \mathbb{R}^3
```

The full trajectory is the sequence $\mathcal{X} = \{\mathbf{T}_0, \mathbf{T}_1, \ldots, \mathbf{T}_T\}$ and the map is the set of 3D landmarks $\mathcal{M} = \{\mathbf{p}_1, \ldots, \mathbf{p}_N\} \subset \mathbb{R}^3$.

---

## Camera Model

A 3D point $\mathbf{p} = [X, Y, Z]^\top$ in camera coordinates projects to pixel $\mathbf{u} = [u, v]^\top$ via the pinhole model:

```math
\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x \frac{X}{Z} + c_x \\ f_y \frac{Y}{Z} + c_y \end{bmatrix}
```

In homogeneous form with calibration matrix $\mathbf{K}$:

```math
\lambda \tilde{\mathbf{u}} = \mathbf{K} \mathbf{p}, \qquad \mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
```

The inverse mapping (back-projection at unit depth) is:

```math
\mathbf{d} = \mathbf{K}^{-1} \tilde{\mathbf{u}} = \left[\frac{u - c_x}{f_x},\ \frac{v - c_y}{f_y},\ 1\right]^\top
```

---

## Feature Detection and Matching

### ORB

ORB (Oriented FAST + Rotated BRIEF) detects keypoints at corners of high image gradient. For a pixel $\mathbf{x}$, the FAST response uses a Bresenham circle $\mathcal{C}_{16}(\mathbf{x})$ of 16 pixels at radius 3. A keypoint is accepted if at least $n$ contiguous pixels on the circle are all brighter or all darker than $I(\mathbf{x})$ by threshold $\varepsilon$:

```math
\exists\, S \subseteq \mathcal{C}_{16}(\mathbf{x}),\ |S| \geq n : \forall \mathbf{s} \in S,\ I(\mathbf{s}) > I(\mathbf{x}) + \varepsilon \ \vee\ I(\mathbf{s}) < I(\mathbf{x}) - \varepsilon
```

Orientation is assigned via the intensity centroid of a patch $\mathcal{P}$:

```math
m_{pq} = \sum_{\mathbf{x} \in \mathcal{P}} x^p y^q\, I(\mathbf{x}), \qquad \theta = \text{atan2}(m_{01},\, m_{10})
```

The descriptor is a length-256 binary string built from steered BRIEF pairs $(\mathbf{x}_i', \mathbf{y}_i')$ rotated by $\theta$:

```math
\tau(\mathbf{p};\, \mathbf{x}, \mathbf{y}) := \begin{cases} 1 & I(\mathbf{p} + \mathbf{x}) < I(\mathbf{p} + \mathbf{y}) \\ 0 & \text{otherwise} \end{cases}, \qquad d_{ORB} = \bigoplus_{i=1}^{256} \tau(\mathbf{p};\, \mathbf{x}_i', \mathbf{y}_i')
```

Matching uses Hamming distance $d_H$ with Lowe's ratio test:

```math
\text{accept match}\ (i, j^*) \iff \frac{d_H(d_i,\, d_{j^*})}{d_H(d_i,\, d_{j^{**}})} < \rho, \qquad \rho = 0.75
```

where $j^*$ and $j^{**}$ are the nearest and second-nearest neighbors in the target frame.

---

## Initialization — Essential Matrix

Given $N$ point correspondences $\{(\mathbf{u}_i, \mathbf{u}_i')\}$, the Essential matrix $\mathbf{E}$ encodes the relative rotation and translation between two views. The epipolar constraint must hold for every inlier:

```math
\tilde{\mathbf{u}}_i'^\top\, \mathbf{E}\, \tilde{\mathbf{u}}_i = 0, \qquad \tilde{\mathbf{u}} = \mathbf{K}^{-1}\mathbf{u}
```

$\mathbf{E}$ is estimated via the **5-point algorithm** inside RANSAC. The algebraic structure of $\mathbf{E}$ constrains it to the essential manifold:

```math
\mathbf{E} = \mathbf{R}\,[\mathbf{t}]_\times, \qquad 2\,\mathbf{E}\mathbf{E}^\top\mathbf{E} - \text{tr}(\mathbf{E}\mathbf{E}^\top)\,\mathbf{E} = \mathbf{0}
```

Decomposing $\mathbf{E}$ via SVD $\mathbf{E} = \mathbf{U}\,\text{diag}(1,1,0)\,\mathbf{V}^\top$ yields four $(\mathbf{R}, \mathbf{t})$ hypotheses. The correct one is the unique solution where the reconstructed 3D points have **positive depth** in both cameras — the cheirality check:

```math
\mathbf{p} = \mathbf{T}_1^{-1}\mathbf{q},\quad Z_1 = \mathbf{e}_3^\top \mathbf{q} > 0,\quad Z_2 = \mathbf{e}_3^\top (\mathbf{R}\mathbf{q} + \mathbf{t}) > 0
```

---

## Triangulation

Given a matched pair $(\mathbf{u}, \mathbf{u}')$ and projection matrices $\mathbf{P} = \mathbf{K}[\mathbf{I}|\mathbf{0}]$, $\mathbf{P}' = \mathbf{K}[\mathbf{R}|\mathbf{t}]$, the 3D point is recovered by solving the homogeneous system $\mathbf{A}\mathbf{X} = \mathbf{0}$:

```math
\mathbf{A} = \begin{bmatrix} u\,\mathbf{p}_3^\top - \mathbf{p}_1^\top \\ v\,\mathbf{p}_3^\top - \mathbf{p}_2^\top \\ u'\mathbf{p}_3'^\top - \mathbf{p}_1'^\top \\ v'\mathbf{p}_3'^\top - \mathbf{p}_2'^\top \end{bmatrix}
```

where $\mathbf{p}_k^\top$ is the $k$-th row of $\mathbf{P}$. The solution is the right singular vector of $\mathbf{A}$ corresponding to the smallest singular value, dehomogenized to $\mathbf{X} = [X/W, Y/W, Z/W]^\top$.

Triangulation degenerates when the **parallax angle** $\alpha$ is small:

```math
\alpha = \arccos\left(\frac{\mathbf{d} \cdot \mathbf{d}'}{\|\mathbf{d}\|\|\mathbf{d}'\|}\right) \approx \frac{\|\Delta\mathbf{u}\|}{f}
```

This is why keyframe selection is gated on a minimum reprojected parallax $\|\Delta\mathbf{u}\| > \delta_{kf}$ pixels.

---

## Tracking — PnP RANSAC

Once a map exists, pose for frame $t$ is recovered via **Perspective-n-Point**: given $n$ 2D–3D correspondences $\{(\mathbf{u}_i, \mathbf{p}_i)\}$, find $\mathbf{T}_t$ minimizing reprojection error:

```math
\mathbf{T}_t^* = \underset{\mathbf{T} \in SE(3)}{\arg\min} \sum_{i=1}^{n} \rho\!\left(\left\|\mathbf{u}_i - \pi\!\left(\mathbf{K},\, \mathbf{T}\,\mathbf{p}_i\right)\right\|^2\right)
```

where $\pi(\mathbf{K}, \cdot)$ is the projection function and $\rho(\cdot)$ is a robust kernel (Huber or trivial inside RANSAC). The **EPnP** algorithm solves this in $O(n)$ via a linear formulation over control points, followed by Gauss-Newton refinement.

RANSAC hypothesis $h$ is accepted as an inlier set if:

```math
\left\|\mathbf{u}_i - \pi\!\left(\mathbf{K},\, \mathbf{T}_h\,\mathbf{p}_i\right)\right\|^2 < \chi^2_{2,\,0.95} \approx 5.99 \text{ px}^2
```

The minimum number of RANSAC iterations to guarantee finding at least one all-inlier sample with probability $\eta$ is:

```math
k \geq \frac{\log(1 - \eta)}{\log\!\left(1 - \epsilon^s\right)}, \quad \epsilon = \text{inlier ratio},\ s = \text{sample size}
```

---

## Map Management

The map $\mathcal{M}$ is a set of **landmarks** $\ell_j = (\mathbf{p}_j,\, \mathcal{O}_j)$ where $\mathcal{O}_j = \{(t, i)\}$ records every frame $t$ and keypoint index $i$ that observed $\ell_j$. Landmarks are culled when:

```math
\frac{|\mathcal{O}_j|}{|\mathcal{F}_j|} < \tau_{obs}, \qquad \mathcal{F}_j = \{\text{frames that could have observed } \ell_j\}
```

The **local map** for the active window $\mathcal{W} = \{k_{t-w}, \ldots, k_t\}$ of $w$ keyframes contains all landmarks observed in any keyframe in $\mathcal{W}$:

```math
\mathcal{M}_{local} = \bigcup_{k \in \mathcal{W}}\ \bigcup_{i \in \mathcal{K}_k} \ell(\mathbf{u}_i^k)
```

---

## Keyframe Selection

A frame $I_t$ is promoted to a keyframe when the mean feature parallax from the last keyframe $k$ exceeds a threshold and enough time has elapsed:

```math
\bar{\delta}(t, k) = \frac{1}{|\mathcal{M}_{t \cap k}|} \sum_{j \in \mathcal{M}_{t \cap k}} \left\|\mathbf{u}_j^t - \mathbf{u}_j^k\right\| > \delta_{kf}
```

and $t - k > \Delta_{min}$ frames. The intuition: below $\delta_{kf}$, new triangulations are poorly conditioned; above it, enough new structure is visible to warrant a keyframe.

---

## Trajectory Evaluation

Output is a **TUM-format** trajectory file — one line per frame:

```
timestamp tx ty tz qx qy qz qw
```

The unit quaternion $\mathbf{q} \in S^3$ parameterizes rotation without singularity. Its relation to $\mathbf{R} \in SO(3)$ via the exponential map is:

```math
\mathbf{R} = \exp([\boldsymbol{\omega}]_\times) = \mathbf{I} + \frac{\sin\theta}{\theta}[\boldsymbol{\omega}]_\times + \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\omega}]_\times^2, \quad \theta = \|\boldsymbol{\omega}\|
```

Quantitative accuracy is measured with **Absolute Trajectory Error (ATE)**:

```math
\text{ATE} = \left(\frac{1}{T} \sum_{t=1}^{T} \left\| \mathbf{t}_t - \mathbf{t}_t^{gt} \right\|^2 \right)^{1/2}
```

and **Relative Pose Error (RPE)** over a fixed time window $\Delta$:

```math
\mathbf{E}_i = \left(\mathbf{T}_i^{gt}\right)^{-1} \mathbf{T}_{i+\Delta}^{gt} \cdot \left(\mathbf{T}_i\right)^{-1} \mathbf{T}_{i+\Delta}, \qquad \text{RPE} = \left(\frac{1}{N}\sum_i \left\|\mathbf{t}(\mathbf{E}_i)\right\|^2\right)^{1/2}
```

Evaluate with `evo`:
```bash
pip install evo
evo_ape tum groundtruth.txt trajectory_tum.txt -va --plot
evo_rpe tum groundtruth.txt trajectory_tum.txt -va --delta 1 --plot
```

---

## Datasets

| Dataset | Scene | Difficulty | Calibration |
|---|---|---|---|
| [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) `fr1_xyz` | Indoor, slow | Easy | `calib_tum_fr1.yaml` |
| [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) `fr1_desk` | Indoor, clutter | Medium | `calib_tum_fr1.yaml` |
| [EuRoC MAV](https://rpg.ifi.uzh.ch/docs/IJRR17_Burri.pdf) `MH_01_easy` | Indoor, drone | Medium | `calib_euroc.yaml` |
| [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) seq `00` | Outdoor, vehicle | Hard | KITTI P0 matrix |

```bash
wget https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
tar xzf rgbd_dataset_freiburg1_xyz.tgz
./visual_slam rgbd_dataset_freiburg1_xyz/rgb/ calib_tum_fr1.yaml
```

---

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# OpenCV must be built with -DWITH_VTK=ON -DBUILD_opencv_viz=ON
```
