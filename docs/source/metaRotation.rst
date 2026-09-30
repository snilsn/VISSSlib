metaRotation - retrieve camera rotation
=======================================


Motivation
----------

Although alignment of both observation volumes is a priority during
installation, the cameras can be rotated or displaced, i.e., misaligned.
As a result, the same particle may be observed at different heights and
:math:`z = Y_\textrm{L} = Y_\textrm{F}` does not hold. The observed
offsets are not constant and can change due to unstable surfaces or
pressure of accumulated snow on the VISSS frame. We could simply ignore
the misalignment and continue to take :math:`z` from the leader, but
this would not allow us to generally use the vertical position to match
particles from both cameras (see above). Also, offsets in :math:`z`
reduce the common observation volume of both cameras, which could lead
to biases when calibrating the PSDs if not accounted for.

Besides a constant offset in the vertical :math:`z` dimension
:math:`O_{\textrm{f}z}`, one of the cameras can also be rotated around
the optical axis (expressed analogously to aircraft coordinate systems
with roll :math:`\varphi`), around the horizontal axis perpendicular to
the optical axis (pitch :math:`\theta`), or around the vertical axis
(yaw :math:`\psi`). As a consequence,
:math:`\Delta z = Y_\textrm{L}-Y_\textrm{F}` depends on the position of
the particle in the observation volume.

To account for the misalignment, we attach the coordinate system to the
leader (i.e., we assume that the leader is perfectly aligned
(:math:`x_\textrm{L}`,\ :math:`y_\textrm{L}`,\ :math:`z_\textrm{L}`) =
(:math:`x`,\ :math:`y`,\ :math:`z`)) and retrieve the misalignment of
the follower with respect to the leader in terms of :math:`\varphi`,
:math:`\theta` and :math:`O_{\textrm{f}z}`. We cannot derive
:math:`\psi` from the observation and we have no choice but to neglect
it by assuming :math:`\psi = 0` to reduce the number of unknowns.
Mathematically, we need to transform the follower coordinate system
(:math:`x_\textrm{F}`,\ :math:`y_\textrm{F}`,\ :math:`z_\textrm{F}`) to
our leader reference coordinate system
(:math:`x_\textrm{L}`,\ :math:`y_\textrm{L}`,\ :math:`z_\textrm{L}`)
using rotation and shear matrices. In the `Coordinate transformation
derivation`_ section below, we show how the transformation matrices can be
arranged so that the follower’s vertical measure :math:`z_\textrm{F}`
can be converted to :math:`z_\textrm{L}` depending on :math:`\varphi`
and :math:`\theta` with

.. math::

   \begin{aligned}
    \label{eq:coordinates}
       z_\textrm{L} =   -& \frac{  \sin\theta }{\cos\theta } x_\textrm{L}      + \frac{\sin \varphi}{\cos\theta } y_\textrm{F}      +  \frac{\cos \varphi}{\cos\theta } (z_\textrm{F} + O_{\textrm{f}z}) .
   \end{aligned}

This equation can be considered as a forward operator that calculates
the expected leader observation :math:`z_\textrm{L}` based on a
misalignment state (:math:`O_{\textrm{f}z}`, :math:`\varphi`, and
:math:`\theta`) and additional parameters (:math:`x_\textrm{L}`,
:math:`y_\textrm{F}`, :math:`z_\textrm{F}`). While we assume that the
misalignment state is constant for each 10 minute observation period,
the other variables (:math:`x_\textrm{L}`, :math:`y_\textrm{F}`,
:math:`z_\textrm{F}`) are available on a per-particle basis, combining
observations from both cameras. Therefore, we can use a Bayesian inverse
Optimal Estimation retrieval (Rodgers, 2000)
implemented by the pyOptimalEstimation library
(Maahn et al. 2020) to retrieve the misalignment
state from the actual observed :math:`z_\textrm{L}`.

The retrieved misalignment parameters are required for matching, but
retrieving the misalignment parameters requires matched particles. To
solve this dilemma, we use an iterative method assuming that
misalignment does not change suddenly. The method starts by using the
misalignment estimates and uncertainties (inflated by a factor of 10)
from the previous time period (10 minutes) to match particles of the
current time period. These particles are used to retrieve values for
:math:`\varphi`, :math:`\theta`, and :math:`O_{\textrm{f}z}` which are
used as a priori input for the next iteration of misalignment retrieval.
The iteration is stopped when the changes in :math:`\varphi`,
:math:`\theta`, and :math:`O_{\textrm{f}z}` are less than the estimated
uncertainties. For efficiency, the iterative method is applied only to
the first 300 observed particles and the resulting coefficients are
stored in the metaRotation product. A drawback of the method is that
this processing step requires processing the 10-minute measurement
chunks in chronological order, creating a serial bottleneck in the
otherwise parallel VISSS processing chain. Obviously, this method does
not work when no information is available from the previous time step,
e.g., after the instrument was set up or adjusted. To get the starting
point for the iteration, the matching algorithm is applied for frames
where only a single, relatively large (:math:`>` 10 px) particle is
detected, so that the matching can be done based on particle height
difference (:math:`\Delta h`) alone, ignoring vertical offset
(:math:`\Delta z`).




Run metaRotation
----------------

Run metaRotation with

.. autofunction:: VISSSlib.matching.createMetaRotation
   :no-index:

``createMetaRotation`` is decorated with :func:`VISSSlib.tools.loopify`, which turns
the per-case function into one that loops over a case range, or in a shell script with

.. code:: console

    python3 -m VISSSlib matching.createMetaRotation $config_file.yaml $case --skip-existing

where ``$case`` is either a number of days to look back, or ``YYYYMMDD``/``YYYYMMDD-YYYYMMDD``.
The command is **not** parallelized because previous results are typically required.

Manual adjustments
------------------



To apply metaRotation for new deployments or when the instrument has been moved, the following code is recommended. Load libraries and set case

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import VISSSlib
    import importlib
    import yaml
    
    print(VISSSlib.__version__)
    
    settings = "/projekt1//ag_maahn/VISSS_config/hyytiala2_v3.yaml"
    case = "20240217-0940" 

    print(case)

    config = VISSSlib.tools.readSettings(settings)
    fl = VISSSlib.files.FindFiles(case, config.leader, config)
    fname1L = fl.listFiles("level1detect")[0]
    fname1Match = VISSSlib.files.FilenamesFromLevel(fname1L, config).fname["level1match"]


Use generic first guess rotation values with high uncertainties

.. code:: ipython3

    rotate_default = pd.Series(
        {
            "camera_phi": 0.0,
            "camera_theta": 0.0,
            "camera_Ofz": 0,
        }
    )
    #
    rotate_err_default = pd.Series(
        {
            "camera_phi": 1,
            "camera_theta": 1,
            "camera_Ofz": 50,
        }
    )

In the first step, use only large particles and do not use Z difference at all to allow
arbitrary offsets

The higher you set minDMax4rot (minimum Dmax of particles used), the better the results.

The option singleParticleFramesOnly makes sure only frames are used where a single particle is observed.

The value nSamples4rot can be increased if necessary but makes the estimation slow. 

.. code:: ipython3

    fout, matchedDat, rot, rot_err, _, _, _, errors = VISSSlib.matching.matchParticles(
        fname1L,
        config,
        doRot=True,
        rotationOnly=True,
        rotate=rotate_default,
        rotate_err=rotate_err_default,
        maxDiffMs="config",
        testing=False,
        minSamples4rot=90,
        minDMax4rot=15,
        singleParticleFramesOnly=True,
        nSamples4rot=1000,
        sigma={
            #             "Z" : 1.7, # estimated from OE results
            "H": 1.2,  # estimated from OE results
            "I": 0.01,
        },
    )
    rot, rot_err

Resulting in

.. parsed-literal::

    (camera_phi      -0.403621
     camera_theta    -0.753237
     camera_Ofz      79.605159
     dtype: float64,
     camera_phi      0.024806
     camera_theta    0.026622
     camera_Ofz      0.323117
     dtype: float64)

The results are used to run the algorithm again, but this time with default settings



.. code:: ipython3

    
    fout, matchedDat, rot2, rot_err2, nL, nF, nM, errors = VISSSlib.matching.matchParticles(
        fname1L,
        config,
        doRot=True,
        rotationOnly=True,
        rotate=rot,
        rotate_err=rot_err,
        nPoints=500,
        minSamples4rot=40,
    )
    nL, nF, nM

And again

.. code:: ipython3

    fout, matchedDat, rot3, rot_err3, _, _, _, errors = VISSSlib.matching.matchParticles(
        fname1L,
        config,
        doRot=True,
        rotationOnly=True,
        rotate=rot2,
        rotate_err=rot_err2,
        nPoints=500,
        minSamples4rot=40,
    )

And again


.. code:: ipython3

    fout, matchedDat, rot4, rot_err4, _, _, _, errors = VISSSlib.matching.matchParticles(
        fname1L,
        config,
        doRot=True,
        rotationOnly=True,
        rotate=rot3,
        rotate_err=rot_err3,
        nPoints=500,
        minSamples4rot=5,
    )

Now format the output so that we can copy paste it in the config files

.. code:: ipython3

    print(
        yaml.dump(
            {
                "rotate": {
                    case: {
                        "transformation": rot4.round(6).to_dict(),
                        "transformation_err": rot_err4.round(6).to_dict(),
                    }
                }  #
            }
        )
    )


.. parsed-literal::

    rotate:
      20240228-0340:
        transformation:
          camera_Ofz: 76.795276
          camera_phi: 0.491272
          camera_theta: -1.000598
        transformation_err:
          camera_Ofz: 0.310487
          camera_phi: 0.044387
          camera_theta: 0.021347
    

Coordinate transformation derivation
-------------------------------------------

This is the derivation behind the forward-operator equation used above and
implemented by :func:`VISSSlib.matching.shiftRotate_F2L` /
:func:`VISSSlib.matching.rotate_F2L` and friends in :doc:`matching`
(moved here from the paper's appendix, since it directly supports this
retrieval and previously had no other home in this documentation — see
:doc:`visss_paper_r2` for the rest of the hardware/paper background).

We use a right handed coordinate system
(:math:`x`,\ :math:`y`,\ :math:`z`) to define the position of particles
in the observation volume, where :math:`z` points to the ground. The
follower coordinate system
(:math:`x_\textrm{F}`,\ :math:`y_\textrm{F}`,\ :math:`z_\textrm{F}`) can
be transformed into the leader coordinate system
(:math:`x_\textrm{L}`,\ :math:`y_\textrm{L}`,\ :math:`z_\textrm{L}`) by
the standard transformation matrix

.. math::

   \begin{aligned}
    \begin{pmatrix} x_\textrm{L} \\y_\textrm{L} \\ z_\textrm{L} \end{pmatrix} &=
     \begin{pmatrix}
      \cos \theta \cos \psi &
      \sin \varphi \sin \theta \cos \psi - \cos \varphi \sin \psi &
      \cos \varphi \sin \theta \cos \psi + \sin \varphi \sin \psi \\
       \cos \theta \sin \psi &
       \sin \varphi \sin \theta \sin \psi + \cos \varphi \cos \psi &
       \cos \varphi \sin \theta \sin \psi - \sin \varphi \cos \psi \\
       -\sin \theta &
       \sin \varphi \cos \theta &
       \cos \varphi \cos \theta
     \end{pmatrix}
     \begin{pmatrix} x_\textrm{F}' \\y_\textrm{F}' \\ z_\textrm{F}' \end{pmatrix}

   \end{aligned}

using the follower’s roll :math:`\varphi`, yaw :math:`\psi`, and pitch
:math:`\theta`, analogous to airborne measurements, and with
:math:`x_\textrm{F}' = x_\textrm{F} + O_{\textrm{f}x}`,
:math:`y_\textrm{F}' = y_\textrm{F} + O_{\textrm{f}y}`, and
:math:`z_\textrm{F}' = z_\textrm{F} + O_{\textrm{f}z}`, where
:math:`O_{\textrm{f}x}`, :math:`O_{\textrm{f}y}`, and
:math:`O_{\textrm{f}z}` are the offsets of the follower coordinate
system in the :math:`x`, :math:`y`, and :math:`z` directions,
respectively. Offsets in
:math:`O_{\textrm{f}x}` and :math:`O_{\textrm{f}y}` are neglected,
because they would only materialize in reduced particle sharpness, but
not in the retrieved three-dimensional position. The opposite
transformation can be described by:

.. math::

   \begin{aligned}
     \begin{pmatrix} x_\textrm{F}' \\y_\textrm{F}' \\ z_\textrm{F}' \end{pmatrix} &=
     \begin{pmatrix}
      \cos \theta \cos \psi &
       \cos \theta \sin \psi &
      -\sin \theta \\
       \sin \varphi \sin \theta \cos \psi - \cos \varphi \sin \psi &
       \sin \varphi \sin \theta \sin \psi + \cos \varphi \cos \psi &
       \sin \varphi \cos \theta \\
       \cos \varphi \sin \theta \cos \psi + \sin \varphi \sin \psi &
       \cos \varphi \sin \theta \sin \psi - \sin \varphi \cos \psi &
       \cos \varphi \cos \theta
     \end{pmatrix}
     \begin{pmatrix} x_\textrm{L} \\y_\textrm{L} \\ z_\textrm{L} \end{pmatrix}
   \end{aligned}

Since we have only one measurement in the :math:`x` and :math:`y`
dimensions, but two in :math:`z`, we use the difference between the
measured :math:`z_\textrm{L}` and the estimated :math:`z_\textrm{L}`
from matched particles to retrieve the misalignment angles and offsets

.. math::

   \label{eq:zl}
       z_\textrm{L} =           -\sin\theta x_\textrm{F}' +
           \sin\varphi \cos\theta y_\textrm{F}' +
           \cos\varphi \cos\theta z_\textrm{F}'.

In this equation, :math:`x_\textrm{F}'` is unknown so it is derived from

.. math::

   \label{eq:xf}
       x_\textrm{F}' = \cos\theta \cos\psi x_\textrm{L} +
       \cos\theta \sin\psi y_\textrm{L} -
       \sin\theta z_\textrm{L}

where, in turn :math:`y_\textrm{L}` is not observed. Therefore,
:math:`y_\textrm{L}` is obtained from

.. math::

   \begin{split}
   \label{eq:yl}
       y_\textrm{L} = \cos\theta \sin\psi x_\textrm{F}'
       + (\sin\varphi \sin\theta \sin\psi + \cos\varphi \cos\psi) y_\textrm{F}'
       + (\cos\varphi \sin\theta \sin\psi - \sin\varphi \cos\psi) z_\textrm{F}'.
   \end{split}

Inserting equations `[eq:yl] <#eq:yl>`__ into `[eq:xf] <#eq:xf>`__
yields after a couple of simplifications

.. math::

   \begin{aligned}
   \begin{split}
   \label{eq:xf3}
       x_\textrm{F}'  & = \frac{\cos\theta \cos\psi}{1 - \cos^2\theta \sin^2\psi} x_\textrm{L} \\
       & + \frac{(\cos\theta \sin\varphi \sin\theta \sin^2\psi + \cos\varphi \cos\psi \cos\theta \sin\psi )}{1 - \cos^2\theta \sin^2\psi} y_\textrm{F}' \\
       & + \frac{(\cos\theta \cos\varphi \sin\theta \sin^2\psi - \sin\varphi \cos\psi \cos\theta \sin\psi )}{1 - \cos^2\theta \sin^2\psi} z_\textrm{F}' \\
       & - \frac{\sin\theta }{1 - \cos^2\theta \sin^2\psi} z_\textrm{L}.
   \end{split}
   \end{aligned}

Inserting equations `[eq:xf3] <#eq:xf3>`__ into `[eq:zl] <#eq:zl>`__
yields:

.. math::

   \begin{aligned}
   \begin{split}
    \label{eq:zl2}
       z_\textrm{L} =   -& \frac{  \sin\theta }{\cos\theta \cos\psi} x_\textrm{L} \\
        -& \frac{\sin \theta \sin \psi \cos \varphi - \cos \psi \sin \varphi}{\cos\theta \cos\psi} y_\textrm{F}' \\+&  \frac{\sin \theta \sin \psi \sin \varphi + \cos \psi \cos \varphi}{\cos\theta \cos\psi}z_\textrm{F}' .
   \end{split}
   \end{aligned}

We have no information about :math:`\psi`, therefore we have no choice
but assuming :math:`\psi = 0` leading to

.. math::

   \begin{aligned}
   \begin{split}
       z_\textrm{L} =   -& \frac{  \sin\theta }{\cos\theta } x_\textrm{L}      + \frac{\sin \varphi}{\cos\theta } y_\textrm{F}'      +  \frac{\cos \varphi}{\cos\theta }z_\textrm{F}' .
   \end{split}
   \end{aligned}

API
---

metaRotation is handled in matching.py, see :doc:`matching`