level1match - match detected particles of both cameras
======================================================


Motivation
----------

The particle detection of each camera is completely separate, so the
particles observed by each camera must be combined. This particle
combination allows for the particle position to be determined in a
three-dimensional reference coordinate system. As a side effect, this
constrains the observation volume by discarding particles outside of the
intersection of their observation volumes, i.e. observed by only one
camera. We use a right-handed reference coordinate system
(:math:`x`,\ :math:`y`,\ :math:`z`) with :math:`z` pointing to the
ground to define the position of particles in the observation volume
(Fig. `1 <#fig:concept>`__). In the absence of an absolute reference, we
attach the coordinate system to the leader camera (i.e.,
(:math:`x_\textrm{L}`,\ :math:`y_\textrm{L}`,\ :math:`z_\textrm{L}`) =
(:math:`x`,\ :math:`y`,\ :math:`z`)) such that :math:`x = X_\textrm{L}`
and :math:`z = Y_\textrm{L}`, where :math:`X_\textrm{L}` and
:math:`Y_\textrm{L}` are the particle positions in the two dimensional
leader images. Note that small letters describe the three dimensional
coordinate system and capital letters describe the two dimensional
position on the images of the individual camera images. The missing
dimension :math:`y` is obtained from the follower camera with
:math:`y = -X_\textrm{F}` where :math:`X_\textrm{F}` the horizontal
position in the follower image.

The matching of the particles from both cameras is based on the
comparison of two variables: The vertical position of the particles and
their vertical extent. Due to measurement uncertainties, the agreement
of these variables cannot be perfect and they are treated
probabilistically. That is, it is assumed that the difference in
vertical extent :math:`\Delta h` (vertical position :math:`\Delta z`)
between the two cameras follows a normally distributed probability
density function (PDF) with mean zero and standard deviation 1.7 px (1.2
px), based on an analysis of manually matched particle pairs. To
determine the probability (of e.g., measuring a certain vertical
extent), the PDF is integrated over an interval of ±0.5 px representing
the discrete 1 px steps.

This process requires matching the observations of both cameras in time.
The internal clocks of the cameras ("capture time") can deviate by more
than 1 frame per 10 minutes. The time assigned by the computers ("record
time") is sometimes, but not always, distorted by computer load.
Therefore, the continuous frame index ("capture id") is used for
matching, but this requires determining the index offset between both
cameras at the start of each measurement (typically 10 minutes). For
this, the algorithm uses pairs of frames with observed particles that
are less than 1 ms (i.e. less than 1/4 of the measurement resolution)
apart in record time assuming that the lag due to computer load is only
sporadically increased. This allows the algorithm to identify the most
common capture id offset of the frame pairs. We found that this method
gives stable results for a subset of 500 frames. Similar to :math:`h`
and :math:`z`, the capture id offset :math:`\Delta i` is used as the
mean of a normal distribution with a standard deviation value of 0.01,
which ensures that only particles observed at the same time are matched.
During MOSAiC, the data acquisition computer CPUs turned out to be too
slow to keep up with processing during heavy snowfall. With the
additional impact of a bug in the data acquisition code and drifting
computer clocks when the network connection to the ship’s reference
clock were interrupted, the particle matching for the MOSAiC data set
often requires manual adjustment. These problems have been resolved for
later campaigns so that matching now works fully automatic.

The joint product of the probabilities from :math:`\Delta h`,
:math:`\Delta z`, and :math:`\Delta i` is considered a match score,
which describes the quality of the particle match. Manual inspection
revealed that the number of false matches increases strongly for match
scores less than 0.001, which is used as a cut-off criterion. Assuming
that the probabilities are correctly determined, this implies that 0.1%
of particle matches are falsely rejected, resulting in a negligible
bias.

For each particle, its three-dimensional position is provided and all
per-particle variables from the detection are carried forward to the
matched particle product level1match. The ratio of matched to observed
particles from a single camera varies with the average particle size,
since larger particles can be identified even when they are out of
focus, and varies between approximately 10% and 90%. [*]_




``VISSSlib.matching`` API
-------------------------


.. automodule:: VISSSlib.matching
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. [*] The former text has been copied from Maahn, M., D. Moisseev, I. Steinke, N. Maherndl, and M. D. Shupe, 2024: Introducing the Video In Situ Snowfall Sensor (VISSS). Atmospheric Measurement Techniques, 17, 899–919, doi:10.5194/amt-17-899-2024.