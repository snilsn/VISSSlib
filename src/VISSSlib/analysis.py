# -*- coding: utf-8 -*-

import datetime
import logging
import os
import sys
from copy import deepcopy

import numpy as np
import xarray as xr
from loguru import logger as log

from . import (
    __version__,
    av,
    detection,
    distributions,
    files,
    fixes,
    matching,
    metadata,
    quicklooks,
    tools,
    tracking,
)


class _stereoViewMatch(object):
    """Stereo view matcher for VISSS data analysis.

    This class handles viewing and matching stereo camera data for VISSS analysis.
    It manages video reading, particle detection, and data synchronization between
    two cameras.

    Attributes
    ----------
    case : str
        Case identifier for the data
    config : object
        Configuration object containing camera parameters
    version : str
        Version string for the analysis
    markParticles : bool
        Whether to mark particles in the displayed frames
    increaseContrast : bool
        Whether to increase image contrast
    showTracks : bool
        Whether to show tracking information
    cameras : list
        List of camera identifiers
    skipNonMatched : bool
        Whether to skip non-matched frames
    lv1match : object
        Level 1 match data
    """

    def __init__(
        self,
        case,
        config,
        version=__version__,
        markParticles=True,
        increaseContrast=True,
        showTracks=False,
        skipNonMatched=False,
        lv1match=None,
    ):
        """Initialize the stereo view matcher.

        Parameters
        ----------
        case : str
            Case identifier for the data
        config : object
            Configuration object containing camera parameters
        version : str, optional
            Version string for the analysis (default: __version__)
        markParticles : bool, optional
            Whether to mark particles in the displayed frames (default: True)
        increaseContrast : bool, optional
            Whether to increase image contrast (default: True)
        showTracks : bool, optional
            Whether to show tracking information (default: False)
        skipNonMatched : bool, optional
            Whether to skip non-matched frames (default: False)
        lv1match : object, optional
            Level 1 match data (default: None)
        """
        self.case = case
        self.config = tools.readSettings(config)
        self.version = version
        self.markParticles = markParticles
        self.increaseContrast = increaseContrast
        self.showTracks = showTracks
        self.cameras = [config.leader, config.follower]
        self.skipNonMatched = skipNonMatched

        self.lv1match = lv1match
        self.open()
        self.rr = 0

    def open(self):
        """Open and initialize all required data files and video readers."""
        fL = files.FindFiles(self.case, self.config.leader, self.config, self.version)
        assert (
            len(fL.listFiles("level0")) == 1
        ), f"Please select case so that this {fL.fnamesPattern.level0} results only in one file"
        fL1 = files.Filenames(fL.listFiles("level0")[0], self.config)

        #  open all the files
        fnamesLv0 = {}
        self.meta = {}
        self.lv1detect = {}
        self.videos = {}
        self.idDiffs = {}
        self.imagesL1detect = {}

        fnamesLv0[self.config.leader] = fL.listFiles("level0")[0]
        fnames0F = fL1.filenamesOtherCamera(graceInterval=-1, level="level0")
        if len(fnames0F) > 1:
            print(
                "Cannot handle camera restarts yet, taking only first file, omitting",
                fnames0F[1:],
            )
            fnames0F = fnames0F[:1]
        fnamesLv0[self.config.follower] = fnames0F[0]

        self.meta[self.config.leader] = tools.open_mfmetaFrames(
            fL.listFiles("metaFrames"), self.config
        )
        self.lv1detect[self.config.leader] = tools.open_mflevel1detect(
            fL.listFiles("level1detect"), self.config
        )
        try:
            self.imagesL1detect[self.config.leader] = fL.listFiles("imagesL1detect")[0]
        except:
            self.imagesL1detect[self.config.leader] = None
        fnamesMF = fL1.filenamesOtherCamera(graceInterval=-1, level="metaFrames")
        if len(fnamesMF) > 1:
            print(
                "Cannot handle camera restarts yet, taking only first file, omitting",
                fnamesMF[1:],
            )
            fnamesMF = fnamesMF[:1]
        fnames1F = fL1.filenamesOtherCamera(graceInterval=-1, level="level1detect")
        if len(fnames1F) > 1:
            print(
                "Cannot handle camera restarts yet, taking only first file, omitting",
                fnames1F[1:],
            )
            fnames1F = fnames1F[:1]

        self.meta[self.config.follower] = tools.open_mfmetaFrames(fnamesMF, self.config)
        self.lv1detect[self.config.follower] = tools.open_mflevel1detect(
            fnames1F, self.config
        )

        if self.lv1match is None:
            if not self.showTracks:
                self.lv1match = tools.open_mflevel1match(
                    fL.listFiles("level1match"), self.config
                )
            else:
                self.lv1match = tools.open_mflevel1match(
                    fL.listFiles("level1track"), self.config
                )
        else:
            self.lv1match = self.lv1match.rename({"pair_id": "fpair_id"})
            self.lv1match["pair_id"] = self.lv1match.fpair_id
        try:
            self.imagesL1detect[self.config.follower] = fL1.filenamesOtherCamera(
                graceInterval=-1, level="imagesL1detect"
            )[0]
        except:
            self.imagesL1detect[self.config.follower] = None

        # #get capture ID diffs
        # self.idDiff, nMatched = tools.estimateCaptureIdDiffCore(*self.meta.values(), "capture_time", nPoints=500, timeDim="record_time")
        # idDiff2, nMatched = tools.estimateCaptureIdDiffCore(*self.lv1detect.values(), "fpid", nPoints=500, timeDim="record_time")
        # self.idDiffs[self.config.follower] = self.idDiff
        # assert  self.idDiff == idDiff2, "estimateCaptureIdDiff did not come to same result for metaFrames and lv1detect"
        self.idDiffs[self.config.follower] = self.idDiff = int(
            self.lv1match.capture_id.diff("camera").values.flatten()[0]
        )
        self.idDiffs[self.config.leader] = 0

        # remove non moving data
        minMovingPixels = np.array(self.config.level1detect.minMovingPixels)
        for camera in self.cameras:
            nChangedPixel = self.meta[camera].nMovingPixel.values
            passesThreshold = nChangedPixel >= minMovingPixels
            print(
                f"{passesThreshold.any(1).sum()/len(self.meta[camera].capture_time)*100}% frames are moving"
            )
            self.meta[camera] = self.meta[camera].isel(
                capture_time=passesThreshold.any(1)
            )

        self.uniqueCaptureIds = xr.DataArray(
            np.sort(
                np.unique(
                    np.concatenate(
                        (
                            self.meta[self.config.leader].capture_id,
                            self.meta[self.config.follower].capture_id - self.idDiff,
                        )
                    )
                )
            ),
            dims=["merged_record_id"],
        )
        self.captureTimes = {}
        for camera in self.cameras:
            self.videos[camera] = av.VideoReaderMeta(
                fnamesLv0[camera],
                self.meta[camera],
                lv1detect=self.lv1detect[camera],
                lv1match=self.lv1match.sel(camera=camera),
                imagesL1detect=self.imagesL1detect[camera],
                config=self.config,
                safeMode=False,
                skipNonMatched=self.skipNonMatched,
            )
            self.captureTimes[camera] = xr.DataArray(
                self.meta[camera].capture_time,
                coords=[self.meta[camera].capture_id],
                dims=["capture_id"],
            )

    def get(self, rr):
        """Get frame and associated data for a specific record index.

        Parameters
        ----------
        rr : int
            Record index to retrieve

        Returns
        -------
        tuple
            Frame array, meta frames, level 1 detects, level 1 matches, particles
        """
        self.rr = rr
        thisID = self.uniqueCaptureIds[rr].values

        frame = list()
        metaFrames = []
        lv1detects = []
        lv1matches = []
        particles = {}
        for camera in self.cameras:
            particles[camera] = []
            if (thisID + self.idDiffs[camera]) in self.meta[camera].capture_id:
                captureTime = (
                    self.captureTimes[camera]
                    .sel(capture_id=thisID + self.idDiffs[camera])
                    .values
                )
                #                 captureTime = 0
                # print(f"found record {rr} in {camera} data at {captureTime}")

                # print(camera, captureTime)
                res, self.frame1, meta1, meta2, meta3, particle1 = self.videos[
                    camera
                ].getFrameByCaptureTimeWithParticles(
                    captureTime,
                    markParticles=self.markParticles,
                    highlightPid="meta",
                    increaseContrast=self.increaseContrast,
                    showTracks=self.showTracks,
                )
                if self.frame1 is not None:
                    frame.append(self.frame1)
                if particle1 is not None:
                    particles[camera] = particle1
                    # if meta2 is not None:

                    # for fpid in meta2.fpid.values:
                    # thisMeta2 = meta2.sel(fpid=fpid)
                    # cnt = thisMeta2.cnt.values[thisMeta2.cnt.values[...,0]>=0]
                    # thisFrame = self.frame1[self.config.level1detect.height_offset:,:,0]
                    # particleBoxMask, xOffset, yOffset = detection.extractRoi(thisMeta2.roi.values, cv2.fillPoly(np.zeros_like(thisFrame), pts =np.array([cnt], dtype=np.int32), color=255))
                    # particleBox, xOffset, yOffset = detection.extractRoi(thisMeta2.roi.values, thisFrame)
                    # particleBoxCropped = deepcopy(particleBox)
                    # particleBoxCropped[particleBoxMask == 0] = 255

                    # # particleBox = np.concatenate((particleBox, particleBoxMask),-1)
                    # particleBox = np.hstack((particleBox, particleBoxCropped ) )

                    # if np.prod(particleBoxCropped.shape) > 100:
                    #     factor = 2
                    # else:
                    #     factor = 4
                    # particleBox = skimage.transform.resize(particleBox,
                    #    np.array(particleBox.shape[:2])*factor,
                    #    mode='edge',
                    #    anti_aliasing=False,
                    #    anti_aliasing_sigma=None,
                    #    preserve_range=True,
                    #    order=0)

                    # if meta3 is None:
                    #     color = (255,0,0)
                    # elif thisMeta2.pid.values in meta3.pid.values:
                    #     color = (0,0,0)
                    # else:
                    #     color = (255,0,0)

                    # cv2.putText(particleBox, str(thisMeta2.pid.values), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                else:
                    frame.append(
                        np.zeros(
                            (
                                self.config.frame_height
                                + self.config.level1detect.height_offset,
                                self.config.frame_width,
                                3,
                            ),
                            dtype=int,
                        )
                        + 130
                    )
            else:
                # print(camera, "not found",(thisID + self.idDiffs[camera]) )
                # print(f"did not find record {rr} in {camera} data")
                frame.append(
                    np.zeros(
                        (
                            self.config.frame_height
                            + self.config.level1detect.height_offset,
                            self.config.frame_width,
                            3,
                        ),
                        dtype=int,
                    )
                )
                meta1 = meta2 = meta3 = None

            frameBorder = np.zeros(
                (
                    self.config.frame_height + self.config.level1detect.height_offset,
                    10,
                    3,
                ),
                dtype=int,
            )

            frame.append(frameBorder)
            metaFrames.append(meta1)
            lv1detects.append(meta2)
            lv1matches.append(meta3)

        try:
            lv1matches = xr.concat(lv1matches, dim="camera")
        except TypeError:
            lv1matches = None

        frame = np.concatenate(frame, axis=1)

        return frame, metaFrames, lv1detects, lv1matches, particles

    def next(self):
        """Get the next record in sequence.

        Returns
        -------
        tuple
            Frame array, meta frames, level 1 detects, level 1 matches, particles
        """
        newrr = self.rr + 1
        if newrr >= len(self.uniqueCaptureIds):
            print("end of movie file")
            return None, None, None, None, None
        return self.get(newrr)

    def previous(self):
        """Get the previous record in sequence.

        Returns
        -------
        tuple
            Frame array, meta frames, level 1 detects, level 1 matches, particles
        """
        newrr = self.rr - 1
        if newrr < 0:
            print("beginning of movie file")
            return None, None, None, None, None
        return self.get(newrr)

    def nextCommon(self):
        """Get the next record where both cameras have data.

        Returns
        -------
        tuple
            Frame array, meta frames, level 1 detects, level 1 matches, particles
        """
        while True:
            frame, metaFrames, lv1detects, lv1match, particles = self.next()
            if np.sum([m is None for m in metaFrames]) == 1:
                continue
            else:
                break

        return frame, metaFrames, lv1detects, lv1match, particles

    def previousCommon(self):
        """Get the previous record where both cameras have data.

        Returns
        -------
        tuple
            Frame array, meta frames, level 1 detects, level 1 matches, particles
        """
        while True:
            frame, metaFrames, lv1detects, lv1match, particles = self.previous()
            if np.sum([m is None for m in metaFrames]) == 1:
                continue
            else:
                break
        return frame, metaFrames, lv1detects, lv1match, particles

    def nextMatch(self):
        """Get the next record that has a match in the leader camera.

        Returns
        -------
        tuple
            Frame array, meta frames, level 1 detects, level 1 matches, particles
        """
        rr = self.rr
        rrs = range(rr + 1, len(self.uniqueCaptureIds))
        if len(rrs) == 0:
            print("end of movie file")
            return None, None, None, None, None
        for newrr in rrs:
            thisID = self.uniqueCaptureIds[newrr].values
            if thisID in self.lv1match.capture_id.sel(camera=self.config.leader).values:
                break
        return self.get(newrr)

    def previousMatch(self):
        """Get the previous record that has a match in the leader camera.

        Returns
        -------
        tuple
            Frame array, meta frames, level 1 detects, level 1 matches, particles
        """
        rr = self.rr
        rrs = range(rr - 1, -1, -1)
        if len(rrs) == 0:
            print("start of movie file")
            return None, None, None, None, None
        for newrr in rrs:
            thisID = self.uniqueCaptureIds[newrr].values
            if thisID in self.lv1match.capture_id.sel(camera=self.config.leader).values:
                break
        return self.get(newrr)

    def close(self):
        """Close all opened data files and video readers."""
        for camera in self.cameras:
            self.meta[camera].close()
            self.lv1detect[camera].close()
            self.videos[camera].release()


class matchGUI:
    """GUI for stereo view matching visualization.

    Provides interactive GUI controls for navigating and visualizing stereo camera
    matching data.

    Attributes
    ----------
    sv : _stereoViewMatch
        Stereo view matcher instance
    showVars : list
        Variables to display in the GUI
    showParticles : bool
        Whether to show particle details
    scale : float
        Scaling factor for displayed images
    """

    def __init__(
        self,
        case,
        config,
        markParticles=True,
        increaseContrast=False,
        showTracks=False,
        skipNonMatched=False,
        showVars=["Dmax"],
        lv1match=None,
        showParticles=False,
        scale=0.5,
    ):
        """Initialize the match GUI.

        Parameters
        ----------
        case : str
            Case identifier for the data
        config : object
            Configuration object containing camera parameters
        markParticles : bool, optional
            Whether to mark particles in the displayed frames (default: True)
        increaseContrast : bool, optional
            Whether to increase image contrast (default: False)
        showTracks : bool, optional
            Whether to show tracking information (default: False)
        skipNonMatched : bool, optional
            Whether to skip non-matched frames (default: False)
        showVars : list, optional
            Variables to display in the GUI (default: ["Dmax"])
        lv1match : object, optional
            Level 1 match data (default: None)
        showParticles : bool, optional
            Whether to show particle details (default: False)
        scale : float, optional
            Scaling factor for displayed images (default: 0.5)
        """
        self.sv = _stereoViewMatch(
            case,
            config,
            markParticles=markParticles,
            increaseContrast=increaseContrast,
            showTracks=showTracks,
            skipNonMatched=skipNonMatched,
            lv1match=lv1match,
        )
        self.showVars = showVars
        self.showParticles = showParticles
        self.scale = scale

        return

    def updateHandlesId(self, fid):
        """Update GUI handles for a specific frame ID.

        Parameters
        ----------
        fid : int
            Frame ID to update

        Returns
        -------
        tuple
            Updated frame, meta frames, level 1 detects, level 1 matches, particles
        """
        return self.updateHandles(*self.sv.get(fid))

    def updateHandles(self, frame, metaFrames, lv1detects, lv1matches, particles):
        """Update GUI handles with new data.

        Parameters
        ----------
        frame : array
            Frame data to display
        metaFrames : list
            Meta frames data
        lv1detects : list
            Level 1 detect data
        lv1matches : list
            Level 1 match data
        particles : dict
            Particle data
        """
        import cv2
        import skimage
        from IPython.display import Image, display

        self.metaFrames = metaFrames
        self.lv1detects = lv1detects
        self.lv1matches = lv1matches

        cc = 0
        mII = []
        if (self.lv1detects is not None) and (self.lv1detects[cc] is not None):
            for pid in self.lv1detects[cc].pid.values:
                mm = np.where(self.sv.lv1match.isel(camera=cc).pid == pid)[0]
                if len(mm) > 0:
                    mII.append(mm)
        if len(mII) > 0:
            lv1match = self.sv.lv1match.isel(fpair_id=np.concatenate(mII))
        else:
            lv1match = None

        _, frame = cv2.imencode(
            ".jpg",
            skimage.transform.resize(
                frame,
                (np.array(frame.shape[:2]) * self.scale).astype(int),
                mode="edge",
                anti_aliasing=True,
                preserve_range=True,
                order=0,
            ),
        )
        try:
            self.display_handle.update(Image(data=frame.tobytes()))
        except AttributeError:
            self.display_handle = display(Image(data=frame.tobytes()), display_id=True)

        try:
            self.setNN(self.sv.rr)
        except AttributeError:  # if no GUI available
            pass
        with self.out:
            self.out.clear_output()

            if self.showParticles and particles is not None:
                for k in particles:
                    if len(particles[k]) == 0:
                        continue
                    for pid, part in particles[k].items():
                        print(k, pid)
                        tools.displayImage(part, rescale=4)

            if metaFrames[0] is None:
                c0, i0, r0 = "n/a", "n/a", "n/a"
            else:
                c0, i0, r0 = (
                    metaFrames[0].capture_time.values,
                    metaFrames[0].capture_id.values,
                    metaFrames[0].record_id.values,
                )
            if metaFrames[1] is None:
                c1, i1, r1 = "n/a", "n/a", "n/a"
            else:
                c1, i1, r1 = (
                    metaFrames[1].capture_time.values,
                    metaFrames[1].capture_id.values,
                    metaFrames[1].record_id.values,
                )

            print("leader:", c0, i0, "follower:", c1, i1)
            if lv1match is not None:
                Zdiff = (
                    lv1match.position3D_centroid.isel(dim3D=[2, 3], drop=True)
                    .diff("dim3D")
                    .values.flatten()
                )
                # print(
                #     lv1match.pair_id.values.flatten(),
                #     ["%.7f score" % (l) for l in lv1match.matchScore.values],
                #     ["%.i ms" % (
                #         l/1e6) for l in lv1match.capture_time.diff("camera").values.astype(int).flatten()],
                #     ["%.2f y" % (l) for l in lv1match.position_upperLeft.diff(
                #         "camera").sel(dim2D="y", drop=True).values.flatten()],
                #     ["%.2f h" % (l) for l in lv1match.Droi.diff(
                #         "camera").sel(dim2D="y", drop=True).values.flatten()],
                #     ["%.2f Z" % z for z in Zdiff]
                # )
                for var in self.showVars:
                    print(var, lv1match[var].values)
                print("#" * 100)
            if (
                (lv1detects[0] is not None)
                and (lv1detects[1] is not None)
                and (lv1detects[0]["pid"].shape == lv1detects[1]["pid"].shape)
            ):
                print(
                    f"{'pid'.ljust(20)}: D {'X'.ljust(23)}, L {str(lv1detects[0]['pid'].values).ljust(23)}, F {str(lv1detects[1]['pid'].values).ljust(23)}"
                )
                for i in lv1detects[0].data_vars:
                    if i in [
                        "touchesBorder",
                        "pixPercentiles",
                        "nThread",
                        "record_id",
                        "cnt",
                    ]:
                        continue
                    print(
                        f"{i.ljust(20)}: D {str(lv1detects[0][i].values - lv1detects[1][i].values).ljust(23)}, L {str(lv1detects[0][i].values).ljust(23)}, F {str(lv1detects[1][i].values).ljust(23)}"
                    )

    def getNN(self):
        """Get current frame number from GUI input.

        Returns
        -------
        int
            Current frame number
        """
        nn = int(self.texts[0].get_interact_value())
        return nn

    def setNN(self, nn):
        """Set current frame number in GUI.

        Parameters
        ----------
        nn : int
            Frame number to set
        """
        self.texts[0].value = str(nn)

    def createGUI(self, pid=0, startId=0):
        """Create the GUI interface.

        Parameters
        ----------
        pid : int, optional
            Starting PID (default: 0)
        startId : int, optional
            Starting ID (default: 0)
        """
        import ipywidgets as widgets
        from IPython.display import Image, display

        self.out = widgets.Output()
        # set width and height
        layout = widgets.Layout(width="auto", height="30px")

        buttonNextMatch = widgets.Button(description=">>>", layout=layout)
        buttonPrevMatch = widgets.Button(description="<<<", layout=layout)
        buttonNextMatch.on_click(lambda x: self.updateHandles(*self.sv.nextMatch()))
        buttonPrevMatch.on_click(lambda x: self.updateHandles(*self.sv.previousMatch()))

        buttonNext = widgets.Button(description=">>", layout=layout)
        buttonPrev = widgets.Button(description="<<", layout=layout)
        buttonNext.on_click(lambda x: self.updateHandles(*self.sv.nextCommon()))
        buttonPrev.on_click(lambda x: self.updateHandles(*self.sv.previousCommon()))

        buttonNextFrame = widgets.Button(description=">", layout=layout)
        buttonPrevFrame = widgets.Button(description="<", layout=layout)
        buttonNextFrame.on_click(lambda x: self.updateHandles(*self.sv.next()))
        buttonPrevFrame.on_click(lambda x: self.updateHandles(*self.sv.previous()))

        buttons = [
            buttonPrevMatch,
            buttonNextMatch,
            buttonPrev,
            buttonNext,
            buttonPrevFrame,
            buttonNextFrame,
        ]
        self.texts = []

        self.texts.append(
            widgets.Text(
                value=str(pid),
                description=f"{len(self.sv.uniqueCaptureIds)} tot. ids",
                disabled=False,
                width="auto",
            )
        )

        load = widgets.Button(description="Load", layout=layout)
        load.on_click(lambda x: self.updateHandlesId(self.getNN()))
        self.texts.append(load)

        display_handle = None
        self.updateHandlesId(startId)

        self.statusP = widgets.HTML(
            value="-",
        )

        # displaying button and its output together
        buttonsH = widgets.HBox(buttons)
        statusH = widgets.HBox(self.texts)

        display(widgets.VBox([statusH, buttonsH, self.out]))

        return


class _stereoViewDetect(object):
    """Stereo view detector for VISSS data analysis.

    This class handles viewing stereo camera data for detection purposes.

    Attributes
    ----------
    case : str
        Case identifier for the data
    config : object
        Configuration object containing camera parameters
    version : str
        Version string for the analysis
    cameras : list
        List of camera identifiers
    markParticles : bool
        Whether to mark particles in the displayed frames
    this_capture_time : dict
        Capture times for each camera
    this_record_time : dict
        Record times for each camera
    """

    def __init__(self, case, config, version=__version__, markParticles=True):
        """Initialize the stereo view detector.

        Parameters
        ----------
        case : str
            Case identifier for the data
        config : object
            Configuration object containing camera parameters
        version : str, optional
            Version string for the analysis (default: __version__)
        markParticles : bool, optional
            Whether to mark particles in the displayed frames (default: True)
        """
        self.case = case
        self.config = tools.readSettings(config)
        self.version = version
        self.cameras = [config.leader, config.follower]
        self.version = version
        self.markParticles = markParticles

        self.this_capture_time = {}
        self.this_record_time = {}

        self.open()

    def open(self):
        """Open and initialize all required data files and video readers."""
        fL = files.FindFiles(self.case, self.config.leader, self.config, self.version)
        assert (
            len(fL.listFiles("level0")) == 1
        ), f"Please select case so that this {fL.fnamesPattern.level0} results only in one file"
        fL1 = files.Filenames(fL.listFiles("level0")[0], self.config)

        #  open all the files
        fnamesLv0 = {}
        self.meta = {}
        self.lv1detect = {}
        self.videos = {}
        self.idDiffs = {}

        fnamesLv0[self.config.leader] = fL.listFiles("level0")[0]
        fnames0F = fL1.filenamesOtherCamera(graceInterval=-1, level="level0")
        if len(fnames0F) > 1:
            print(
                "Cannot handle camera restarts yet, taking only first file, omitting",
                fnames0F[1:],
            )
            fnames0F = fnames0F[:1]
        fnamesLv0[self.config.follower] = fnames0F[0]

        self.meta[self.config.leader] = tools.open_mfmetaFrames(
            fL.listFiles("metaFrames"), self.config
        )
        self.lv1detect[self.config.leader] = tools.open_mflevel1detect(
            fL.listFiles("level1detect"), self.config
        )

        fnamesMF = fL1.filenamesOtherCamera(graceInterval=-1, level="metaFrames")
        if len(fnamesMF) > 1:
            print(
                "Cannot handle camera restarts yet, taking only first file, omitting",
                fnamesMF[1:],
            )
            fnamesMF = fnamesMF[:1]
        fnames1F = fL1.filenamesOtherCamera(graceInterval=-1, level="level1detect")
        if len(fnames1F) > 1:
            print(
                "Cannot handle camera restarts yet, taking only first file, omitting",
                fnames1F[1:],
            )
            fnames1F = fnames1F[:1]

        self.meta[self.config.follower] = tools.open_mfmetaFrames(fnamesMF, self.config)
        self.lv1detect[self.config.follower] = tools.open_mflevel1detect(
            fnames1F, self.config
        )

        self.index = {}
        for camera in self.cameras:
            self.videos[camera] = av.VideoReaderMeta(
                fnamesLv0[camera],
                self.meta[camera],
                lv1detect=self.lv1detect[camera],
                lv1match=None,
                config=self.config,
                safeMode=False,
            )

            self.index[camera] = range(len(self.meta[camera].capture_time))

        return

    def get(self, rrs):
        """Get frame and associated data for specific record indices.

        Parameters
        ----------
        rrs : dict
            Dictionary mapping camera names to record indices

        Returns
        -------
        tuple
            Frame array, meta frames, level 1 detects
        """
        self.rrs = rrs

        frame = list()
        metaFrames = []
        lv1detects = []

        for camera in self.cameras:
            rr = rrs[camera]
            if rr in self.index[camera]:
                #                 captureTime = 0
                # print(f"found record {rr} in {camera} data at {captureTime}")

                res, self.frame1, meta1, meta2, _, _ = self.videos[
                    camera
                ].getFrameByIndexWithParticles(rr, markParticles=self.markParticles)
                if self.frame1 is not None:
                    frame.append(self.frame1)
                else:
                    frame.append(
                        np.zeros(
                            (
                                self.config.frame_height
                                + self.config.level1detect.height_offset,
                                self.config.frame_width,
                                3,
                            ),
                            dtype=int,
                        )
                        + 130
                    )
                self.this_capture_time[camera] = meta1.capture_time.values
                self.this_record_time[camera] = meta1.record_time.values
            else:
                print(f"did not find record {rr} in {camera} data")
                frame.append(
                    np.zeros(
                        (
                            self.config.frame_height
                            + self.config.level1detect.height_offset,
                            self.config.frame_width,
                            3,
                        ),
                        dtype=int,
                    )
                )
                meta1 = meta2 = None
                self.this_capture_time[camera] = None
                self.this_record_time[camera] = None
            frame.append(
                np.zeros(
                    (
                        self.config.frame_height
                        + self.config.level1detect.height_offset,
                        10,
                        3,
                    ),
                    dtype=int,
                )
            )
            metaFrames.append(meta1)
            lv1detects.append(meta2)

        frame = np.concatenate(frame, axis=1)
        return frame, metaFrames, lv1detects

    def next(self, camera):
        """Get the next record for specified camera(s).

        Parameters
        ----------
        camera : str
            Camera name or 'all' for both cameras

        Returns
        -------
        tuple
            Frame array, meta frames, level 1 detects
        """
        if camera == "all":
            cameras = self.cameras
        else:
            cameras = [camera]
        newrr = deepcopy(self.rrs)

        for camera in cameras:
            newrr[camera] += 1
            if newrr[camera] >= len(self.index[camera]):
                print("end of movie file", newrr)
                return None, None, None
        return self.get(newrr)

    def previous(self, camera):
        """Get the previous record for specified camera(s).

        Parameters
        ----------
        camera : str
            Camera name or 'all' for both cameras

        Returns
        -------
        tuple
            Frame array, meta frames, level 1 detects
        """
        if camera == "all":
            cameras = self.cameras
        else:
            cameras = [camera]
        newrr = deepcopy(self.rrs)

        for camera in cameras:
            newrr[camera] -= 1
            if newrr[camera] < 0:
                print("beginning of movie file")
                return None, None, None
        return self.get(newrr)

    def matchTime(self, time1, camera):
        """Match time between cameras.

        Parameters
        ----------
        time1 : str
            Time variable to match ('capture_time' or 'record_time')
        camera : str
            Camera to match against

        Returns
        -------
        tuple
            Frame array, meta frames, level 1 detects
        """
        refCam = tools.otherCamera(self.config, camera)
        nn = self.rrs
        tDiff = np.abs(
            getattr(self, f"this_{time1}")[refCam] - self.meta[camera][time1].values
        )
        minII = np.argmin(tDiff)
        nn[camera] = minII
        #         print('found time difference', time1, camera, nn)

        return self.get(nn)

    def close(self):
        """Close all opened data files and video readers."""
        for camera in self.cameras:
            self.meta[camera].close()
            self.lv1detect[camera].close()
            self.videos[camera].release()


class manualMatchGUI:
    """Manual matching GUI for stereo view analysis.

    Provides interactive GUI controls for manual stereo camera matching.

    Attributes
    ----------
    sv : _stereoViewDetect
        Stereo view detector instance
    """

    def __init__(self, case, config, markParticles=True):
        """Initialize the manual match GUI.

        Parameters
        ----------
        case : str
            Case identifier for the data
        config : object
            Configuration object containing camera parameters
        markParticles : bool, optional
            Whether to mark particles in the displayed frames (default: True)
        """
        self.sv = _stereoViewDetect(case, config, markParticles=markParticles)
        return

    def updateHandles(self, frame, metaFrames, lv1detects):
        """Update GUI handles with new data.

        Parameters
        ----------
        frame : array
            Frame data to display
        metaFrames : list
            Meta frames data
        lv1detects : list
            Level 1 detect data
        """
        import cv2
        from IPython.display import Image, display

        self.metaFrames = metaFrames
        self.lv1detects = lv1detects

        cc = 0
        mII = []

        if frame is None:
            return

        _, frame = cv2.imencode(".jpeg", frame)
        try:
            self.display_handle.update(Image(data=frame.tobytes()))
        except AttributeError:
            self.display_handle = display(Image(data=frame.tobytes()), display_id=True)

        self.setNN(self.sv.rrs)
        with self.out:
            self.out.clear_output()
            if metaFrames[0] is None:
                c0, i0 = "n/a", "n/a"
            else:
                c0, i0 = (
                    metaFrames[0].capture_time.values,
                    metaFrames[0].capture_id.values,
                )
            if metaFrames[1] is None:
                c1, i1 = "n/a", "n/a"
            else:
                c1, i1 = (
                    metaFrames[1].capture_time.values,
                    metaFrames[1].capture_id.values,
                )

            print("leader:", c0, i0, "follower:", c1, i1)

            if (lv1detects[0] is not None) and (lv1detects[1] is not None):
                print(
                    f"{'pid'.ljust(20)}: D {'X'.ljust(23)}, L {str(lv1detects[0]['pid'].values).ljust(23)}, F {str(lv1detects[1]['pid'].values).ljust(23)}"
                )
                for i in lv1detects[0].data_vars:
                    if i in [
                        "touchesBorder",
                        "pixPercentiles",
                        "nThread",
                        "record_id",
                    ]:
                        continue
                    print(
                        f"{i.ljust(20)}: D {str(lv1detects[0][i].values - lv1detects[1][i].values).ljust(23)}, L {str(lv1detects[0][i].values).ljust(23)}, F {str(lv1detects[1][i].values).ljust(23)}"
                    )

        return

    def getNN(self):
        """Get current frame numbers from GUI inputs.

        Returns
        -------
        dict
            Dictionary mapping camera names to frame numbers
        """
        nn = {}
        for ii, camera in enumerate(self.sv.config.instruments):
            nn[camera] = int(self.texts[ii].get_interact_value())
        return nn

    def setNN(self, nn):
        """Set current frame numbers in GUI.

        Parameters
        ----------
        nn : dict
            Dictionary mapping camera names to frame numbers
        """
        for ii, camera in enumerate(self.sv.config.instruments):
            self.texts[ii].value = str(nn[camera])

    def createGUI(self):
        """Create the manual matching GUI interface."""
        import ipywidgets as widgets
        from IPython.display import Image, display

        self.out = widgets.Output()

        # set width and height
        layout = widgets.Layout(width="auto", height="30px")

        buttons = []

        #             buttonNext = widgets.Button(description='>>', layout=layout)
        #             buttonPrev = widgets.Button(description='<<', layout=layout)
        #             buttonNext.on_click(lambda x: self.updateHandles(*self.sv.nextCommon()))
        #             buttonPrev.on_click(lambda x: self.updateHandles(*self.sv.previousCommon()))

        buttonNextFrame = widgets.Button(description=f">", layout=layout)
        buttonPrevFrame = widgets.Button(description=f"<", layout=layout)
        buttonNextFrame.on_click(lambda x: self.updateHandles(*self.sv.next("all")))
        buttonPrevFrame.on_click(lambda x: self.updateHandles(*self.sv.previous("all")))
        buttons += [buttonPrevFrame, buttonNextFrame]

        buttonNextFrame1 = widgets.Button(
            description=f"{self.sv.config.instruments[0]} >", layout=layout
        )
        buttonPrevFrame1 = widgets.Button(
            description=f"{self.sv.config.instruments[0]} <", layout=layout
        )
        buttonNextFrame1.on_click(
            lambda x: self.updateHandles(
                *self.sv.next(deepcopy(self.sv.config.instruments[0]))
            )
        )
        buttonPrevFrame1.on_click(
            lambda x: self.updateHandles(
                *self.sv.previous(deepcopy(self.sv.config.instruments[0]))
            )
        )
        buttons += [buttonPrevFrame1, buttonNextFrame1]

        buttonNextFrame2 = widgets.Button(
            description=f"{self.sv.config.instruments[1]} >", layout=layout
        )
        buttonPrevFrame2 = widgets.Button(
            description=f"{self.sv.config.instruments[1]} <", layout=layout
        )
        buttonNextFrame2.on_click(
            lambda x: self.updateHandles(*self.sv.next(self.sv.config.instruments[1]))
        )
        buttonPrevFrame2.on_click(
            lambda x: self.updateHandles(
                *self.sv.previous(self.sv.config.instruments[1])
            )
        )
        buttons += [buttonPrevFrame2, buttonNextFrame2]

        self.texts = []

        startId = {}
        for ii, camera in enumerate(self.sv.config.instruments):
            startId[camera] = 0

        for ii, camera in enumerate(self.sv.config.instruments):
            w = widgets.Text(
                value=str(startId[camera]),
                description=f"{camera}: {len(self.sv.lv1detect[camera].capture_id)} tot. ids",
                disabled=False,
                width=500,
            )
            # w.observe(lambda x: self.updateHandles(*self.sv.get(self.getNN()))) #doesnt work?!
            self.texts.append(w)

        load = widgets.Button(description="Load", layout=layout)
        load.on_click(lambda x: self.updateHandles(*self.sv.get(self.getNN())))
        self.texts.append(load)

        matching = []

        match1 = widgets.Button(
            description="Match CT of %s" % self.sv.config.instruments[0], layout=layout
        )
        match1.on_click(
            lambda _: self.updateHandles(
                *self.sv.matchTime("capture_time", self.sv.config.instruments[0])
            )
        )
        matching.append(match1)
        match1a = widgets.Button(
            description="Match CT of %s" % self.sv.config.instruments[1], layout=layout
        )
        match1a.on_click(
            lambda _: self.updateHandles(
                *self.sv.matchTime("capture_time", self.sv.config.instruments[1])
            )
        )
        matching.append(match1a)

        match2 = widgets.Button(
            description="Match RT of %s" % self.sv.config.instruments[0], layout=layout
        )
        match2.on_click(
            lambda _: self.updateHandles(
                *self.sv.matchTime("record_time", self.sv.config.instruments[0])
            )
        )
        matching.append(match2)
        match2a = widgets.Button(
            description="Match RT of %s" % self.sv.config.instruments[1], layout=layout
        )
        match2a.on_click(
            lambda _: self.updateHandles(
                *self.sv.matchTime("record_time", self.sv.config.instruments[1])
            )
        )
        matching.append(match2a)

        display_handle = None
        self.updateHandles(*self.sv.get(startId))

        self.statusP = widgets.HTML(
            value="-",
        )

        # displaying button and its output together
        buttonsH = widgets.HBox(buttons)
        statusH = widgets.HBox(self.texts)
        matchingH = widgets.HBox(matching)

        display(widgets.VBox([statusH, buttonsH, matchingH, self.out]))

        return
