# -*- coding: utf-8 -*-
import functools
import os
import sys
import warnings
import zipfile

import numpy as np
import xarray as xr
from loguru import logger as log

from . import files, tools

__all__ = ["VideoReader", "VideoReaderMeta"]


def create_VideoReader():
    """
    Create a VideoReader class that extends cv2.VideoCapture.

    This function creates a specialized video reader class that adds caching
    functionality for frame retrieval, allowing efficient access to specific
    frames without re-reading the entire video stream.

    Returns
    -------
    class
        A VideoReader class that inherits from cv2.VideoCapture with additional
        frame caching capabilities.
    """
    import cv2

    class VideoReader(cv2.VideoCapture):
        """
        Video reader class with frame caching capabilities.

        This class extends cv2.VideoCapture to provide efficient frame retrieval
        with caching to avoid re-reading the entire video stream when accessing
        specific frames multiple times.

        Attributes
        ----------
        None
        """

        @functools.lru_cache(maxsize=100, typed=False)
        def getFrameByIndex(self, ii, safeMode=False):
            """
            Retrieve a specific frame by index with caching.

            This method retrieves a frame at a specific index from the video,
            caching the result for efficiency. It supports safe mode which
            prevents backward seeking.

            Parameters
            ----------
            ii : int
                Frame index to retrieve
            safeMode : bool, optional
                If True, prevents seeking backwards in the video, default is False

            Returns
            -------
            tuple
                A tuple containing (success_flag, frame_data) where success_flag
                indicates if the operation was successful and frame_data is the
                retrieved frame or None if unsuccessful.

            Raises
            ------
            RuntimeError
                If safeMode is enabled and backward seeking is attempted.
            """
            if int(self.get(cv2.CAP_PROP_POS_FRAMES)) != ii:
                if safeMode:
                    if int(self.get(cv2.CAP_PROP_POS_FRAMES)) < ii:
                        while int(self.get(cv2.CAP_PROP_POS_FRAMES)) < ii:
                            # print('fast forwarding', int(self.get(cv2.CAP_PROP_POS_FRAMES)), ii, )
                            _, _ = self.read()
                    elif int(self.get(cv2.CAP_PROP_POS_FRAMES)) > ii:
                        raise RuntimeError("Cannot go back in save mode")
                else:
                    self.set(cv2.CAP_PROP_POS_FRAMES, ii)
            res, frame = self.read()
            if frame is not None:
                frame = cvtColor(frame)
            return res, frame

        @property
        def position(self):
            """
            Get the current frame position in the video.

            Returns
            -------
            int
                Current frame position in the video stream.
            """
            return int(self.get(cv2.CAP_PROP_POS_FRAMES))

        @property
        def total_frames(self):
            """
            Get the total number of frames in the video.

            Returns
            -------
            int
                Total number of frames in the video file.
            """
            return int(self.get(cv2.CAP_PROP_FRAME_COUNT))

    return VideoReader


# can cause segfaults!


class VideoReaderMeta(object):
    """
    Metadata manager for video readers with associated data.

    This class manages video reading operations along with associated metadata
    and detection data for stereo camera systems. It handles multiple threads
    and provides methods for frame retrieval with particle annotations.

    Attributes
    ----------
    metaFrames : xarray.Dataset
        Metadata about frames including capture times and thread information
    lv1detect : xarray.Dataset or None
        Level 1 detection data
    lv1match : xarray.Dataset or None
        Level 1 matching data
    tarFile : object or None
        Archive file handle for particle images
    safeMode : bool
        Whether to enable safe mode for frame seeking
    config : object or None
        Configuration object
    skipNonMatched : bool
        Whether to skip non-matched particles
    movFilePattern : str
        Pattern for video file names
    threads : array
        Unique thread identifiers
    video : dict
        Dictionary mapping thread IDs to video readers
    position : int
        Current frame position
    positions : dict
        Dictionary mapping thread IDs to their current positions
    currentThread : int or None
        Currently active thread ID
    currentFrame : array or None
        Current frame data
    currentMetaFrames : xarray.Dataset or None
        Current metadata frames
    currentlv1detect : xarray.Dataset or None
        Current level 1 detection data
    currentPids : array or None
        Current particle IDs
    """

    def __init__(
        self,
        movFilePattern,
        metaFrames,
        lv1detect=None,
        lv1match=None,
        imagesL1detect=None,
        safeMode=False,
        config=None,
        skipNonMatched=False,
    ):
        """
        Initialize VideoReaderMeta with video and metadata.

        Parameters
        ----------
        movFilePattern : str
            Pattern for video file names
        metaFrames : str or xarray.Dataset
            Metadata frames file path or dataset
        lv1detect : str or xarray.Dataset, optional
            Level 1 detection data file path or dataset
        lv1match : str or xarray.Dataset, optional
            Level 1 matching data file path or dataset
        imagesL1detect : str, optional
            Path to particle image archive file
        safeMode : bool, optional
            Enable safe mode for frame seeking, default is False
        config : object, optional
            Configuration object
        skipNonMatched : bool, optional
            Skip non-matched particles, default is False

        Raises
        ------
        ValueError
            If lv1match is not provided as a Dataset
        """
        if type(metaFrames) is xr.Dataset:
            self.metaFrames = metaFrames
        else:
            self.metaFrames = xr.open_dataset(metaFrames)
        if (lv1detect is None) or (type(lv1detect) is xr.Dataset):
            self.lv1detect = lv1detect
        else:
            self.lv1detect = xr.open_dataset(lv1detect)
        if (lv1match is None) or (type(lv1match) is xr.Dataset):
            self.lv1match = lv1match
        else:
            raise ValueError(
                "provide level1match as Dataset with data selected for corresponding camera"
            )

        if imagesL1detect is not None:
            # self.tarFile = tools.imageTarFile.open(imagesL1detect, "r:bz2")
            try:
                self.tarFile = tools.imageZipFile(imagesL1detect, mode="r")
            except FileNotFoundError:
                self.tarFile = None
                log.warning(f"did not find {imagesL1detect}")
            except zipfile.BadZipFile:
                self.tarFile = None
                log.warning(f"BROKEN {imagesL1detect}")

            # self.tarRoot = imagesL1detect.split("/")[-1].replace(".tar.bz2","")
        else:
            self.tarFile = None
            # self.tarRoot = None

        self.safeMode = safeMode
        self.config = config
        self.skipNonMatched = skipNonMatched

        self.movFilePattern = movFilePattern
        self.threads = np.unique(self.metaFrames.nThread)
        if len(self.threads) > 1:
            self.movFilePattern = self.movFilePattern.replace("_0.", "_{thread}.")
        if config:
            assert self.movFilePattern.endswith(config.movieExtension)
        self.video = {}
        self.position = 0
        self.positions = {}
        self.currentThread = None

        self.currentFrame = None
        self.currentMetaFrames = None
        self.currentlv1detect = None
        self.currentPids = None

        self._openVideo()

    def _openVideo(self):
        """
        Open video files for all threads.

        This internal method initializes video readers for each thread
        and sets up the video position tracking.

        Raises
        ------
        AssertionError
            If video file does not exist or no videos were opened
        """
        VideoReader = create_VideoReader()  # constructor for lazy loading
        for tt in self.threads:
            fname = self.movFilePattern.format(thread=tt)
            assert os.path.isfile(fname)
            self.video[tt] = VideoReader(fname)
            self.positions[tt] = 0
        assert len(self.video) > 0

    def resetVideo(self):
        """
        Reset video readers by releasing and reopening them.

        This method releases all current video readers and reopens them,
        useful when video files need to be reloaded or reset.
        """
        for tt in self.threads:
            self.video[tt].release()
        self._openVideo()

    def getNextFrame(self, markParticles=False):
        """
        Get the next frame from the video sequence.

        Parameters
        ----------
        markParticles : bool, optional
            Whether to mark particles on the frame, default is False

        Returns
        -------
        tuple
            Frame data depending on markParticles parameter
        """
        ii = self.position + 1
        if self.lv1detect:
            return self.getFrameByIndexWithParticles(ii, markParticles=markParticles)
        else:
            return self.getFrameByIndex(ii)

    def getPrevFrame(self, markParticles=False):
        """
        Get the previous frame from the video sequence.

        Parameters
        ----------
        markParticles : bool, optional
            Whether to mark particles on the frame, default is False

        Returns
        -------
        tuple
            Frame data depending on markParticles parameter
        """
        ii = self.position - 1
        if self.lv1detect:
            return self.getFrameByIndexWithParticles(ii, markParticles=markParticles)
        else:
            return self.getFrameByIndex(ii)

    def getFrameByIndex(self, ii, increaseContrast=False):
        """
        Retrieve frame by index with associated metadata.

        Parameters
        ----------
        ii : int
            Frame index to retrieve
        increaseContrast : bool, optional
            Whether to increase image contrast, default is False

        Returns
        -------
        tuple
            A tuple containing (success_flag, frame_data, metadata) where
            success_flag indicates if the operation was successful, frame_data
            is the retrieved frame or None, and metadata is the associated
            metadata or None.
        """
        """
        like read, but with meta data and appropriate thread
        """
        if ii < 0:
            return False, None, None

        try:
            captureTime = self.metaFrames.capture_time[ii].values
        except IndexError:
            return False, None, None

        return self.getFrameByCaptureTime(
            captureTime, increaseContrast=increaseContrast
        )

    @functools.lru_cache(maxsize=100, typed=False)
    def getFrameByCaptureTime(self, captureTime, increaseContrast=False):
        """
        Retrieve frame by capture time with associated metadata.

        Parameters
        ----------
        captureTime : datetime64
            Capture time to retrieve frame for
        increaseContrast : bool, optional
            Whether to increase image contrast, default is False

        Returns
        -------
        tuple
            A tuple containing (success_flag, frame_data, metadata) where
            success_flag indicates if the operation was successful, frame_data
            is the retrieved frame or None, and metadata is the associated
            metadata or None.
        """
        """
        like read, but with capturetime, meta data and appropriate thread
        """
        try:
            self.currentMetaFrames = self.metaFrames.sel(capture_time=captureTime)
        except IndexError:
            self.res = None
            self.curentFrame = None
            self.currentMetaFrames = None
            self.position = None
        else:
            self.currentThread = int(self.currentMetaFrames.nThread.values)
            rr = int(self.currentMetaFrames.record_id.values)
            self.res, self.curentFrame = self.video[self.currentThread].getFrameByIndex(
                rr, safeMode=self.safeMode
            )
            if increaseContrast:
                self.curentFrame = doubleDynamicRange(self.curentFrame)
            self.position = np.where(self.metaFrames.capture_time == captureTime)[0][0]
            self.positions[self.currentThread] = self.video[self.currentThread].position

        return self.res, self.curentFrame, self.currentMetaFrames

    @functools.lru_cache(maxsize=100, typed=False)
    def getFrameByCaptureTimeWithParticles(
        self,
        captureTime,
        pad=4,
        markParticles=False,
        highlightPid=None,
        increaseContrast=False,
        showTracks=False,
    ):
        """
        Retrieve frame by capture time with particle annotations.

        Parameters
        ----------
        captureTime : datetime64
            Capture time to retrieve frame for
        pad : int, optional
            Padding around particle bounding boxes, default is 4
        markParticles : bool, optional
            Whether to mark particles on the frame, default is False
        highlightPid : int or str, optional
            Particle ID to highlight, default is None
        increaseContrast : bool, optional
            Whether to increase image contrast, default is False
        showTracks : bool, optional
            Whether to show particle tracks, default is False

        Returns
        -------
        tuple
            A tuple containing (success_flag, annotated_frame, metadata,
            detection_data, matched_data, particles) where:
            - success_flag indicates if the operation was successful
            - annotated_frame is the frame with annotations or None
            - metadata is the associated metadata or None
            - detection_data is the detection data or None
            - matched_data is the matched data or None
            - particles is a dictionary of particle images or None
        """
        import cv2

        assert self.lv1detect is not None

        res, _, _ = self.getFrameByCaptureTime(
            captureTime, increaseContrast=increaseContrast
        )
        if (res is None) or (res == False):
            return None, None, None, None, None, None

        self.curentFrameC = cv2.cvtColor(self.curentFrame, cv2.COLOR_GRAY2BGR)

        if self.config.level1detect.cropImage is not None:
            color = (255, 255, 255)
            cv2.rectangle(
                self.curentFrameC,
                (
                    self.config.level1detect.cropImage[0],
                    self.config.level1detect.cropImage[1]
                    + self.config.level1detect.height_offset,
                ),
                (
                    self.config.level1detect.cropImage[0]
                    + (
                        self.config.frame_width
                        - 2 * self.config.level1detect.cropImage[0]
                    ),
                    self.config.level1detect.cropImage[1]
                    + self.config.level1detect.height_offset
                    + (
                        self.config.frame_height
                        - 2 * self.config.level1detect.cropImage[1]
                    ),
                ),
                color,
                2,
            )

        ct = self.currentCaptureTime
        try:
            self.currentlv1detect = self.lv1detect.isel(
                pid=(self.lv1detect.capture_time == ct)
            )
        except:
            self.currentlv1detect = self.lv1detect.isel(
                fpid=(self.lv1detect.capture_time == ct)
            )

        self.currentPids = self.currentlv1detect.pid

        assert np.all(
            self.currentMetaFrames.capture_time == self.currentlv1detect.capture_time
        )

        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
        ] * 30

        matchedDats = []
        particles = {}

        if markParticles:
            for jj, pid in enumerate(self.currentlv1detect.pid.values):
                try:
                    partic1 = self.currentlv1detect.sel(pid=pid)
                except KeyError:
                    partic1 = self.currentlv1detect.sel(
                        fpid=(self.currentlv1detect.pid == pid)
                    ).squeeze()

                (x, y) = partic1.position_upperLeft.values.astype(int)
                (w, h) = partic1.Droi.values.astype(int)
                y = y + self.config.level1detect.height_offset

                if self.config.level1detect.cropImage is not None:
                    y = y + self.config.level1detect.cropImage[1]
                    x = x + self.config.level1detect.cropImage[0]
                if (
                    (self.lv1match is not None)
                    and (highlightPid == "meta")
                    and pid in self.lv1match.pid
                ):
                    thisMatch = self.lv1match.isel(fpair_id=self.lv1match.pid == pid)
                    matchedDats.append(thisMatch)
                    np.random.seed(int(thisMatch.pair_id))
                    color = (
                        np.random.randint(0, 170),
                        np.random.randint(0, 170),
                        np.random.randint(0, 170),
                    )
                    extraInfo = "%.2g" % thisMatch.matchScore.values

                    if showTracks:
                        track_id = thisMatch.track_id.values[0]
                        trackDat = self.lv1match.position_centroid.isel(
                            fpair_id=(self.lv1match.track_id == track_id)
                        )
                        trackDat[:, 1] += self.config.level1detect.height_offset
                        np.random.seed(int(100000 + track_id))
                        colorT = (
                            np.random.randint(0, 170),
                            np.random.randint(0, 170),
                            np.random.randint(0, 170),
                        )
                        cv2.polylines(
                            self.curentFrameC,
                            np.int32([trackDat.values]),
                            isClosed=False,
                            color=colorT,
                            thickness=2,
                        )
                        cv2.putText(
                            self.curentFrameC,
                            "T%i" % (track_id),
                            trackDat.values[0] + np.array([10, 10]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            colorT,
                            2,
                        )

                elif highlightPid == pid:
                    color = (255, 255, 255)
                    extraInfo = ""
                else:
                    color = (0, 0, 0)
                    extraInfo = ""
                    if self.skipNonMatched:
                        continue

                x1, y1, x2, y2 = x - pad, y - pad, x + w + pad, y + h + pad

                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 >= self.curentFrameC.shape[1]:
                    x2 = self.curentFrameC.shape[1] - 1
                if y2 >= self.curentFrameC.shape[0]:
                    y2 = self.curentFrameC.shape[0] - 1

                cv2.rectangle(self.curentFrameC, (x1, y1), (x2, y2), color, 2)

                # cnt = partic1.cnt.values[partic1.cnt.values[...,0]>=0]
                # cv2.drawContours(self.curentFrameC, np.array([cnt], dtype=np.int32),0,color,1)

                extra1 = str(partic1.capture_time.values)[:-6].split("T")[-1]

                posY = int(
                    partic1.position_upperLeft[1]
                    + self.config.level1detect.height_offset
                    - 10
                )
                posX = int(partic1.position_upperLeft[0])
                if self.config.level1detect.cropImage is not None:
                    posY = posY + self.config.level1detect.cropImage[1]
                    posX = posX + self.config.level1detect.cropImage[0]

                cv2.putText(
                    self.curentFrameC,
                    "%i %s %s" % (partic1.pid, extra1, extraInfo),
                    (posX, posY),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                )

                pidStr = "%07i" % pid
                imName = "%s.png" % (pidStr)

                if self.tarFile is not None:
                    try:
                        # imfname = '%s/%s/%s' % (self.tarRoot, pidStr[:4], imName)
                        # particles[pid] = self.tarFile.extractimage(imfname)
                        particles[pid] = self.tarFile.extractnpy(pidStr)
                    except KeyError:
                        # print(f"{pid} not found")
                        continue

                    # np.random.seed(int(thisMatch.fpair_id))
                    # color = (np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))

            if len(matchedDats) > 0:
                matchedDats = xr.concat(matchedDats, dim="fpair_id")
            else:
                matchedDats = None

        return (
            self.res,
            self.curentFrameC,
            self.currentMetaFrames,
            self.currentlv1detect,
            matchedDats,
            particles,
        )

    def getFrameByIndexWithParticles(
        self, ii, markParticles=False, highlightPid=None, increaseContrast=False
    ):
        """
        Retrieve frame by index with particle annotations.

        Parameters
        ----------
        ii : int
            Frame index to retrieve
        markParticles : bool, optional
            Whether to mark particles on the frame, default is False
        highlightPid : int or str, optional
            Particle ID to highlight, default is None
        increaseContrast : bool, optional
            Whether to increase image contrast, default is False

        Returns
        -------
        tuple
            A tuple containing (success_flag, annotated_frame, metadata,
            detection_data, matched_data, particles) where:
            - success_flag indicates if the operation was successful
            - annotated_frame is the frame with annotations or None
            - metadata is the associated metadata or None
            - detection_data is the detection data or None
            - matched_data is the matched data or None
            - particles is a dictionary of particle images or None
        """
        """
        like read, but with even more meta data and appropriate thread
        """

        if ii < 0:
            return False, None, None, None, None, None

        try:
            captureTime = self.metaFrames.capture_time[ii].values
        except IndexError:
            return False, None, None, None, None, None

        return self.getFrameByCaptureTimeWithParticles(
            captureTime,
            markParticles=markParticles,
            highlightPid=highlightPid,
            increaseContrast=increaseContrast,
        )

    @property
    def currentCaptureTime(self):
        """
        Get the current capture time.

        Returns
        -------
        datetime64 or None
            Current capture time or None if no metadata is available.
        """
        if self.currentMetaFrames is None:
            return None
        else:
            return np.datetime64(self.currentMetaFrames.capture_time.values)

    @property
    def total_frames(self):
        """
        Get the total number of frames across all videos.

        Returns
        -------
        int
            Total number of frames in all video streams.
        """
        nFrames = 0
        for tt in self.threads:
            nFrames += self.video[tt].total_frames
        return nFrames

    def release(self):
        """
        Release all video resources.

        This method releases all video readers and closes archive files
        to free up system resources.
        """
        for tt in self.threads:
            self.video[tt].release()
        if self.tarFile is not None:
            self.tarFile.close()

    def getParticle(self, pid, heightOffset=64):
        """
        Retrieve a specific particle image.

        Parameters
        ----------
        pid : int
            Particle ID to retrieve
        heightOffset : int, optional
            Height offset for particle cropping, default is 64

        Returns
        -------
        tuple
            A tuple containing (cropped_particle_image, full_frame_image)
        """
        particle = self.lv1detect.sel(pid=pid)
        kk = int(particle.record_id.values)
        _, frame1, _ = self.getFrameByIndex(kk)
        (x, y) = particle.position_upperLeft.values.astype(int)
        (w, h) = particle.Droi.values.astype(int)
        if len(frame1.shape) == 3:
            frame1 = frame1[:, :, 0]
        return frame1[y + heightOffset : y + heightOffset + h, x : x + w], frame1


def doubleDynamicRange(frame, offset="estimate", factor=2):
    """
    Double the dynamic range of an image to improve feature detection.

    This function increases the dynamic range of an image by multiplying
    pixel values by a factor and adjusting for brightness. The offset ensures
    that the brightest pixels don't overflow while maintaining contrast.

    Parameters
    ----------
    frame : array
        Input image frame
    offset : str or int, optional
        Offset calculation method ('estimate' or numeric value), default is 'estimate'
    factor : int, optional
        Multiplication factor for increasing dynamic range, default is 2

    Returns
    -------
    array
        Image with doubled dynamic range

    Notes
    -----
    The factor of 2 ensures all gradients scale with the same factor even for integers.
    The offset is chosen so that the brightest point is max. 255 after multiplication.
    If this means the darkest point becomes negative, the offset is adjusted accordingly.
    """
    import cv2

    if offset == "estimate":
        # offset so that brightest spot is 255 even if doubled
        offset1 = frame.max() - (254 // factor)
        # offset so that darkest point is zero.
        offset2 = frame.min()
        # take smaller one so that in doubt information is lost for brighter pixels
        offset = min(offset1, offset2)
        # make sure offset is >= 0
        offset = max(0, offset)
        # apply to frame. cv2.multiply handles overflows properly

    frame = cv2.subtract(frame, int(offset))
    frame = cv2.multiply(frame, factor)

    return frame


def main():
    """
    Main function for command-line usage.

    This function provides command-line interface for creating double-image
    visualizations of particles from two different videos.

    Usage:
        python -m VISSSlib.av doubleImage fname1 index1 pid1 fname2 index2 pid2 confFile version  outfile

    Raises
    ------
    SystemExit
        If arguments are incorrect or processing fails
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    """
    python -m VISSSlib.av doubleImage fname1 index1 pid1 fname2 index2 pid2 confFile version  outfile 
    """

    assert sys.argv[1] == "doubleImage"

    fname1 = sys.argv[2]
    video1ii = int(sys.argv[3])
    pid1 = int(sys.argv[4])
    fname2 = sys.argv[5]
    video2ii = int(sys.argv[6])
    pid2 = int(sys.argv[7])
    confFile = sys.argv[8]
    version = sys.argv[9]
    outFile = sys.argv[10]

    try:
        config = tools.readSettings(confFile)

        f1 = files.Filenames(fname1, config, version)
        f2 = files.Filenames(fname2, config, version)

        frames = []
        video1 = VideoReaderMeta(
            f1.fnameLevel0, f1.fnameLevel1, f1.fnameLevel2, config=config
        )

        res, frame1, meta11, meta12 = video1.getFrameByIndexWithParticles(
            video1ii, markParticles=True, highlightPid=pid1
        )  # frame number

        frames.append(frame1)
        frames.append(np.zeros((frame1.shape[0], 10, 3), dtype=int))

        video2 = VideoReaderMeta(
            f2.fnameLevel0, f2.fnameLevel1, f2.fnameLevel2, config=config
        )

        res, frame2, meta21, meta22 = video2.getFrameByIndexWithParticles(
            video2ii, markParticles=True, highlightPid=pid2
        )  # frame number
        frames.append(frame2)

        frame = np.concatenate(frames, axis=1)

        plt.figure(figsize=(20, 10))
        plt.imshow(frame, cmap="gray", vmin=0, vmax=255)
        plt.title(f"{pid1} {pid2}")

        plt.savefig(outFile)

        print(outFile)
    except:
        print(outFile, "FAILED")
    return 0


def cvtColor(frame):
    """
    Convert BGR to grayscale using direct indexing.

    This is a faster alternative to cv2.cvtColor for grayscale conversion
    when working with single-channel images.

    Parameters
    ----------
    frame : array
        Input BGR frame

    Returns
    -------
    array
        Grayscale frame
    """
    # faster than cv2.cvtColor but works only for gray images
    return frame[:, :, 0]


def cvtGray(frame):
    """
    Convert grayscale to BGR format.

    Parameters
    ----------
    frame : array
        Input grayscale frame

    Returns
    -------
    array
        BGR formatted frame
    """
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    main()
