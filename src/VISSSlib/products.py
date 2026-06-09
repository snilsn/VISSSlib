import glob
import os
import random
import string
import sys
from functools import cached_property, partial

import numpy as np
import xarray as xr
from loguru import logger as log

from . import __version__, files, matching, metadata, quicklooks, tools
from .tools import ipython_debug, runCommandInQueue


class DataProduct(object):
    @log.catch(reraise=True)
    def __init__(
        self,
        level,
        case,
        settings,
        fileQueue,
        camera,
        relatives=None,
        addRelatives=True,
        childrensRelatives=None,
    ):
        """
        Initialize a DataProduct for processing VISSS data.

        Parameters
        ----------
        level : str
            Processing level (e.g., 'level0', 'level1detect', 'metaEvents')
        case : str
            Case identifier for the data
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management. If None, a temporary queue will be created.
        camera : str
            Camera identifier ('leader' or 'follower')
        relatives : str, optional
            Relative path specification
        addRelatives : bool, default True
            Whether to add relatives of the corresponding product
        childrensRelatives : dict, default {}
            Dictionary of child relatives

        Raises
        ------
        ValueError
            If camera is not 'leader' or 'follower'
        """
        import taskqueue

        """
        Class for processing VISSS data

        """
        log.debug(f"created  {level} {camera} for {case} with {childrensRelatives}.")
        self.level = level
        self.config = tools.readSettings(settings)
        if relatives is not None:
            self.relatives = f"{relatives}.{level}"
        else:
            self.relatives = level
        if childrensRelatives is None:
            self.childrensRelatives = tools.DictNoDefault({})
        else:
            self.childrensRelatives = childrensRelatives
        if camera == "leader":
            self.cameraFull = self.config.leader
        elif camera == "follower":
            self.cameraFull = self.config.follower
        else:
            raise ValueError(f"do not understand camera: {camera}")
        self.camera = camera
        self.case = case

        if fileQueue is None:
            fileQueue = f"/tmp/visss_{''.join(random.choice(string.ascii_uppercase) for _ in range(10))}"

        if type(fileQueue) is str:
            self.fileQueue = fileQueue
            self.tq = taskqueue.TaskQueue(f"fq://{self.fileQueue}")
        else:
            self.tq = fileQueue
            self.fileQueue = self.tq.path.path

        self.commands = []

        self.fn = files.FindFiles(str(self.case), self.cameraFull, self.config)
        self.path = self.fn.fnamesPatternExt[self.level]

        self.parents = tools.DictNoDefault({})

        if self.level == "level0":
            self.parentNames = []
        elif self.level == "level0txt":
            self.parentNames = []
        elif level == "metaEvents":
            self.parentNames = [f"{camera}_level0txt"]
        elif level == "metaFrames":
            self.parentNames = [f"{camera}_level0txt"]
        elif level == "level1detect":
            self.parentNames = [
                # f"{camera}_metaFrames", # done by level1detect
                # f"{camera}_metaEvents", # done by metaRotation
            ]
        elif level == "metaRotation":
            assert camera == "leader"
            self.parentNames = [
                f"leader_level1detect",
                f"follower_level1detect",
                f"leader_metaEvents",  # metaEvents are added to all the L2 products to force regenration when event file is updated (ie more data is transferred)
                f"follower_metaEvents",
            ]
        elif level == "level1match":
            assert camera == "leader"
            self.parentNames = [f"{camera}_metaRotation"]
        elif level == "level1track":
            assert camera == "leader"
            self.parentNames = [f"{camera}_level1match"]
        # elif level == "level1shape":
        #     assert camera == "leader"
        #     self.parentNames = [f"{camera}_level1track"]
        elif level == "level2detect":
            self.parentNames = [f"{camera}_level1detect", f"{camera}_metaEvents"]
        elif level == "level2match":
            assert camera == "leader"
            self.parentNames = [
                f"{camera}_level1match",
                f"leader_metaEvents",  # metaEvents are aded to all the L2 products to force regenration when events file is updated (ie more data is transferred)
                f"follower_metaEvents",
            ]
        elif level == "level2track":
            assert camera == "leader"
            self.parentNames = [
                f"{camera}_level1track",
                f"leader_metaEvents",
                f"follower_metaEvents",
            ]
        elif level == "level3combinedRiming":
            assert camera == "leader"
            self.parentNames = [
                f"{camera}_level2track",
                f"leader_metaEvents",
                f"follower_metaEvents",
            ]
        elif level == "allDone":
            assert camera == "leader"
            self.parentNames = [
                f"leader_metaEvents",
                f"follower_metaEvents",
            ]
            if self.config.level1match.processL1match:
                self.parentNames += [
                    "leader_level2track",
                    "leader_level2match",
                ]
            if self.config.level2.processL2detect:
                self.parentNames += [
                    "leader_level2detect",
                    "follower_level2detect",
                ]
            if self.config.level3.combinedRiming.processRetrieval:
                self.parentNames += [
                    "leader_level3combinedRiming",
                ]

        else:
            raise ValueError(f"Do not understand {level}")
        if addRelatives:
            for parentCam in self.parentNames:
                # save time by not adding a product more than once
                if parentCam in self.childrensRelatives.keys():
                    # print(f"{self.relatives}, found {parentCam} from other relative")
                    self.parents[parentCam] = self.childrensRelatives[parentCam]
                    assert self.case == self.childrensRelatives[parentCam].case
                    continue
                camera, parent = parentCam.split("_")
                self.parents[parentCam] = DataProduct(
                    parent,
                    self.case,
                    self.config,
                    self.tq,
                    camera,
                    relatives=f"{self.relatives}",
                    childrensRelatives=self.parents,
                )
                self.parents.update(self.parents[parentCam].parents)
                self.childrensRelatives.update(self.parents)

    def __repr__(self):
        """
        Return string representation of the DataProduct object.

        Returns
        -------
        str
            String representation of the object
        """
        reprstr = (
            f"<VISSSlib.products.DataProduct object {self.level} "
            f"using {self.camera} on {self.case}>"
        )
        return reprstr

    @log.catch(reraise=True)
    def generateAllCommands(self, skipExisting=True, withParents=True):
        """
        Generate all commands for processing this product and its dependencies.

        Parameters
        ----------
        skipExisting : bool, default True
            Whether to skip existing files
        withParents : bool, default True
            Whether to include parent commands

        Returns
        -------
        list
            List of commands to execute
        """
        # cache for this function
        isComplete = self.isComplete

        if (not self.dataAvailable) and (self.config.end == "today"):
            log.warning(
                f"{self.case} {self.relatives}: no data found (yet?) in {self.fn.fnamesPattern.level0txt}"
            )
            return []

        if (
            skipExisting
            and isComplete
            and self._youngerThanParents
            and self.parentsComplete
        ):
            if withParents:
                log.info(f"{self.case} {self.relatives}: everything processed")
            return []
        if isComplete and (not self._youngerThanParents):
            for name, younger in self._youngerThanParentsDict.items():
                if not younger:
                    log.warning(
                        f"{self.case} {self.relatives} redoing level, parent {name} is younger"
                    )
        if self.parentsComplete and self._parentsYoungerThanGrandparents:
            commands = self.generateCommands(
                skipExisting=skipExisting,
            )
            if len(commands) > 0:
                log.info(
                    f"{self.case} {self.relatives} generated commands for level {self.level} {self.camera}"
                )
        elif not self.parentsComplete:
            log.warning(
                f"{self.case} {self.relatives} no commands generated yet, parents not complete yet"
            )
            commands = []
        else:
            log.warning(
                f"{self.case} {self.relatives} no commands generated, grandparents older"
            )
            commands = []
        if withParents:
            for parent in self.parents.keys():
                # parents always with skipExisting = True to avoid chain reaction
                commands = commands + self.parents[parent].generateAllCommands(
                    skipExisting=True,
                    withParents=False,
                )
        self.commands = list(set(commands))
        if (len(self.commands) == 0) and (withParents):
            log.warning(
                f"{self.level} {self.camera} {self.case} no commands created",
            )
        return self.commands

    @log.catch(reraise=True)
    def generateCommands(self, skipExisting=True, nCPU=1, bin=None):
        """
        Generate commands for processing this product.

        Parameters
        ----------
        skipExisting : bool, default True
            Whether to skip existing files
        nCPU : int, default 1
            Number of CPU cores to use
        bin : str, optional
            Python binary path

        Returns
        -------
        list
            List of commands to execute

        Raises
        ------
        ValueError
            If the level is not recognized
        """
        if self.level == "level0":
            return []
        elif self.level == "level0txt":
            return []
        elif self.level == "metaEvents":
            return self._commandTemplateDaily(
                "metadata.createEvent", skipExisting=skipExisting, nCPU=nCPU, bin=bin
            )
        elif self.level == "metaFrames":
            return self._commandTemplateDaily(
                "metadata.createMetaFrames",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "level1detect":
            originLevel = "level0txt"
            call = "detection.detectParticles"
            return self._commandTemplateL1(
                originLevel,
                call,
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "metaRotation":
            return self._commandTemplateDaily(
                "matching.createMetaRotation",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "level1match":
            originLevel = "level1detect"
            call = "matching.matchParticles"
            return self._commandTemplateL1(
                originLevel,
                call,
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
                extraOrigin="metaRotation",
            )
        elif self.level == "level1track":
            originLevel = "level1match"
            call = "tracking.trackParticles"
            return self._commandTemplateL1(
                originLevel, call, skipExisting=skipExisting, nCPU=nCPU, bin=bin
            )
        # elif self.level == "level1shape":
        #     originLevel = "level1track"
        #     call = "particleshape.classifyParticles"
        #     return self._commandTemplateL1(
        #         originLevel, call, skipExisting=skipExisting, nCPU=nCPU, bin=bin
        #     )
        elif self.level == "level2detect":
            return self._commandTemplateDaily(
                "distributions.createLevel2detect",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "level2match":
            return self._commandTemplateDaily(
                "distributions.createLevel2match",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "level2track":
            return self._commandTemplateDaily(
                "distributions.createLevel2track",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "level3combinedRiming":
            return self._commandTemplateDaily(
                "level3.retrieveCombinedRiming",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "allDone":
            outFile = self.fn.fnamesDaily["allDone"]
            command = f"mkdir -p {os.path.dirname(outFile)} && touch {outFile}"
            return [(command, outFile)]
        else:
            raise ValueError(f"Do not understand {self.level}")

    def _commandTemplateL1(
        self,
        originLevel,
        call,
        skipExisting=True,
        nCPU=1,
        bin=None,
        extraOrigin=None,
    ):
        """
        Generate commands for L1 processing steps.

        Parameters
        ----------
        originLevel : str
            Origin level for processing
        call : str
            Function call to execute
        skipExisting : bool, default True
            Whether to skip existing files
        nCPU : int, default 1
            Number of CPU cores to use
        bin : str, optional
            Python binary path
        extraOrigin : str, optional
            Extra origin level for comparison

        Returns
        -------
        list
            List of commands to execute
        """
        if skipExisting:
            skipExistingStr = "--skip-existing"
        else:
            skipExistingStr = ""
        if bin is None:
            bin = os.path.join(sys.exec_prefix, "bin", "python")
        commands = []
        for pName in self.fn.listFilesExt(originLevel):
            if originLevel.startswith("level0"):
                f1 = files.Filenames(pName, self.config)
            else:
                f1 = files.FilenamesFromLevel(pName, self.config)
            outFile = f1.fname[self.level]
            exisiting = glob.glob(f"{outFile}*")

            if (len(exisiting) >= 1) and (extraOrigin is not None):
                extraOlder = os.path.getmtime(
                    self.fn.listFilesExt(extraOrigin)[0]
                ) < os.path.getmtime(exisiting[0])
            else:
                extraOlder = True

            if (
                skipExisting
                and (len(exisiting) >= 1)
                and (os.path.getmtime(pName) < os.path.getmtime(exisiting[0]))
                and extraOlder
            ):
                log.debug(f"{self.relatives} skip exisiting {exisiting[0]}")
                continue

            if len(exisiting) > 1:
                for ex in exisiting:
                    os.remove(ex)
                    log.warning(f"too many files, removed {ex}")

            command = f"{bin} -m VISSSlib {call} {self.config.filename} {pName} {skipExistingStr}"
            if nCPU is not None:
                command = f"export OPENBLAS_NUM_THREADS={nCPU}; export MKL_NUM_THREADS={nCPU}; export NUMEXPR_NUM_THREADS={nCPU}; export OMP_NUM_THREADS={nCPU}; {command}"
            commands.append((command, outFile))
        return commands

    def _commandTemplateDaily(self, call, skipExisting=True, nCPU=1, bin=None):
        """
        Generate commands for daily processing steps.

        Parameters
        ----------
        call : str
            Function call to execute
        skipExisting : bool, default True
            Whether to skip existing files
        nCPU : int, default 1
            Number of CPU cores to use
        bin : str, optional
            Python binary path

        Returns
        -------
        list
            List of commands to execute
        """
        nCPU = 1
        if skipExisting:
            skipExistingStr = "--skip-existing"
        else:
            skipExistingStr = ""
        if bin is None:
            bin = os.path.join(sys.exec_prefix, "bin", "python")
        if (
            call.endswith("detect")
            or call.endswith("MetaFrames")
            or call.endswith("createEvent")
            or call.endswith("createLevel1detectQuicklook")
        ):
            case = f"{self.case} --camera={self.camera}"
        else:
            case = self.case

        outFile = self.fn.fnamesDaily[self.level]

        exisiting = glob.glob(f"{outFile}*")
        if skipExisting and (len(exisiting) >= 1) and (self._youngerThanParents):
            log.info(f"{self.relatives} skip exisiting {exisiting[0]}")
            return []

        command = (
            f"{bin} -m VISSSlib {call} {self.config.filename} {case} {skipExistingStr}"
        )
        if nCPU is not None:
            command = f"export OPENBLAS_NUM_THREADS={nCPU}; export MKL_NUM_THREADS={nCPU}; export NUMEXPR_NUM_THREADS={nCPU}; export OMP_NUM_THREADS={nCPU}; {command}"
        return [(command, outFile)]

    @log.catch(reraise=True)
    def process(
        self,
        skipExisting=True,
        checkForDuplicates=False,
        withParents=True,
        runWorkers=False,
    ):
        """
        Process product using the task queue. Runs submitCommands and
        runWorkers. Sometimes, needs to be called multiple times until all parent
        products are processed

        Parameters
        ----------
        skipExisting : bool, default True
            Whether to skip existing files
        checkForDuplicates : bool, default False
            Whether to check for duplicate commands in the queue
        withParents : bool, default True
            Whether to include parent commands
        """

        self.submitCommands(
            skipExisting=skipExisting,
            checkForDuplicates=checkForDuplicates,
            withParents=withParents,
            runWorkers=True,
        )

        self.runWorkers()

    @log.catch(reraise=True)
    def submitCommands(
        self,
        skipExisting=True,
        checkForDuplicates=False,
        withParents=True,
    ):
        """
        Submit commands to the task queue.

        Parameters
        ----------
        skipExisting : bool, default True
            Whether to skip existing files
        checkForDuplicates : bool, default False
            Whether to check for duplicate commands in the queue
        withParents : bool, default True
            Whether to include parent commands
        runWorkers : bool, default False
            Whether to run workers immediately
        """
        if len(self.commands) == 0:
            self.generateAllCommands(
                skipExisting=skipExisting,
                withParents=withParents,
            )

        if len(self.commands) == 0:
            log.error("nothing to submit")
            return

        if checkForDuplicates:
            running = [t.args[0] for t in self.tq.tasks()]
            commands = []
            for command in self.commands:
                if command[0][0] in running:
                    continue
                else:
                    commands.append(command)
        else:
            commands = self.commands

        log.warning(f"sending {len(commands)} commands to {self.fileQueue}")
        # region is SQS specific, green means cooperative threading

        self.tq.insert([partial(runCommandInQueue, c) for c in commands])
        log.warning(f"{self.tq.enqueued} tasks in Queue")

        return

    @log.catch(reraise=True)
    def runWorkers(self, nJobs=os.cpu_count(), waitTime=1):
        """
        Run worker processes.

        Parameters
        ----------
        nJobs : int, default os.cpu_count()
            Number of jobs to run
        """
        tools.workers(self.fileQueue, nJobs=nJobs, waitTime=waitTime)

    def deleteQueue(self):
        """
        Delete all tasks from the queue.
        """
        log.info(f"Deleting {self.tq.enqueued} tasks")
        [self.tq.delete(t) for t in self.tq.tasks()]
        return

    @cached_property
    def isComplete(self):
        """
        Check if all required files for this level exist.

        Returns
        -------
        bool
            True if all files are complete, False otherwise
        """
        nMissing = self.fn.nMissing(self.level)
        if nMissing > 0:
            log.info(f"{self.case} {self.relatives} {nMissing} files are missing")
        return nMissing == 0

    @cached_property
    def _youngerThanParentsDict(self):
        """
        Check if this product is younger than its parents.

        Returns
        -------
        dict
            Dictionary mapping parent names to boolean values indicating
            whether this product is younger than each parent
        """
        youngerThanParentsDict = tools.DictNoDefault()
        for name, parent in self.parents.items():
            isYounger = parent.fileCreation < self.fileCreation
            if (self.level == "level1detect") and (parent.level == "metaEvents"):
                # special case: no need to do level1detect again due to updated metaEvents
                youngerThanParentsDict[name] = True
            else:
                youngerThanParentsDict[name] = isYounger
            if not youngerThanParentsDict[name]:
                log.debug(
                    f"{self.relatives} is older "
                    f"({tools.timestamp2str(self.fileCreation)}) than parent "
                    f"{name} ({tools.timestamp2str(parent.fileCreation)})",
                )
        return youngerThanParentsDict

    @cached_property
    def _youngerThanParents(self):
        """
        Check if this product is younger than all parents.

        Returns
        -------
        bool
            True if this product is younger than all parents, False otherwise
        """
        youngerThanParents = np.all(list(self._youngerThanParentsDict.values()))
        return youngerThanParents

    @cached_property
    def _parentsYoungerThanGrandparents(self):
        """
        Check if parents are younger than their grandparents.

        Returns
        -------
        bool
            True if all parents are younger than their grandparents, False otherwise
        """
        parentsYoungerThanGrandparents = True
        for name, parent in self.parents.items():
            parentsYoungerThanGrandparents = (
                parentsYoungerThanGrandparents and parent._youngerThanParents
            )
            log.debug(
                f"{self.relatives} parent {name} is younger than its (grand)parents { parent._youngerThanParents}"
            )
        return parentsYoungerThanGrandparents

    def _fileCreation(self, files):
        """
        Get the creation time of the most recent file.

        Parameters
        ----------
        files : list
            List of file paths

        Returns
        -------
        float
            Maximum modification time of the files
        """
        if len(files) > 0:
            return np.max([os.path.getmtime(f) for f in files])
        else:
            return 0

    @cached_property
    def fileCreation(self):
        """
        Get the creation time of this product.

        Returns
        -------
        float
            Modification time of the newest file
        """
        files = self.listFilesExt()
        return self._fileCreation(files)

    @cached_property
    def parentsComplete(self):
        """
        Check if product's parent are complete.

        Returns
        -------
        bool
            True if all parents are complete, False otherwise
        """
        parentsComplete = True
        for name, parent in self.parents.items():
            thisParentIsComplete = parent.isComplete
            log.debug(
                f"{self.relatives} {name} parentsComplete {thisParentIsComplete}",
            )
            parentsComplete = parentsComplete and thisParentIsComplete
            if not parentsComplete:  # shortcut
                break
        return parentsComplete

    def report(self, withParents=True):
        """
        Print a report about this product's status.

        Parameters
        ----------
        withParents : bool, default True
            Whether to include parent reports
        """
        nMissing = self.fn.nMissing(self.level)
        print(
            self.camera,
            self.level,
            "nMissing",
            nMissing,
            "newest file",
            tools.timestamp2str(self.fileCreation),
            "younger than parents",
            self._youngerThanParents,
        )
        if nMissing > 0:
            print(
                " " * 5,
                [(p, self.fn.nMissing(p.split("_")[1])) for p in self.parentNames],
            )
        if withParents:
            for name, parent in self.parents.items():
                parent.report(withParents=False)

    @cached_property
    def dataAvailable(self):
        """
        Check if data is available for this product.

        Returns
        -------
        bool
            True if data is available, False otherwise
        """
        return len(self.fn.listFiles("level0txt")) > 0

    @cached_property
    def allComplete(self):
        """
        Check if this product and all its dependencies are complete.

        Returns
        -------
        bool
            True if all is complete, False otherwise
        """
        return self.isComplete and self._youngerThanParents and self.parentsComplete

    @cached_property
    def nFiles(self):
        """
        Get the number of files for this product.

        Returns
        -------
        int
            Number of files
        """
        return len(self.fn.listFilesExt(self.level))

    def listFilesExt(self):
        """
        List all files for this product.

        Returns
        -------
        list
            List of file paths
        """
        return self.fn.listFilesExt(self.level)

    def listFiles(self):
        """
        List files for this product.

        Returns
        -------
        list
            List of file paths
        """
        return self.fn.listFiles(self.level)

    def listBroken(self):
        """
        List broken files for this product.

        Returns
        -------
        list
            List of broken file paths
        """
        return self.fn.listBroken(self.level)

    def listNoData(self):
        """
        List files with no data for this product.

        Returns
        -------
        list
            List of no-data file paths
        """
        return self.fn.listNoData(self.level)

    def cleanUpBroken(self, withParents=False, withNoData=False):
        """
        Clean up broken files.

        Parameters
        ----------
        withParents : bool, default False
            Whether to clean up parents too
        withNoData : bool, default False
            Whether to clean up no-data files too
        """
        for fname in self.listBroken():
            assert fname.endswith("broken.txt")
            try:
                os.remove(fname)
            except FileNotFoundError:  # usally caused by caching listBroken
                log.warning(f"{fname} not found")
            else:
                log.warning(f"{fname} removed")
        if withNoData:
            for fname in self.listNoData():
                assert fname.endswith("nodata")
                try:
                    os.remove(fname)
                except FileNotFoundError:  # usally caused by caching listBroken
                    log.warning(f"{fname} not found")
                else:
                    log.warning(f"{fname} removed")
        if withParents:
            for name, parent in self.parents.items():
                if not isinstance(parent, list):
                    parent = [parent]
                [
                    p.cleanUpBroken(withParents=False, withNoData=withNoData)
                    for p in parent
                ]

    def cleanUpDuplicates(self, withParents=False):
        """
        Clean up duplicate files.

        Parameters
        ----------
        withParents : bool, default False
            Whether to clean up parents too
        """
        try:
            dups = self.fn.reportDuplicates(self.level)
        except AttributeError:
            dups = list(
                np.array([f.reportDuplicates(self.level) for f in self.fn]).ravel()
            )

        for fname in dups:
            os.remove(fname)
            log.warning(f"{fname} removed")
        if withParents:
            for name, parent in self.parents.items():
                if not isinstance(parent, list):
                    parent = [parent]
                [p.cleanUpDuplicates(withParents=False) for p in parent]


class allDone(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize an allDone product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management. If None, a temporary queue will be created.
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("allDone", case, settings, fileQueue, camera)


class level2track(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a level2track product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management. If None, a temporary queue will be created.
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("level2track", case, settings, fileQueue, camera)


class level2match(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a level2match product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management. If None, a temporary queue will be created.
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("level2match", case, settings, fileQueue, camera)


class level2detect(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a level2detect product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management. If None, a temporary queue will be created.
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("level2detect", case, settings, fileQueue, camera)


class level1track(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a level1track product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management. If None, a temporary queue will be created.
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("level1track", case, settings, fileQueue, camera)


class level1match(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a level1match product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management. If None, a temporary queue will be created.
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("level1match", case, settings, fileQueue, camera)


class metaRotation(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a metaRotation product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management. If None, a temporary queue will be created.
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("metaRotation", case, settings, fileQueue, camera)


class level1detect(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a level1detect product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management. If None, a temporary queue will be created.
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("level1detect", case, settings, fileQueue, camera)


# class level1shape(DataProduct):
#     def __init__(self, case, settings, fileQueue, camera="leader"):
#         super().__init__("level1shape", case, settings, fileQueue, camera)


class metaFrames(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a metaFrames product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management. If None, a temporary queue will be created.
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("metaFrames", case, settings, fileQueue, camera)


class metaEvents(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a metaEvents product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management. If None, a temporary queue will be created.
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("metaEvents", case, settings, fileQueue, camera)


class level0(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a level0 product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management. If None, a temporary queue will be created.
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("level0", case, settings, fileQueue, camera)


class DataProductRange(DataProduct):
    """Range of data products for multiple cases."""

    def __init__(
        self,
        level,
        cases,
        settings,
        fileQueue,
        camera,
        relatives=None,
        addRelatives=True,
        childrensRelatives=None,
    ):
        """Initialize DataProductRange instance.

        Parameters
        ----------
        level : str
            Processing level
        cases : str or list
            Case identifiers
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management
        camera : str
            Camera identifier
        relatives : str, optional
            Relative path information
        addRelatives : bool, default True
            Whether to add relatives
        childrensRelatives : dict, optional
            Children relatives dictionary
        """
        import taskqueue

        self.cases = tools.getCaseRange(cases, settings)
        self.config = tools.readSettings(settings)

        if fileQueue is None:
            fileQueue = f"/tmp/visss_{''.join(random.choice(string.ascii_uppercase) for _ in range(10))}"
        if type(fileQueue) is str:
            self.fileQueue = fileQueue
            self.tq = taskqueue.TaskQueue(f"fq://{self.fileQueue}")
        else:
            self.tq = fileQueue
            self.fileQueue = self.tq.path.path

        self._instances = [
            DataProduct(
                level,
                case,
                self.config,
                self.tq,
                camera,
                relatives=relatives,
                addRelatives=addRelatives,
            )
            for case in self.cases
        ]
        self.level = level
        self.camera = camera
        self.casesStr = str(cases)

    def __getitem__(self, key):
        """Get item by key.

        Parameters
        ----------
        key : str or int
            Key to retrieve

        Returns
        -------
        DataProduct
            Data product instance
        """
        if isinstance(key, str):
            try:
                return self._instances[self.cases.index(key)]
            except ValueError:
                raise KeyError(f"Case '{key}' not found. Available: {self.cases}")
        return self._instances[key]

    def __iter__(self):
        """Iterate over instances.

        Yields
        ------
        DataProduct
            Data product instances
        """
        return iter(self._instances)

    def __len__(self):
        """Get length of instances.

        Returns
        -------
        int
            Number of instances
        """
        return len(self._instances)

    def __dir__(self):
        """Get directory of attributes.

        Returns
        -------
        list
            List of attribute names
        """
        own = set(super().__dir__())
        instance_attrs = set(dir(self._instances[0])) if self._instances else set()
        return sorted(own | instance_attrs)

    def __getattr__(self, name):
        """Get attribute value.

        Parameters
        ----------
        name : str
            Attribute name

        Returns
        -------
        object
            Attribute value
        """
        # Guard against calls during __init__ before _instances is set
        if name == "_instances" or "_instances" not in self.__dict__:
            raise AttributeError(name)
        if not self._instances:
            raise AttributeError(name)
        attr = getattr(self._instances[0], name)
        if callable(attr):

            def multi_method(*args, **kwargs):
                results = [getattr(dp, name)(*args, **kwargs) for dp in self._instances]
                return tools._aggregate(results)

            return multi_method
        elif name == "config":  # the config is the same for all cases
            return getattr(self._instances[0], name)
        else:
            results = [getattr(dp, name) for dp in self._instances]
            return tools._aggregate(results)

    # overwrite some functions
    def listBroken(self):
        """List broken files for all instances.

        Returns
        -------
        list
            List of broken file paths
        """
        return tools._aggregate([dp.listBroken() for dp in self._instances])

    def listFiles(self):
        """List files for all instances.

        Returns
        -------
        list
            List of file paths
        """
        return tools._aggregate([dp.listFiles() for dp in self._instances])

    def listFilesExt(self):
        """List files with extension for all instances.

        Returns
        -------
        list
            List of file paths
        """
        return tools._aggregate([dp.listFilesExt() for dp in self._instances])

    def listNoData(self):
        """List no-data files for all instances.

        Returns
        -------
        list
            List of no-data file paths
        """
        return tools._aggregate([dp.listNoData() for dp in self._instances])

    def submitCommands(
        self,
        skipExisting=True,
        checkForDuplicates=False,
        withParents=True,
        runWorkers=False,
    ):
        """Submit commands for all instances.

        Parameters
        ----------
        skipExisting : bool, default True
            Whether to skip existing files
        checkForDuplicates : bool, default False
            Whether to check for duplicate commands
        withParents : bool, default True
            Whether to include parent commands
        runWorkers : bool, default False
            Whether to run workers immediately
        """
        commands = tools._aggregate(
            [
                dp.generateAllCommands(
                    skipExisting=skipExisting,
                    withParents=withParents,
                )
                for dp in self._instances
            ]
        )

        if not commands:
            log.error("nothing to submit")
            return
        if checkForDuplicates:
            running = [t.args[0] for t in self.tq.tasks()]
            commands = [c for c in commands if c[0] not in running]
        log.warning(f"sending {len(commands)} commands to {self.fileQueue}")
        self.tq.insert([partial(runCommandInQueue, c) for c in commands])
        log.warning(f"{self.tq.enqueued} tasks in Queue")
        if runWorkers:
            self.runWorkers()


@log.catch(reraise=True)
def submitAll(
    case,
    settings,
    fileQueue,
    doMetaRot=True,
    submitJobs=True,
    skipExisting=True,
    checkForDuplicates=True,
    cleanUpBroken=False,
    cleanUpDuplicates=True,
):
    """
    Submit all processing jobs of for a given range of days. All processing
    levels are considered if corresponding input files are available

    Parameters
    ----------
    case : str
        Case or case range identifier for the data to process
    settings : str
        Path to settings file
    fileQueue : str
        File queue for task management. If None, a temporary queue will be created.
    doMetaRot : bool, default True
        Whether to perform meta rotation
    submitJobs : bool, default True
        Whether to submit jobs to the queue
    skipExisting : bool, default True
        Whether to skip existing files
    checkForDuplicates : bool, default True
        Whether to check for duplicate commands in the queue
    cleanUpBroken : bool, default False
        Whether to clean up broken files
    cleanUpDuplicates : bool, default False
        Whether to clean up duplicate files

    Returns
    -------
    object
        DataProductRange object
    """
    if submitJobs:
        import taskqueue

        tq = taskqueue.TaskQueue(f"fq://{fileQueue}")
        log.warning(f"{tq.enqueued} tasks in Queue")

        prod = DataProductRange("allDone", case, settings, fileQueue, "leader")
        if cleanUpBroken:
            prod.cleanUpBroken(withParents=True, withNoData=False)
        if cleanUpDuplicates:
            prod.cleanUpDuplicates(withParents=True)
        prod.submitCommands(
            checkForDuplicates=checkForDuplicates,
            skipExisting=skipExisting,
        )
    else:
        prod = None

    if doMetaRot:
        log.warning(
            f"{sys.executable} -m VISSSlib matching.createMetaRotation  {settings} {case}"
        )
        matching.createMetaRotation(case, settings, skipExisting=skipExisting)

        years = [c[:4] for c in tools.getCaseRange(case, settings)]
        for year in years:
            quicklooks.metaRotationYearlyQuicklook(year, settings)
    return prod


@tools.loopify
def processAll(
    case,
    config,
    ignoreErrors=False,
    nJobs=os.cpu_count,
    fileQueue=None,
    skipExisting=True,
):
    """
    Process VISSS data for a specific case across all processing levels.

    This function orchestrates the complete processing pipeline for a given case,
    handling both leader and follower cameras where applicable. It processes
    through various levels of data products including metadata creation,
    particle detection, matching, tracking, and level 2/3 retrievals.

    Parameters
    ----------
    case : str
        Case or case range identifier for the data to process
    config : str or object
        Configuration settings for processing. Can be a path to a settings file
        or a configuration object
    ignoreErrors : bool, default False
        If True, continue processing even if errors occur in individual steps
    nJobs : int or callable, default os.cpu_count
        Number of parallel jobs to run. If callable, it will be called to get
        the number of jobs. This parameter is passed to the workers function.
    fileQueue : str, optional
        File queue for task management. If None, a temporary queue will be created.
    skipExisting : bool, default True
        Whether to skip existing files during processing

    Notes
    -----
    The actual processing flow is:

    1. Meta Events creation
    2. Level 1 detection
    3. Meta Rotation (if enabled)
    4. Level 1 matching
    5. Level 1 tracking
    6. Level 2 matching
    7. Level 2 tracking
    8. Level 2 detection (if enabled)
    9. Level 3 combined riming retrieval (if enabled)
    10. All Done marker

    For each processing level, both leader and follower cameras are processed
    where applicable. The function also handles error checking to ensure
    successful completion of each stage.

    Note that this is a rather unefficient way of processing the data and mostly
    for testing. Instead, it is recommended to use submitAll and run the workers
    separately.

    """
    if fileQueue is None:
        randString = "".join(random.choice(string.ascii_uppercase) for _ in range(10))
        fileQueue = f"/tmp/visss_{randString}"

    products = [
        "metaEvents",
        "level1detect",
    ]
    if config.level1match.processL1match:
        products += [
            "metaRotation",
            "level1match",
            "level1track",
            "level2match",
            "level2track",
        ]
    if config.level2.processL2detect:
        products += ["level2detect"]
    if config.level3.combinedRiming.processRetrieval:
        products += ["level3combinedRiming"]
    products += [
        "allDone",
    ]

    followerProducts = ["metaEvents", "level1detect", "level2detect"]
    for prod in products:
        print("#" * 10, prod, "#" * 10)
        dp1 = DataProduct(prod, case, config, fileQueue, "leader")
        dp1.submitCommands(withParents=False, skipExisting=skipExisting)
        if prod in followerProducts:
            dp2 = DataProduct(prod, case, config, fileQueue, "follower")
            dp2.submitCommands(withParents=False, skipExisting=skipExisting)
        tools.workers(fileQueue, waitTime=1, nJobs=nJobs)
        if not ignoreErrors:
            assert len(dp1.listBroken()) == 0, "leader files broken"
            assert len(dp1.listFiles()) > 0, "no leader output"
            if prod in followerProducts:
                assert len(dp2.listBroken()) == 0, "follower files broken"
                assert len(dp2.listFiles()) > 0, "no follower output"
    return


@log.catch(reraise=True)
def processRealtime(case, settings, skipExisting=True):
    """
    Process VISSS data products that do not require significant computing
    resources for a specific case or case range. Calls
    * metadata.createEvent
    * quicklooks.level0Quicklook
    * metadata.createMetaFrames
    * tools.reportLastFiles


    Parameters
    ----------
    case : str
        Case identifier for the data to process
    settings : str
        Path to settings file
    skipExisting : bool, default True
        Whether to skip existing files during processing

    Notes
    -----
    The processing sequence includes:
    1. Creating metadata events
    2. Generating level 0 quicklooks
    3. Creating metadata frames
    4. Reporting last processed files

    """
    if skipExisting:
        skipExistingStr = "--skip-existing"
    else:
        skipExistingStr = ""

    print("#" * 50)
    print(
        f"python3 -m VISSSlib metadata.createEvent {settings} {case} {skipExistingStr}"
    )
    print("#" * 50)
    metadata.createEvent(case, "all", settings, skipExisting=skipExisting)

    print("#" * 50)
    print(
        f"python3 -m VISSSlib quicklooks.level0Quicklook {settings} {case} {skipExistingStr}"
    )
    print("#" * 50)
    quicklooks.level0Quicklook(case, "all", settings, skipExisting=skipExisting)

    print("#" * 50)
    print(
        f"python3 -m VISSSlib metadata.createMetaFrames {settings} {case} {skipExistingStr}"
    )
    print("#" * 50)
    metadata.createMetaFrames(case, "all", settings, skipExisting=skipExisting)

    print("#" * 50)
    print(f"python3 -m VISSSlib tools.reportLastFiles {settings}")
    print("#" * 50)
    tools.reportLastFiles(settings)


def checkCompleteness(
    settings,
    nDays=0,
    cameras="all",
    listDuplicates=True,
    listMissing=False,
    products=[
        "metaFrames",
        "level1detect",
        "metaRotation",
        "level1match",
        "level1track",
        # "level2detect",
        "level2match",
        "level2track",
    ],
):
    """
    loop through days to check whether products have been completely processed

    Parameters
    ----------
    settings : str
        VISSS settings YAML file
    nDays : number or str, optional
        number of days N`` to go back or date ``str(YYYYMMDD)`` or date range ``str(YYYYMMDD-YYYYMMDD)`` (the default is 0)
    cameras : str, optional
        list of camera names to process (the default is "all", which means leader and follower)
    listDuplicates : bool, optional
        list duplicates (the default is True)
    listMissing : bool, optional
        list missing files (the default is False)
    products : list, optional
        products to list (the default is [ "metaFrames", "level1detect", "metaRotation", "level1match", "level1track", "level2match", "level2track", ])
    """
    config = tools.readSettings(settings)

    days = tools.getDateRange(nDays, config)

    if cameras == "all":
        cameras = [config.follower, config.leader]

    print("looking for these products:")
    print(products)

    for dd in days:
        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        for camera in cameras:
            # find files
            ff = files.FindFiles(case, camera, config)

            nMissing = {}
            for prod in products:
                if camera == config.follower and (
                    (prod in ["level1match", "level1track", "metaRotation"])
                    or prod.startswith("level2")
                ):
                    continue
                nMissing[prod] = ff.nMissing(prod)
            allDone = np.array(list(nMissing.values())) == 0

            if np.all(allDone):
                print(camera, case, "all done", np.all(allDone))
            else:
                print(camera, case, "MISSING", nMissing, "of", ff.nL0)

                firstMiss = products[np.where(np.array(allDone) == False)[0][0]]
                recFiles = np.array(ff.listFiles("level0txt"))
                nRec = len(recFiles)
                procFiles = np.array(ff.listFilesExt(firstMiss))
                nProc = len(procFiles)
                print(
                    "# level0 has",
                    nRec,
                    "files #",
                    firstMiss,
                    "has only",
                    nProc,
                    "files.",
                )

                processedTimes = np.array(
                    [
                        files.FilenamesFromLevel(f, config).datetime64
                        for f in ff.listFilesExt(firstMiss)
                    ]
                )

                if listDuplicates and nProc > nRec:
                    print("too many files processed, check these files:")
                    print("*" * 50)
                    seen = set()
                    dupes = [x for x in processedTimes if x in seen or seen.add(x)]
                    dupeFiles = []
                    for dupe in dupes:
                        dupeFiles.append(procFiles[(dupe == processedTimes)])
                    if len(dupeFiles) > 0:
                        dupeFiles = np.concatenate(dupeFiles)
                        for dupeFile in dupeFiles:
                            print(dupeFile)
                elif listMissing:
                    print("files missing")
                    print("*" * 50)

                    recTimes = [
                        files.Filenames(f, config).datetime64
                        for f in ff.listFiles("level0txt")
                    ]
                    missingTimes = set(recTimes).difference(set(processedTimes))
                    for missingTime in missingTimes:
                        print(camera, firstMiss, missingTime)
    return
