import datetime
import glob
import json
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


# LEVEL_REGISTRY (the single authoritative "what depends on what, and how
# is it built" statement for every processing level) lives in tools.py,
# not here -- it needs to be reachable from tools.checkForExisting's own
# call sites (distributions.py, matching.py, tracking.py, detection.py)
# via tools.resolveLevelParents, and tools.py sits below products.py in
# the import graph. Use tools.LEVEL_REGISTRY / tools.resolveLevelParents.


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

        if (
            level in ("metaRotation", "level1match", "level1track", "level2match", "level2track")
            and not self.config.level1match.processL1match
        ):
            raise ValueError(
                f"{level} was requested, but level1match.processL1match is "
                f"False in {self.config.filename}. This deployment has the "
                f"stereo-matching branch disabled; set "
                f"level1match.processL1match: true in the settings file to "
                f"process this level, or call the underlying "
                f"matching.*/tracking.* function directly (bypassing "
                f"DataProduct) if this is an intentional one-off."
            )

        try:
            levelSpec = tools.LEVEL_REGISTRY[level]
        except KeyError:
            raise ValueError(f"Do not understand {level}")
        if levelSpec.get("leaderOnly", False):
            assert camera == "leader"
        self.parentNames = levelSpec["parents"](self.camera, self.config)
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

        if (not self.dataTransfered) and (self.config.end == "today"):
            log.warning(
                f"{self.case} {self.relatives}: no data found (yet?) in {self.fn.fnamesPattern.level0txt}"
            )
            return []

        if (
            skipExisting
            and isComplete
            and self._upToDateWithParents
            and self.parentsComplete
        ):
            if withParents:
                log.info(f"{self.case} {self.relatives}: everything processed")
            return []
        if isComplete and (not self._upToDateWithParents):
            for name, upToDate in self._upToDateWithParentsDict.items():
                if not upToDate:
                    log.warning(
                        f"{self.case} {self.relatives} redoing level, parent {name} was updated more recently"
                    )
        if self.parentsComplete and self._parentsUpToDateWithGrandparents:
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
            # self._parentsUpToDateWithGrandparents is False: the coarse
            # day-level heuristic behind _upToDateWithParentsDict (MIN of
            # this product's own file mtimes vs MAX of a parent's -- see
            # its docstring) flagged some parent as possibly stale
            # relative to its own parents. That comparison is deliberately
            # conservative for THIS product (see _upToDateWithParentsDict),
            # but using it here to decide whether to even attempt
            # generating a command is a different question: it can
            # false-positive under partial/rolling reprocessing, e.g. one
            # parent file touched at an unrelated time poisons the whole
            # day's flag even though every parent file individually
            # already reflects current grandparent content (confirmed via
            # hyytiala2_v3's level2track getting permanently stuck this
            # way after iterative level1match self-heal fixes touched
            # scattered files across many days). Before giving up, ask
            # each flagged parent directly (via its own, per-file-granular
            # generateAllCommands) whether it actually still has real work
            # pending -- if none do, the flag was a false positive and it's
            # safe to generate this product's own command anyway.
            stillPending = any(
                len(
                    parent.generateAllCommands(skipExisting=True, withParents=False)
                )
                > 0
                for name, parent in self.parents.items()
                if not parent._upToDateWithParents
            )
            if stillPending:
                log.warning(
                    f"{self.case} {self.relatives} no commands generated, grandparents older"
                )
                commands = []
            else:
                log.info(
                    f"{self.case} {self.relatives} grandparents flagged older by the "
                    "coarse day-level check, but have no real per-file work pending -- "
                    "generating own command anyway"
                )
                commands = self.generateCommands(skipExisting=skipExisting)
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
        try:
            command = tools.LEVEL_REGISTRY[self.level]["command"]
        except KeyError:
            raise ValueError(f"Do not understand {self.level}")

        kind = command[0]
        if kind == "none":
            return []
        elif kind == "daily":
            _, call = command
            return self._commandTemplateDaily(
                call, skipExisting=skipExisting, nCPU=nCPU, bin=bin
            )
        elif kind == "l1":
            _, originLevel, call, extraOrigin = command
            return self._commandTemplateL1(
                originLevel,
                call,
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
                extraOrigin=extraOrigin,
            )
        elif kind == "touch":
            outFile = self.fn.fnamesDaily["allDone"]
            command = f"mkdir -p {os.path.dirname(outFile)} && touch {outFile}"
            return [(command, outFile)]
        else:
            raise ValueError(f"Do not understand command kind {kind} for {self.level}")

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

        if (extraOrigin is not None) and (len(self.fn.listNoData(extraOrigin)) > 0):
            # extraOrigin (e.g. metaRotation) is marked nodata for this day:
            # nothing will ever be produced here. Rather than spawning a
            # doomed subprocess per origin file (matchParticles would just
            # hit the same nodata file and write its own .nodata marker),
            # write those markers directly -- same filenames matchParticles
            # itself would use -- so this level gets real files with real
            # mtimes and the normal nMissing/isComplete/freshness machinery
            # resolves it without any special-casing.
            log.warning(
                f"{self.relatives} {extraOrigin} is nodata: writing {self.level} "
                f"nodata markers directly instead of spawning doomed commands"
            )
            for pName in self.fn.listFilesExt(originLevel):
                if originLevel.startswith("level0"):
                    f1 = files.Filenames(pName, self.config)
                else:
                    f1 = files.FilenamesFromLevel(pName, self.config)
                outFile = f1.fname[self.level]
                if len(glob.glob(f"{outFile}*")) == 0:
                    f1.writeStatus(
                        self.level, "nodata", f"{extraOrigin} is nodata for {pName}"
                    )
            return []

        minMtime = tools.reprocessBreakpoint(self.level)

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

            if (len(exisiting) >= 1) and (minMtime is not None):
                pastBreakpoint = os.path.getmtime(exisiting[0]) >= minMtime
            else:
                pastBreakpoint = True

            if (
                skipExisting
                and (len(exisiting) >= 1)
                and (os.path.getmtime(pName) < os.path.getmtime(exisiting[0]))
                and extraOlder
                and pastBreakpoint
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

        if self.level in files.dailyLevels:
            outFile = self.fn.fnamesDaily[self.level]
            exisiting = glob.glob(f"{outFile}*")
            if skipExisting and (len(exisiting) >= 1) and (self._upToDateWithParents):
                log.info(f"{self.relatives} skip exisiting {exisiting[0]}")
                return []
        else:
            # levels like metaFrames: one CLI call covers a whole day (via
            # tools.loopify_with_camera) but produces one output file per
            # level0 input file, so there is no single fnamesDaily entry to
            # check against (only files.dailyLevels have exactly one output
            # file per day). Use nMissing instead, and build a synthetic
            # per-day marker path -- matching fnamesPatternExt's naming
            # scheme so it is still picked up by the broken/nodata glob --
            # purely for runCommandInQueue's locking/broken-file bookkeeping.
            outFile = os.path.join(
                self.fn.outpath[self.level],
                f"{self.level}_V{self.fn.version}_{self.fn.camera}_{self.fn.case}.nc",
            )
            if (
                skipExisting
                and (self.fn.nMissing(self.level) == 0)
                and (self._upToDateWithParents)
            ):
                log.info(f"{self.relatives} skip exisiting {outFile}")
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
    def _upToDateWithParentsDict(self):
        """
        Check whether this product was (re)processed after each of its
        parents -- i.e. whether it reflects each parent's current state,
        or is stale and needs to be redone.

        Compares each parent's newest file (`parent.newestFileCreation`, a
        MAX over that parent's files -- its most recent update) against
        this product's OLDEST file (`self.oldestFileCreation`, a MIN
        over this product's own files). The oldest file has to be used
        here, not the newest: if even a single one of this product's
        own files predates a parent's latest update, that one file is
        stale, so the product as a whole is not fully up to date --
        using the newest file instead would let one recently-touched
        file mask staleness in all the others.

        A product that is complete but has zero real files (e.g. a day
        with a confirmed data gap upstream, so nothing was ever expected
        here) has no meaningful file time to compare against its
        parents' mtimes. Without an exception, that would look
        "infinitely stale" against any real parent mtime and would block
        every descendant -- and ultimately allDone -- from ever being
        marked up to date. Such a product is treated as vacuously
        up to date with all of its parents instead.

        This exemption (`vacuouslyFresh`, see below) must be based on
        REAL files only (`selfVacuous`, zero results from `listFiles()`),
        not `newestFileCreation == 0`: `newestFileCreation` is computed
        from `listFilesExt()`, which -- unlike `listFiles()` -- also
        counts `.nodata`/`.broken.txt` sentinels, so it is essentially
        never 0 once a sentinel has been written for this level+camera+day
        (its own mtime keeps it nonzero). A `newestFileCreation == 0`
        check therefore only fires for a product with *no file of any
        kind*, missing the far more common case of a day that already has
        its terminal `.nodata`/`.broken.txt` sentinel -- exactly the "zero
        real files, nothing was ever expected here" case this exemption
        is meant to cover. Confirmed causing real, permanently-unfixable
        `generateAllCommands()` noise: hyytiala_v1's level2track for
        2022-04-21 (a permanent no-data day) kept getting regenerated as
        "pending" forever, every time metaRotation (a real, non-vacuous
        grandparent elsewhere in the chain) was legitimately touched,
        because `newestFileCreation` for the `.nodata`-only level2track
        was its own sentinel's nonzero mtime, not 0 -- so `vacuouslyFresh`
        was always False and the plain parent-mtime comparison kicked in
        against a self that can, by definition, never produce a newer
        real file to catch up with.

        The mirror image applies per parent: a parent that is complete
        but only has a .nodata/.broken.txt sentinel (no real output) can
        have that sentinel rewritten at any time -- e.g. a bulk backfill
        or a retried failure -- without there being any new real
        information for us to react to. Letting a sentinel's mtime alone
        mark us stale would force endless, permanently unproductive
        "redo" commands (the redo is always skip-existing and a no-op,
        since there is nothing to redo), so such a parent is treated as
        vacuously old for this comparison instead.

        Returns
        -------
        dict
            Dictionary mapping parent names to boolean values indicating
            whether this product is up to date with each parent
        """
        # Mirrors parentVacuous below, but for self: do THIS product's own
        # files already reflect "no real data" (only a .nodata/.broken.txt
        # sentinel, no real output)? Needed to guard parentVacuous --
        # without it, a parent that only just turned vacuous (e.g. a
        # badData window added to the config after this product had
        # already been successfully computed from what's now known-bad
        # upstream data) would be permanently ignored: this product's own
        # stale-but-real output would never get revisited into a matching
        # nodata/broken marker, since parentVacuous alone would keep
        # calling it up to date forever (confirmed against
        # gochang_v1/level1match for 20251209-20251221: metaRotation
        # turned broken there, but level1match kept its pre-badData real
        # files and was never flagged stale). Once both sides are
        # sentinel-only there truly is nothing left to react to, so the
        # skip is safe again.
        selfVacuous = self.isComplete and (len(self.listFiles()) == 0)
        # see the docstring section above for why this must be
        # `selfVacuous`, not a `newestFileCreation == 0` check
        vacuouslyFresh = selfVacuous
        # levels whose CLI call covers a whole day (tools.LEVEL_REGISTRY's
        # "daily" command kind) but that are NOT in files.dailyLevels (i.e.
        # they produce many per-file outputs, one per level0 input file,
        # rather than a single daily aggregate -- today this is only
        # metaFrames) have their own files' creation times track their
        # parent's files' creation times roughly 1:1 throughout the day, so
        # the FIRST file of the day is always older than the parent's LAST
        # file of that same day regardless of whether the day is fully and
        # correctly processed -- comparing self.oldestFileCreation against
        # parent.newestFileCreation is a structural false positive for
        # these levels. Using self.newestFileCreation instead is safe here
        # specifically because each such output has a 1:1 correspondence to
        # a single input file with no OTHER shared parent able to
        # invalidate every sibling file at once -- unlike e.g. level1match/
        # level1track (also many-files-per-day, but "l1"-kind commands, not
        # "daily"-kind, so unaffected by this branch), where a single
        # updated shared parent (metaRotation/level1match) legitimately can
        # invalidate every existing per-file sibling simultaneously, which
        # is exactly the "one recently-touched file masks staleness in all
        # the others" failure mode the oldest-file comparison below is
        # designed to catch for them (see
        # test_generateAllCommands_survives_stale_day_level_false_positive).
        manyFilesPerDailyCall = (
            tools.LEVEL_REGISTRY[self.level]["command"][0] == "daily"
            and self.level not in files.dailyLevels
        )
        selfComparisonTime = (
            self.newestFileCreation if manyFilesPerDailyCall else self.oldestFileCreation
        )
        upToDateWithParentsDict = tools.DictNoDefault()
        for name, parent in self.parents.items():
            parentVacuous = parent.isComplete and (len(parent.listFiles()) == 0)
            isUpToDate = (
                vacuouslyFresh
                or (parentVacuous and selfVacuous)
                or (parent.newestFileCreation < selfComparisonTime)
            )
            if (self.level == "level1detect") and (parent.level == "metaEvents"):
                # special case: no need to do level1detect again due to updated metaEvents
                upToDateWithParentsDict[name] = True
            else:
                upToDateWithParentsDict[name] = isUpToDate
            if not upToDateWithParentsDict[name]:
                log.debug(
                    f"{self.relatives} has files older "
                    f"({tools.timestamp2str(selfComparisonTime)}) than parent "
                    f"{name}'s newest ({tools.timestamp2str(parent.newestFileCreation)})",
                )
        return upToDateWithParentsDict

    @cached_property
    def _upToDateWithParents(self):
        """
        Check if this product was (re)processed after all of its parents.

        Returns
        -------
        bool
            True if this product is up to date with all parents, False otherwise
        """
        upToDateWithParents = (
            np.all(list(self._upToDateWithParentsDict.values()))
            and self._pastReprocessBreakpoint
        )
        return upToDateWithParents

    @cached_property
    def _pastReprocessBreakpoint(self):
        """
        Whether this product's own files are all at or after any
        pending reprocessing breakpoint for this level
        (tools.REPROCESS_AFTER) -- True (vacuously) if this level has
        no such entry, or if it has no real files yet. Reuses
        oldestFileCreation (already computed via the cached freshness
        summary -- see _freshnessSummary) rather than re-stat'ing or
        opening any file, so this costs nothing extra. Folded into
        _upToDateWithParents so a code change that invalidates
        previously produced files -- without any parent file changing,
        which the normal mtime-based check can't see -- still forces
        regeneration. See tools.REPROCESS_AFTER for why/when to use
        this.

        Returns
        -------
        bool
        """
        minMtime = tools.reprocessBreakpoint(self.level)
        if minMtime is None:
            return True
        if self.oldestFileCreation == 0:
            return True
        return self.oldestFileCreation >= minMtime

    @cached_property
    def _parentsUpToDateWithGrandparents(self):
        """
        Check if parents were themselves (re)processed after their own
        parents (this product's grandparents) -- i.e. whether the whole
        ancestor chain, not just the direct parents, is current.

        Returns
        -------
        bool
            True if all parents are up to date with their own parents, False otherwise
        """
        parentsUpToDateWithGrandparents = True
        for name, parent in self.parents.items():
            parentsUpToDateWithGrandparents = (
                parentsUpToDateWithGrandparents and parent._upToDateWithParents
            )
            log.debug(
                f"{self.relatives} parent {name} is up to date with its (grand)parents: { parent._upToDateWithParents}"
            )
        return parentsUpToDateWithGrandparents

    @cached_property
    def _freshnessSummary(self):
        """
        (n, oldest, newest) mtime summary for this product's own files.

        Read from the on-disk cache maintained by
        tools.readLevelSummary/writeLevelSummary when possible, falling
        back to a real glob+stat scan (the previous, exact behavior of
        newestFileCreation/oldestFileCreation) on any cache miss.

        The cache is fenced against a "touch" marker that every real
        write bumps (tools.open2/to_netcdf2's hooks): the fence is
        captured before this scan starts, and the freshly computed
        summary is only published (tools.writeLevelSummary) if the
        fence still has that exact value afterwards. That's what makes
        this safe with many concurrent SLURM workers writing files for
        the same level+camera+day -- a scan that overlaps a concurrent
        write never gets to cache what it saw, so a later reader can't
        be handed stale data; it just falls back to scanning again,
        the same as if nothing had ever been cached.

        Raw/passthrough levels (level0, level0txt, ...; identified by
        having no per-level output directory in self.fn.outpath) are
        never written by us via open2/to_netcdf2 -- there's nothing for
        a marker to cache -- so those always take the plain scan path.

        Same reasoning applies to "touch"-kind levels (currently just
        allDone): generateCommands builds those as a bare shell
        `touch <file>`, which never goes through open2/to_netcdf2 and
        so never bumps the fence marker either. Caching them anyway
        would mean the very first summary ever computed for a given
        level+camera+day is trusted forever after -- the fence stays
        at 0 (no marker file == no write has ever gone through the
        hook), so `readLevelSummary`'s fence check (`fence == 0`)
        always passes, and no amount of re-touching the real file is
        ever able to invalidate the cached (and increasingly stale)
        oldest/newest. That silently broke allDone's
        _upToDateWithParents check: it kept comparing parents against
        a months-old cached mtime even after the sentinel had just
        been freshly re-touched (confirmed by hand against
        gochang_v1/20251120: repeated `touch` via the real task queue
        updated the file on disk every time, but the cached summary --
        and thus every DataProduct built fresh afterwards -- kept
        reporting the original creation time regardless), so allDone
        commands were regenerated on every single check, forever, no
        matter how many times they were (correctly) run.

        Returns
        -------
        tuple
            (n, oldest, newest) -- file count and min/max mtime, or
            (0, 0, 0) if this product has no files.
        """
        cacheable = (self.level in self.fn.outpath) and (
            tools.LEVEL_REGISTRY[self.level]["command"][0] != "touch"
        )
        if cacheable:
            cached = tools.readLevelSummary(self.fn, self.level)
            if cached is not None:
                return cached
            fenceBefore = tools.getLevelTouchTime(self.fn, self.level)

        fileList = self.listFilesExt()
        if len(fileList) > 0:
            mtimes = [os.path.getmtime(f) for f in fileList]
            summary = (len(fileList), np.min(mtimes), np.max(mtimes))
        else:
            summary = (0, 0, 0)

        if cacheable:
            tools.writeLevelSummary(
                self.fn, self.level, *summary, fenceBefore, self.config
            )
        return summary

    @cached_property
    def newestFileCreation(self):
        """
        Get the creation time of this product's most recently modified
        file.

        See `oldestFileCreation` for the complementary MIN-based
        property used in staleness comparisons.

        Returns
        -------
        float
            Modification time of the newest file
        """
        return self._freshnessSummary[2]

    @cached_property
    def oldestFileCreation(self):
        """
        Get the creation time of this product's least recently modified
        file.

        `newestFileCreation` (the newest file) answers "when was this product
        last touched" -- useful for reporting. Freshness comparisons
        against a parent need the opposite: if even one of this
        product's own files predates a parent's newest update, that one
        file is stale, so the product as a whole is not fully up to
        date. Using `newestFileCreation` there would let a single
        recently-touched file mask staleness in all the others (see
        `_upToDateWithParentsDict`).

        Returns
        -------
        float
            Modification time of the oldest file
        """
        return self._freshnessSummary[1]

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
            tools.timestamp2str(self.newestFileCreation),
            "younger than parents",
            self._upToDateWithParents,
        )
        if nMissing > 0:
            print(
                " " * 5,
                [(p, self.fn.nMissing(p.split("_")[1])) for p in self.parentNames],
            )
        if withParents:
            for name, parent in self.parents.items():
                parent.report(withParents=False)

    def reportBroken(self, withParents=False, returnAllInformation=True):
        """Report broken files.

        Parameters
        ----------
        withParents : bool, default False
            Whether to include parent reports
        returnAllInformation : bool, default True
            Whether to return all information

        Returns
        -------
        pandas.DataFrame
            DataFrame with broken file information
        """
        import pandas as pd

        results_data = []
        for brokenFile in self.listBroken():
            with open(brokenFile) as f:
                lines = f.readlines()
            if len(lines) == 1:
                command = "n/a"
                outfile = "n/a"
                gist = lines[0].rstrip()
                fullError = "".join(lines)
            else:
                command = lines[1][9:].split(";")[-1].strip()
                outfile = lines[2][9:].rstrip()
                gist = f"{lines[-2].rstrip(), lines[-1].rstrip()}"
                fullError = "".join(lines[4:])
            ff = files.FilenamesFromLevel(brokenFile, self.config)
            index = f"{ff.camera.split("_")[0]}_{ff.case}_{self.level}"

            # Create a dict and append it to the list
            row = {
                "index": index,
                "command": command,
                "outfile": outfile,
                "gist": gist,
                "fullError": fullError,
            }
            results_data.append(row)

        if len(results_data) == 0:
            df = pd.DataFrame(
                columns=["index", "command", "outfile", "gist", "fullError"]
            )
        else:
            df = pd.DataFrame(
                results_data,
            )

        df = df.set_index("index")

        if withParents:
            df1 = [df]
            for name, parent in self.parents.items():
                # dual-camera parents (e.g. level1detect) are stored as a
                # list of DataProduct instances rather than a single one
                if not isinstance(parent, list):
                    parent = [parent]
                for p in parent:
                    df1.append(
                        p.reportBroken(
                            withParents=False,
                            returnAllInformation=returnAllInformation,
                        )
                    )
            df = pd.concat(df1)
            # df = df.iloc[~df.index.duplicated()]
            df = df.sort_index()

        if returnAllInformation:
            return df
        else:
            return df[["command", "gist"]]


    @cached_property
    def dataTransfered(self):
        """
        Check if data is available for this product, or -- if not --
        whether its absence has been confirmed as a genuine gap rather
        than data that simply has not synced yet.

        Returns
        -------
        bool
            True if data is available or the gap is confirmed, False if
            still pending (data may yet arrive).
        """
        return not self.fn.isDataTransferPending("level0txt")

    @cached_property
    def allComplete(self):
        """
        Check if this product and all its dependencies are complete.

        Returns
        -------
        bool
            True if all is complete, False otherwise
        """
        return self.isComplete and self._upToDateWithParents and self.parentsComplete

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
                # Raw os.remove, same as runCommandInQueue's own
                # shutil.copy used to be before it started bumping the
                # marker (see 6c7d3bb) -- without this, the freshness
                # cache still counts the just-deleted file, so a repaired
                # day can keep looking complete/up to date until
                # something else happens to rewrite it.
                tools._touchLevelMarker(fname, self.config)
        if withNoData:
            for fname in self.listNoData():
                assert fname.endswith("nodata")
                try:
                    os.remove(fname)
                except FileNotFoundError:  # usally caused by caching listBroken
                    log.warning(f"{fname} not found")
                else:
                    log.warning(f"{fname} removed")
                    tools._touchLevelMarker(fname, self.config)
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
            tools._touchLevelMarker(fname, self.config)
        if withParents:
            for name, parent in self.parents.items():
                if not isinstance(parent, list):
                    parent = [parent]
                [p.cleanUpDuplicates(withParents=False) for p in parent]

    def repairStaleFreshnessCache(self, withParents=False):
        """
        Find and invalidate a stale on-disk freshness-cache summary for
        this product.

        tools.writeLevelSummary caches this level+camera+day's (file
        count, oldest mtime, newest mtime) in a small `.done` file (see
        files.FindFiles.markerPath) so _freshnessSummary doesn't have to
        re-glob and re-stat every file on every DAG check. That cache is
        only ever supposed to go stale for the length of one write --
        every real write bumps a "touch" fence via tools.open2/to_netcdf2
        (tools._touchLevelMarker), which invalidates it immediately.

        In practice a handful of write paths have been found that write
        a real terminal artifact (a real file, `.broken.txt`, or
        `.nodata`) *without* going through that hook, so the cache never
        gets invalidated and keeps reporting its original (n, oldest,
        newest) forever, however many times the real file is later
        rewritten: allDone's own `touch` command (c5341c5),
        runCommandInQueue's `.broken.txt` write on task failure (6c7d3bb,
        5c136b0), and cleanUpBroken/cleanUpDuplicates' plain os.remove
        (fixed alongside this method). There is no guarantee that is an
        exhaustive list -- a long-running worker process holding
        pre-fix code in memory reproduces the exact same symptom
        regardless of how many such bugs get fixed on disk, since it
        never re-imports the fix until it restarts. This method doesn't
        depend on knowing the cause: it just compares the cached summary
        against a real, live scan and invalidates it on any disagreement,
        so it's safe to run as a periodic sweep (e.g. before a QC pass)
        independent of whatever bug produced the mismatch.

        Parameters
        ----------
        withParents : bool, default False
            Whether to also repair parents' caches

        Returns
        -------
        bool
            True if this product's own cache was found stale and repaired
            (does not reflect whether any parent's cache was repaired --
            check the log for those).
        """
        repaired = False
        # Raw/passthrough levels (level0, level0txt, ...) have no
        # per-level output directory at all -- nothing is ever cached
        # for them (see _freshnessSummary's own cacheable check) and
        # markerPath has no outpath entry to build a path from.
        if self.level not in self.fn.outpath:
            cached = None
        else:
            donePath = self.fn.markerPath(self.level, "done")
            try:
                with open(donePath) as f:
                    cached = json.load(f)
            except (OSError, ValueError):
                cached = None
        if cached is not None:
            cachedNewest = cached.get("newest")
            if cachedNewest is not None:
                realNewest = max(
                    (os.path.getmtime(f) for f in self.listFilesExt()), default=0
                )
                # a few seconds of slop: filesystem mtime resolution and
                # the gap between this scan's individual os.path.getmtime
                # calls are not meaningfully "staleness", just noise
                if abs(realNewest - cachedNewest) > 5:
                    touchPath = self.fn.markerPath(self.level, "touch")
                    tools._invalidateLevelCacheAt(touchPath, donePath, self.config)
                    log.warning(
                        f"{self.relatives} stale freshness cache repaired "
                        f"(cached newest {tools.timestamp2str(cachedNewest)}, "
                        f"real newest {tools.timestamp2str(realNewest)})"
                    )
                    repaired = True
        if withParents:
            for name, parent in self.parents.items():
                if not isinstance(parent, list):
                    parent = [parent]
                for p in parent:
                    repaired = (
                        p.repairStaleFreshnessCache(withParents=True) or repaired
                    )
        return repaired


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

    def repairStaleFreshnessCache(self, withParents=False):
        """Repair stale freshness caches for all instances in this range.

        DataProduct.repairStaleFreshnessCache relies on per-case
        attributes (self.fn, self.parents, ...) that a DataProductRange
        does not have, so it cannot simply be inherited -- same reasoning
        as allComplete/report/reportBroken/listBroken above.

        Parameters
        ----------
        withParents : bool, default False
            Whether to also repair parents' caches

        Returns
        -------
        bool
            True if any instance's (or, with withParents, any parent's)
            cache was found stale and repaired.
        """
        return any(
            dp.repairStaleFreshnessCache(withParents=withParents)
            for dp in self._instances
        )

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

    @cached_property
    def allComplete(self):
        """
        Check if this product and all its dependencies are complete,
        for every case in this range.

        Overridden from DataProduct: without this, `allComplete` (a
        cached_property) would be found via normal inheritance and
        evaluated with `self` bound to this DataProductRange instance
        instead of an individual case's DataProduct -- but the
        properties it depends on (isComplete, _upToDateWithParents,
        parentsComplete, and transitively newestFileCreation/parents/etc.)
        assume per-case attributes like `self.fn`/`self.parents` that
        only exist on a genuine single-case DataProduct. That failure
        is an AttributeError, which Python's attribute-lookup protocol
        silently swallows and retries via `__getattr__` -- repeatedly,
        once per property in the chain -- eventually surfacing as a
        confusing bare `AttributeError: allComplete` with the real
        cause (whatever actually went wrong, e.g. a genuinely missing
        parent, or a file removed mid-scan by a concurrent worker)
        discarded.

        Returns
        -------
        bool
            True only if every case in this range is complete
        """
        return all(dp.allComplete for dp in self._instances)

    def report(self, withParents=True):
        """Print a report about this product's status for all instances.

        DataProduct.report relies on per-case attributes (self.fn,
        self.newestFileCreation, self.parents, ...) that a DataProductRange
        does not have, so it cannot simply be inherited -- it needs to be
        run per case instance, same as reportBroken/listBroken.

        Parameters
        ----------
        withParents : bool, default True
            Whether to include parent reports
        """
        for dp in self._instances:
            dp.report(withParents=withParents)

    def reportBroken(self, withParents=False, returnAllInformation=True):
        """Report broken files for all instances.

        DataProduct.reportBroken relies on per-case attributes (self.case,
        self.fn, self.parents, ...) that a DataProductRange does not have,
        so it cannot simply be inherited -- it needs to be run per case
        instance and the results concatenated, same as listBroken/listFiles.

        Parameters
        ----------
        withParents : bool, default False
            Whether to include parent reports
        returnAllInformation : bool, default True
            Whether to return all information

        Returns
        -------
        pandas.DataFrame
            DataFrame with broken file information across all cases
        """
        import pandas as pd

        dfs = [
            dp.reportBroken(
                withParents=withParents,
                returnAllInformation=returnAllInformation,
            )
            for dp in self._instances
        ]
        if len(dfs) == 0:
            columns = (
                ["command", "outfile", "gist", "fullError"]
                if returnAllInformation
                else ["command", "gist"]
            )
            return pd.DataFrame(columns=columns)

        df = pd.concat(dfs)
        df = df.sort_index()
        return df

    def generateAllCommands(self, skipExisting=True, withParents=True):
        """Generate all pending commands for all instances in this range.

        DataProduct.generateAllCommands relies on per-case attributes
        (self.parents, self.isComplete, ...) that a DataProductRange
        does not have, so it cannot simply be inherited -- same reasoning
        as allComplete/report/reportBroken/listBroken above. Without this
        override, `generateAllCommands` is found via normal inheritance
        (DataProductRange subclasses DataProduct) and runs with `self`
        bound to the whole range instead of a per-case DataProduct: its
        `self.parents.items()` then resolves through `__getattr__`'s
        generic per-attribute aggregation, which -- for a dict whose
        values are DataProduct objects -- returns a list of DataProduct
        instances per parent name instead of a single one, crashing with
        `AttributeError: 'list' object has no attribute
        '_upToDateWithParents'` deep inside the per-case implementation.

        Parameters
        ----------
        skipExisting : bool, default True
            Whether to skip existing files
        withParents : bool, default True
            Whether to include parent commands

        Returns
        -------
        list
            Deduplicated list of commands pending across every case in
            this range.
        """
        return tools._aggregate(
            [
                dp.generateAllCommands(skipExisting=skipExisting, withParents=withParents)
                for dp in self._instances
            ]
        )

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
    nJobs=os.cpu_count(),
    fileQueue=None,
    skipExisting=True,
    maxIdleSeconds=60,
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
    nJobs : int, default os.cpu_count()
        Number of parallel jobs to run. This parameter is passed to the
        workers function.
    fileQueue : str, optional
        File queue for task management. If None, a temporary queue will be created.
    skipExisting : bool, default True
        Whether to skip existing files during processing
    maxIdleSeconds : int, default 60
        Passed through to tools.workers -- how long each stage's workers
        wait for confirmation the queue is genuinely empty before moving
        on to the next level. See tools.worker1's docstring; callers not
        running against a real SLURM allocation (e.g. tests) can pass a
        much smaller value to skip most of this wait.

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
        tools.workers(
            fileQueue, waitTime=1, nJobs=nJobs, maxIdleSeconds=maxIdleSeconds
        )
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
