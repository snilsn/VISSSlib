import os
import socket
import sys

from loguru import logger as log

from . import (
    av,
    detection,
    distributions,
    files,
    fixes,
    level3,
    matching,
    metadata,
    products,
    quicklooks,
    tools,
    tracking,
)


@log.catch(reraise=True)
def main():
    """Main entry point for the VISSS processing pipeline."""

    print(f"{sys.executable} {sys.version} {os.getpid()} {socket.gethostname()}")

    parser = tools._create_parser()
    args = parser.parse_args()

    cmd = args.command

    # Detection commands
    if cmd == "detection.detectParticles":
        detection.detectParticles(
            args.fname, args.settings, skipExisting=args.skip_existing
        )

    # Distribution commands
    elif cmd == "distributions.createLevel2detect":
        distributions.createLevel2detect(
            args.case,
            args.camera,
            args.settings,
            skipExisting=args.skip_existing,
        )

    elif cmd == "distributions.createLevel2match":
        distributions.createLevel2match(
            args.case, args.settings, skipExisting=args.skip_existing
        )

    elif cmd == "distributions.createLevel2track":
        distributions.createLevel2track(
            args.case, args.settings, skipExisting=args.skip_existing
        )

    # Level 3 commands
    elif cmd == "level3.retrieveCombinedRiming":
        level3.retrieveCombinedRiming(
            args.case, args.settings, skipExisting=args.skip_existing
        )

    # Matching commands
    elif cmd == "matching.createMetaRotation":
        matching.createMetaRotation(
            args.case, args.settings, skipExisting=args.skip_existing
        )

    elif cmd == "matching.matchParticles":
        matching.matchParticles(
            args.fname, args.settings, skipExisting=args.skip_existing
        )

    # Metadata commands
    elif cmd == "metadata.createEvent":
        metadata.createEvent(
            args.case,
            args.camera,
            args.settings,
            skipExisting=args.skip_existing,
        )

    elif cmd == "metadata.createMetaFrames":
        metadata.createMetaFrames(
            args.case,
            args.camera,
            args.settings,
            skipExisting=args.skip_existing,
        )

    # Products commands
    elif cmd == "products.processAll":
        products.processAll(args.case, args.settings, skipExisting=args.skip_existing)

    elif cmd == "products.processRealtime":
        products.processRealtime(args.case, args.settings, args.skip_existing)

    elif cmd == "products.submitAll":
        products.submitAll(args.case, args.settings, args.task_queue)

    # Quicklook commands
    elif cmd == "quicklooks.createLevel1detectQuicklook":
        quicklooks.createLevel1detectQuicklook(
            args.case,
            args.camera,
            args.settings,
            skipExisting=args.skip_existing,
        )

    elif cmd == "quicklooks.createLevel1matchParticlesQuicklook":
        quicklooks.createLevel1matchParticlesQuicklook(
            args.case, args.settings, skipExisting=args.skip_existing
        )

    elif cmd == "quicklooks.createMetaCoefQuicklook":
        quicklooks.createMetaCoefQuicklook(
            args.case, args.settings, skipExisting=args.skip_existing
        )

    elif cmd == "quicklooks.level0Quicklook":
        quicklooks.level0Quicklook(
            args.case, "all", args.settings, skipExisting=args.skip_existing
        )

    elif cmd == "quicklooks.metaRotationYearlyQuicklook":
        quicklooks.metaRotationYearlyQuicklook(args.year, args.settings)

    # Tracking commands
    elif cmd == "tracking.trackParticles":
        tracking.trackParticles(
            args.fname, args.settings, skipExisting=args.skip_existing
        )

    # Tools commands
    elif cmd == "tools.copyLastMetaRotation":
        tools.copyLastMetaRotation(args.settings, args.from_case, args.to_case)

    elif cmd == "tools.reportLastFiles":
        output = tools.reportLastFiles(args.settings)
        print(output)

    # Worker
    elif cmd == "worker":
        assert os.path.isdir(
            args.task_queue
        ), f"Queue directory not found: {args.task_queue}"
        n_jobs = args.n_jobs if args.n_jobs is not None else os.cpu_count()
        tools.workers(args.task_queue, nJobs=n_jobs, waitTime=60)

    return 0


sys.exit(main())
