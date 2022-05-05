import sys
import socket
import os

from . import files
from . import quicklooks
from . import av
from . import tools
from . import detection
from . import metadata
from . import matching
from . import tracking
from . import time
from . import scripts



def main():

    print(sys.executable, file=sys.stderr)
    print(sys.version, file=sys.stderr)
    print('%i %s\r' % (os.getpid(), socket.gethostname()), file=sys.stderr)


    if sys.argv[1] == "scripts.loopCreateEvents":
        settings = sys.argv[2]
        nDays = int(sys.argv[3])
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopCreateEvents(settings, skipExisting=skipExisting, nDays = nDays)

    elif sys.argv[1] == "quicklooks.createLv1Quicklook":
        case = sys.argv[2]
        camera = sys.argv[3]
        settings = sys.argv[4]
        lv2Version = sys.argv[5]
        try:
            skipExisting = bool(int(sys.argv[6]))
        except IndexError:
            skipExisting = True

        quicklooks.createLv1Quicklook(case, camera, config, lv2Version,
                           skipExisting=skipExisting)

    elif sys.argv[1] == "quicklooks.createLv3CoefQuicklook":
        case = sys.argv[2]
        config = sys.argv[3]
        lv3Version = sys.argv[4]
        try:
            skipExisting = bool(int(sys.argv[5]))
        except IndexError:
            skipExisting = True
        quicklooks.createLv3CoefQuicklook(case, config, lv3Version,
                               skipExisting=skipExisting)


    elif sys.argv[1] == "scripts.loopLv1Quicklooks":
        settings = sys.argv[2]
        lv2Version = sys.argv[3]
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopLv1Quicklooks(settings, lv2Version, skipExisting=skipExisting)


    elif sys.argv[1] == "scripts.loopLv3CoefQuicklooks":
        settings = sys.argv[2]
        lv3Version = sys.argv[3]
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopLv3CoefQuicklooks(settings, lv3Version, skipExisting=skipExisting)


    elif sys.argv[1] == "detection.detectParticles":

        fname = sys.argv[2]
        settings = sys.argv[3]

        detection.detectParticles(fname, settings)

    else:
        print(f"Do not understand {sys.argv[1]}")
        return 1
    return 0


sys.exit(main())