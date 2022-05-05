# -*- coding: utf-8 -*-

import uuid

import numpy as np
import os
import sys
import datetime

from copy import deepcopy

import pandas as pd
import xarray as xr
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib as mpl

from image_packer import packer


import warnings
try:
    import cv2
except ImportError:
    warnings.warn("opencv not available!")

from .tools import nicerNames, readSettings
from .av import VideoReaderMeta

from . import files


def crop(image):
    """
    crop black image parts
    """
    y_nonzero, x_nonzero = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero)+1, np.min(x_nonzero):np.max(x_nonzero)+1]


def createLv1Quicklook(timestamp, camera, config, lv2Version,
                       minBlur=100,
                       minSize=17,
                       container_width=200,
                       container_height_max=300,
                       nTiles=60,
                       nRows=4,
                       extra=1,
                       readParticlesFromFiles=True,
                       skipExisting=True,
                       ):

    if type(config) is str:
        config = readSettings(config)

    total_width = (container_width + extra) * nTiles // nRows
    max_height = (20 + container_height_max) * nRows + 60

    # let use a matplotlib font becuase we can be sure it is there
    mpl_data_dir = os.path.dirname(mpl.matplotlib_fname())
    mpl_ttf_dir = os.path.join(mpl_data_dir, 'fonts', 'ttf')
    font = ImageFont.truetype(f"{mpl_ttf_dir}/DejaVuSans.ttf", 35)
    fontL = ImageFont.truetype(f"{mpl_ttf_dir}/DejaVuSans.ttf", 16)

    ff = files.FindFiles(timestamp, camera, config, lv2Version)

    site = config["site"]

    particlesPloted = 0

    if os.path.isfile(ff.quicklook1) and skipExisting:
        print("SKIPPING - file exists", ff.quicklook1)
        return None, None

    if (len(ff.fnames1) == 0) and (len(ff.fnames0) > 0):
        print("NO DATA YET ", ff.quicklook1)
        return None, None

    if not ff.isCompleteL1:
        print("NOT COMPLETE YET %i/%i %s" %
              (len(ff.fnames1Ext), len(ff.fnames0), ff.quicklook1))
#         if (len(ff.fnames1Ext) == len(ff.fnames0)):
#             afshgsa
        return None, None

#     else:
    print("RUNNING ", ff.quicklook1)

    ff.createQuicklookDirs()

    dats2 = []

    for fname1 in ff.fnames1:
        fname2 = fname1.replace('level1', 'level2')
        try:
            dat2 = xr.open_dataset(fname2)
        except FileNotFoundError:
            print("FileNotFoundError (probably no data)", fname2)
            continue
        dat2 = dat2[["Dmax", "blur", "touchesBorder",
                     "record_time", "record_id", "roi"]]
        dat2 = dat2.where((dat2.blur > minBlur) & (dat2.Dmax > minSize) & (
            ~(dat2.touchesBorder.any('side')))).dropna('pid')

        if len(dat2.pid) == 0:
            continue
        dat2 = dat2[["record_time", "record_id", "roi"]]
        dat2 = dat2.sel(ROI_elements=["w", "h"])

        dat2 = dat2.expand_dims(dict(file=[fname1]))
    #     dat2 = dat2.set_coords(dict(file = fname2))
        dat2 = dat2.stack(fpid=("file", "pid"))

        dats2.append(dat2)

    print("opened")
    new_im = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))

    if len(dats2) == 0:

        draw = ImageDraw.Draw(new_im)
        draw.text((total_width//3, max_height//3),
                  'no raw data', (0, 0, 0), font=font)

        nParticles = 0

    else:
        limDat = xr.concat(dats2, dim='fpid')
    #         limDat = dats2
        print("merged")

        if len(limDat.fpid) == 0:
            print("TOO FEW DATA ", ff.quicklook1)

            draw = ImageDraw.Draw(new_im)
            draw.text((total_width//3, max_height//3),
                      'no data recorded', (0, 0, 0), font=font)

            nParticles = 0

        else:
            print('Total number of particles for plotting %i' %
                  len(limDat.fpid))

            timeSteps = np.percentile(
                limDat.record_time, np.linspace(0, 100, nTiles+1))

            mosaics = []

            videos = {}
            for tt, (t1, t2) in enumerate(zip(timeSteps[:-1], timeSteps[1:])):

                if tt == len(timeSteps) - 2:
                    whereCond = (limDat.record_time >= t1)
                else:
                    whereCond = ((limDat.record_time >= t1) &
                                 (limDat.record_time < t2))

                thisDat = limDat.where(whereCond).dropna('fpid')
                totalArea = 0

                # select pids randomly, figure out how much we need, and sort them again
                pids = deepcopy(thisDat.fpid.values)
                nPids = len(pids)
                np.random.seed(tt)
                np.random.shuffle(pids)

                containerSize = (container_width*container_height_max)

                if nPids < 5:  # fo very few particles I want to see them all!
                    nParticlesNeeded = len(pids)
                else:
                    try:
                        nParticlesNeeded = np.where(thisDat.sel(fpid=pids).roi.sel(ROI_elements=[
                                                    "w", "h"]).prod("ROI_elements").cumsum("fpid")/containerSize > 1)[0][0] + 1
                    except IndexError:
                        nParticlesNeeded = nPids

                pids = np.sort(pids[:nParticlesNeeded])
                print(tt, "/", nTiles, t1, t2, nParticlesNeeded, 'of', nPids)
                particlesPloted += nParticlesNeeded
                ims = []

                for fname, pid in pids:

                    basenameImg = fname.split('/')[-1]

                    if not readParticlesFromFiles:

                        basename = '_'.join(fname.split(
                            '/')[-1].split('.')[-2].split('_')[3:])
                        thisfname_lv0 = fname_lv0.format(root=root, computer=computer, visssGen=visssGen, camera=camera, timestamp=timestamp,
                                                         site=site, year=year, month=month, day=day, nThread='{thread}', basename=basename, movieExtension=movieExtension,)

                        if thisfname_lv0 not in videos.keys():
                            for k in videos.keys():
                                videos[k].release()

                            videos[thisfname_lv0] = VideoReaderMeta(
                                thisfname_lv0, fname)
                #             print('opened %s'%thisfname_lv0)

                    particle = thisDat.sel(fpid=(fname, pid))
                    kk = int(particle.record_id.values)
                    if not readParticlesFromFiles:
                        _, frame1, _ = videos[thisfname_lv0].getFrameByIndex(
                            kk)

                        if frame1 is None:
                            continue

                        x, y, w, h = particle.roi.values.astype(int)
                        if len(frame1.shape) == 3:
                            frame1 = frame1[:, :, 0]
                        im = frame1[y+height_offset:y+height_offset+h, x:x+w]
                    else:

                        fn = files.FilenamesFromLevel(fname, config)

                        pidStr = '%07i' % pid
                        imName = '%s.png' % (pidStr)
                        imfname = '%s/%s' % (
                            fn.fnameLevel2images.format(ppid=pidStr[:4]), imName)
                        try:
                            im = np.array(Image.open(imfname))
                        except FileNotFoundError:
                            print("NOT FOUND ", imfname)
                            continue

                    im = np.pad(im, [(0, 1), (0, 1)])
                    fid = np.where(fname == np.array(ff.fnames1))[0][0]

                    text = np.zeros((100, 100))
                    text = cv2.putText(text, '%i.%i' % (fid, pid),
                                       (0, 50), cv2.FONT_HERSHEY_PLAIN, .75, 255, 1,)

                    text = crop(text)

                    y1, x1 = im.shape
                    y2, x2 = text.shape

                    # only add label if large enough
                    if x1 >= x2:

                        y3 = y1+y2
                        x3 = max(x1, x2)
                        imT = np.zeros((y3, x3), dtype=np.uint8)
                        imT[:y1, :x1] = im
                        imT[y1:, :x2] = text
                    else:
                        imT = im
                    ims.append(imT)
                    totalArea += np.prod(imT.shape)

                # make tile
                images = [Image.fromarray(im) for im in ims]
                if len(images) == 0:
                    continue
                mosaic = Packer_patched(images).pack(
                    container_width=container_width, container_height_max=container_height_max)
                mosaic = np.array(mosaic)

                if container_width > mosaic.shape[1]:
                    mosaic = np.pad(
                        mosaic, [(0, 0), (0, container_width-mosaic.shape[1]), (0, 0)])

                # sometimes container is too large...
                mosaic = mosaic[:container_height_max, :container_width]

                label = Image.fromarray(
                    np.ones((20, mosaic.shape[1], 3), dtype=np.uint8) * 255)
                drawL = ImageDraw.Draw(label)
                drawL.text((0, 0), '%s-%s' % (str(t1).split('.')[0].split('T')[1], str(
                    t2).split('.')[0].split('T')[1]), (0, 0, 0), font=fontL)

                mosaic = Image.fromarray(np.vstack((label, mosaic)))
    #             display(mosaic)
                mosaics.append(mosaic)

            nCols = nTiles//nRows

            widths, heights = zip(*(i.size for i in mosaics))

            for nRow in range(nRows):
                x_offset = 0

                for im in mosaics[nCols*(nRow):nCols*(nRow+1)]:
                    new_im.paste(im, (x_offset, max(heights)*nRow + 50))
                    x_offset += im.size[0] + extra

                # x_offset = 0
    # for im in mosaics[len(mosaics)//nRows:]:
    #   new_im.paste(im, (x_offset,max(heights) +50))
    #   x_offset += im.size[0] + extra

            nParticles = len(limDat.fpid)

    tenmm = 1e6/config["resolution"]/100

    title = '%s-%s-%s %s %s, size threshold for plotting: %i px (%.2f mm), %i of %i larger detections plotted, 10 mm = %.1f px =' % (
        ff.year, ff.month, ff.day, nicerNames(camera), config["name"], minSize, minSize * config["resolution"] * 1e-6 * 1000, particlesPloted, nParticles, tenmm)

    # new_im = cv2.putText(np.array(new_im), title,
    #                      (0, 45), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS,)
    # (label_width, label_height), baseline = cv2.getTextSize(
    #     title, FONT, FONT_SCALE, FONT_THICKNESS)

    draw = ImageDraw.Draw(new_im)
    draw.text((0, 0), title, (0, 0, 0), font=font)
    width, height = draw.textsize(title, font=font)

    draw.line((width + 15, 30, width +
               15+round(tenmm), 30), fill=0, width=5)

    new_im.save(ff.quicklook1)

    return ff.quicklook1, new_im


class Packer_patched(packer.Packer):
    """
    patched image_packer routine that works without files
    """

    def __init__(self, images):
        # Ensure plugins are fully loaded so that Image.EXTENSION is populated.
        Image.init()

        self._uid_to_filepath = dict()
        self._pieces = list()
        self._has_alpha = False

        for im in images:
            width = im.width
            height = im.height
            uid = uuid.uuid4()
            self._uid_to_filepath[uid] = deepcopy(im)
            self._pieces.append(packer.blf.Piece(
                uid=uid, size=packer.blf.Size(width, height)))
            if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                self._has_alpha = True

    def pack(self, container_width, options=None, container_height_max=100):
        '''Packs multiple images of different sizes or formats into one image.
        Args:
            container_width (int):
            options (dict):
        '''
        if options is None:
            options = self._DEFAULT_OPTIONS
        else:
            options = {
                key: options[key] if key in options else self._DEFAULT_OPTIONS[key]
                for key in self._DEFAULT_OPTIONS.keys()
            }

        margin_ = options['margin']
        assert isinstance(margin_, tuple) and len(margin_) == 4

        if options['enable_vertical_flip']:
            margin = packer.blf.Thickness(top=margin_[2], right=margin_[
                                          1], bottom=margin_[0], left=margin_[3])
        else:
            margin = packer.blf.Thickness(top=margin_[0], right=margin_[
                                          1], bottom=margin_[2], left=margin_[3])

        blf_options = {
            'margin': margin,
            'collapse_margin': options['collapse_margin'],
            'enable_auto_size': options['enable_auto_size'],
            'force_pow2': options['force_pow2']
        }

        container_width, container_height, regions = packer.blf_solver.solve(
            pieces=self._pieces,
            container_width=container_width,
            options=blf_options
        )

        compImage = self._save_image(
            container_width=container_width,
            container_height=container_height_max,
            regions=regions,
            options=options
        )
        return compImage

    def _save_image(
        self,
        container_width,
        container_height,
        regions,
        options
    ):
        bg_color_ = options['bg_color']
        assert isinstance(bg_color_, tuple) and (3 <= len(bg_color_) <= 4)
        bg_color = tuple(int(channel * 255.0) for channel in bg_color_)
        if len(bg_color) == 3:
            bg_color += (255,)

        if self._has_alpha:
            blank_image = Image.new(
                mode='RGBA',
                size=(container_width, container_height),
                color=bg_color
            )
        else:
            blank_image = Image.new(
                mode='RGB',
                size=(container_width, container_height),
                color=bg_color[0:3]
            )

        enable_vertical_flip = options['enable_vertical_flip']

        for region in regions:
            x = region.left
            if enable_vertical_flip:
                y = region.bottom
            else:
                y = container_height - region.top

            im = self._uid_to_filepath.get(region.uid)

            blank_image.paste(im=im, box=(x, y))

        return blank_image


def createLv3CoefQuicklook(case, config, lv3Version, skipExisting=True):
    '''
    Quicklooks of the coefficients obtained for matching in level3
    '''

    if type(config) is str:
        config = readSettings(config)

    leader = config["leader"]
    follower = config["follower"]

    fn = files.FindFiles(case, leader, config, lv3Version)
    ff = files.FindFiles(case, follower, config, lv3Version)

    if os.path.isfile(fn.quicklook3Coef) and skipExisting:
        print(f"skip {case} ")
        return None, None

    if len(fn.fnames3Coef) == 0:
        print(f"no data {case} {fn.fnamesPattern3Coef}")
        return None, None

    print(f"running {case}")

    dat3 = xr.open_mfdataset(fn.fnames3Coef, concat_dim="file_starttime")

    fig, (bx1, bx2, bx3, bx4) = plt.subplots(
        figsize=(10, 10), nrows=4, sharex=True)

    dat3Stats = dat3.isel(iteration=1)

    bx1.plot(dat3Stats.file_firsttime,
             dat3Stats.usedSamples, marker=".", ls="None")
    bx1.set_yscale("log")
    bx1.set_ylabel("Used matches  [#]")

    bx2.fill_between(dat3Stats.file_firsttime, (dat3Stats.muH + dat3Stats.sigmaH),
                     (dat3Stats.muH - dat3Stats.sigmaH), facecolor='C1', alpha=0.4)
    bx2.plot(dat3Stats.file_firsttime, dat3Stats.muH,
             marker=".", ls="None", color="C1")
    bx2.set_ylabel("Height difference [px]")
    bx2.axhline(0, color="gray")

    bx3.fill_between(dat3Stats.file_firsttime, (dat3Stats.muY + dat3Stats.sigmaY),
                     (dat3Stats.muY - dat3Stats.sigmaY), facecolor='C2', alpha=0.4)
    bx3.plot(dat3Stats.file_firsttime, dat3Stats.muY,
             marker=".", ls="None", color="C2")
    bx3.set_ylabel("Y position difference [px]")

    bx4.fill_between(dat3Stats.file_firsttime, (dat3Stats.muT + dat3Stats.sigmaT)
                     * 1000, (dat3Stats.muT - dat3Stats.sigmaT)*1000, facecolor='C3', alpha=0.4)
    bx4.plot(dat3Stats.file_firsttime, dat3Stats.muT *
             1000, marker=".", ls="None", color="C3")
    bx4.set_ylabel("Time difference [ms]")
    # bx4.axhline(1/config["fps"], color= "gray")
    # bx4.axhline(-1/config["fps"], color= "gray")
    bx4.axhline(0, color="gray")


    tStart = dat3Stats.file_firsttime.to_pandas().index[0].replace(hour=0, minute=0, second=0)
    tEnd = dat3Stats.file_firsttime.to_pandas().index[-1].replace(hour=23, minute=59, second=59)
    
    if config.site == "mosaic":
        software_starttimes = xr.open_dataset(f"{fn.logpath}/{fn.computer}_{config.visssGen}_{fn.camera}_softwareStarttimes.nc").software_starttimes
        tt = (software_starttimes >= tStart) & (software_starttimes <= tEnd)
        software_starttimes = software_starttimes.isel(software_starttimes=tt)

        for software_starttime in software_starttimes:
            for bx in [bx1,bx2,bx3,bx4]:
                bx.axvline(software_starttime.values, color="k", ls="-")
                
                
        software_starttimes = xr.open_dataset(f"{ff.logpath}/{ff.computer}_{config.visssGen}_{ff.camera}_softwareStarttimes.nc").software_starttimes
        tt = (software_starttimes >= tStart) & (software_starttimes <= tEnd)
        software_starttimes = software_starttimes.isel(software_starttimes=tt)

        for software_starttime in software_starttimes:
            for bx in [bx1,bx2,bx3,bx4]:
                bx.axvline(software_starttime.values, color="k", ls=":")

    bx1.grid(True)
    bx2.grid(True)
    bx3.grid(True)
    bx4.grid(True)


    bx4.set_xlim(
        tStart, 
        tEnd,
    )

    fig.suptitle(f"{fn.year}-{fn.month}-{fn.day}")
    fig.tight_layout()

    fn.createQuicklookDirs()
    print(fn.quicklook3Coef)
    fig.savefig(fn.quicklook3Coef)
    return fn.quicklook3Coef, fig

