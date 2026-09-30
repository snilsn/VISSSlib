#!/usr/bin/env bash
# coding: utf-8

# sample script to process quicklooks after receiving data

#stop on error
set -e

echo $HOSTNAME

days=$1 # how many days to look back?
sett="${@:2}" #list of settings files to process



for SETTINGS in $sett
do
   echo $SETTINGS

   # metaEvents
   echo "python3 -m VISSSlib metadata.createEvent /projekt1//ag_maahn/VISSS_config/$SETTINGS $days --skip-existing"
   python3 -m VISSSlib metadata.createEvent /projekt1//ag_maahn/VISSS_config/$SETTINGS $days --skip-existing
done


for SETTINGS in $sett
do
   # level 0 quicklooks
   echo "python3 -m VISSSlib quicklooks.level0Quicklook /projekt1//ag_maahn/VISSS_config/$SETTINGS $days --skip-existing"
   python3 -m VISSSlib quicklooks.level0Quicklook /projekt1//ag_maahn/VISSS_config/$SETTINGS $days --skip-existing
done


for SETTINGS in $sett
do
   # metaFrames
   echo "python3 -m VISSSlib metadata.createMetaFrames /projekt1//ag_maahn/VISSS_config/$SETTINGS $days --skip-existing"
   python3 -m VISSSlib metadata.createMetaFrames /projekt1//ag_maahn/VISSS_config/$SETTINGS $days --skip-existing
   #metaFrames Quicklook
   echo "python3 -m VISSSlib quicklooks.metaFramesQuicklook /projekt1//ag_maahn/VISSS_config/$SETTINGS $days --skip-existing"
   python3 -m VISSSlib quicklooks.metaFramesQuicklook /projekt1//ag_maahn/VISSS_config/$SETTINGS $days --skip-existing
done

for SETTINGS in $sett
do
    # level1detect quicklooks, level1 detect on cluster!
    echo "python3 -m VISSSlib quicklooks.createLevel1detectQuicklook  /projekt1//ag_maahn/VISSS_config/$SETTINGS $days --skip-existing"
    python3 -m VISSSlib quicklooks.createLevel1detectQuicklook  /projekt1//ag_maahn/VISSS_config/$SETTINGS $days --skip-existing
done

for SETTINGS in $sett
do
    # level1match quicklooks, level1 match on cluster!
    echo "python3 -m VISSSlib quicklooks.createLevel1matchParticlesQuicklook  /projekt1//ag_maahn/VISSS_config/$SETTINGS $days --skip-existing"
    python3 -m VISSSlib quicklooks.createLevel1matchParticlesQuicklook  /projekt1//ag_maahn/VISSS_config/$SETTINGS $days --skip-existing
done


for SETTINGS in $sett
do
    echo "python3 -m VISSSlib tools.reportLastFiles   /projekt1//ag_maahn/VISSS_config/$SETTINGS"
    python3 -m VISSSlib tools.reportLastFiles   /projekt1//ag_maahn/VISSS_config/$SETTINGS
done
