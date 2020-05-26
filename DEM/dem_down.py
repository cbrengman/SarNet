#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:20:45 2019

@author: Glarus
"""
import numpy as np
import wget
import os
from zipfile import ZipFile

def down_dem(num_per_region):
    #Generate arrays for land coordinates (rough estimate) of major continents
    NA_Lat = np.arange(33,50,1)
    NA_Lon = np.arange(-116,-80,1)
    
    SA_Lat = np.arange(-23,-3,1)
    SA_Lon = np.arange(-69,-43,1)
    
    EU_Lat = np.arange(47,54,1)
    EU_Lon = np.arange(8,44,1)
    
    AS_Lat = np.arange(28,60,1)
    AS_Lon = np.arange(55,114,1)
    
    AF_Lat = np.arange(-17,15,1)
    AF_Lon = np.arange(15,37,1)
    
    #Generate 100 pairs for each major continent
    NA_pairs = set([])
    SA_pairs = set([])
    EU_pairs = set([])
    AS_pairs = set([])
    AF_pairs = set([])
    for i in range(num_per_region):
        NA_pairs.add((np.random.choice(NA_Lat),np.random.choice(NA_Lon)))
        
        SA_pairs.add((np.random.choice(SA_Lat),np.random.choice(SA_Lon)))
        
        EU_pairs.add((np.random.choice(EU_Lat),np.random.choice(EU_Lon)))
        
        AS_pairs.add((np.random.choice(AS_Lat),np.random.choice(AS_Lon)))
        
        AF_pairs.add((np.random.choice(AF_Lat),np.random.choice(AF_Lon)))
        
    #Setup URL for each region
    NA_Url = 'http://dds.cr.usgs.gov/srtm/version2_1/SRTM3/North_America/'
    SA_Url = 'http://dds.cr.usgs.gov/srtm/version2_1/SRTM3/South_America/'
    EU_Url = 'http://dds.cr.usgs.gov/srtm/version2_1/SRTM3/Eurasia/'
    AS_Url = 'http://dds.cr.usgs.gov/srtm/version2_1/SRTM3/Eurasia/'
    AF_Url = 'http://dds.cr.usgs.gov/srtm/version2_1/SRTM3/Africa/'
    
    #Download zip files
    NA_fname = []
    for pair in NA_pairs:
        lat = pair[0]
        lon = pair[1]
        if lat < 0:
            if lat > -10:
                lat = 'S' + '0' + str(abs(lat))
            else:
                lat = 'S' + str(abs(lat))
        else:
            if lat < 10:
                lat = 'N' + '0' + str(abs(lat))
            else:
                lat = 'N' + str(abs(lat))
        if lon < 0:
            if lon > -100:
                if lon > -10:
                    lon = 'W' + '00' + str(abs(lon))
                else:
                    lon = 'W' + '0' + str(abs(lon))
            else:
                lon = 'W' + str(abs(lon))
        else:
            if lon < 100:
                if lon < 10:
                    lon = 'E' + '00' + str(abs(lon))
                else:
                    lon = 'E' + '0' + str(abs(lon))
            else:
                lon = 'E' + str(abs(lon))
        geturl = NA_Url + lat + lon + '.hgt.zip'
        try:
            exists = os.path.isfile('dem_zips/' + lat + lon + '.hgt.zip')
            exists1 = os.path.isfile('dem_hgts/' + lat + lon + '.hgt')
            if exists and exists1:
                print("Zip file exists and has been unzipped. Skipping")
            elif not exists and exists1:
                print("Zip file does not exist, but unzipped file does. Skipping")
            elif exists and not exists1:
                NA_fname.append(lat + lon + '.hgt.zip')
            else:
                wget.download(geturl,out='dem_zips/')
                NA_fname.append(lat + lon + '.hgt.zip')
        except:
            print("Download of " + lat+lon+".hgt.zip failed. Skipping and Moving on to next")
    
    SA_fname = []
    for pair in SA_pairs:
        lat = pair[0]
        lon = pair[1]
        if lat < 0:
            if lat > -10:
                lat = 'S' + '0' + str(abs(lat))
            else:
                lat = 'S' + str(abs(lat))
        else:
            if lat < 10:
                lat = 'N' + '0' + str(abs(lat))
            else:
                lat = 'N' + str(abs(lat))
        if lon < 0:
            if lon > -100:
                if lon > -10:
                    lon = 'W' + '00' + str(abs(lon))
                else:
                    lon = 'W' + '0' + str(abs(lon))
            else:
                lon = 'W' + str(abs(lon))
        else:
            if lon < 100:
                if lon < 10:
                    lon = 'E' + '00' + str(abs(lon))
                else:
                    lon = 'E' + '0' + str(abs(lon))
            else:
                lon = 'E' + str(abs(lon))
        geturl = SA_Url + lat + lon + '.hgt.zip'
        try:
            exists = os.path.isfile('dem_zips/' + lat + lon + '.hgt.zip')
            exists1 = os.path.isfile('dem_hgts/' + lat + lon + '.hgt')
            if exists and exists1:
                print("Zip file exists and has been unzipped. Skipping")
            elif not exists and exists1:
                print("Zip file does not exist, but unzipped file does. Skipping")
            elif exists and not exists1:
                SA_fname.append(lat + lon + '.hgt.zip')
            else:
                wget.download(geturl,out='dem_zips/')
                SA_fname.append(lat + lon + '.hgt.zip')
        except:
            print("Download of " + lat+lon+".hgt.zip failed. Skipping and Moving on to next")
        
    EU_fname = []
    for pair in EU_pairs:
        lat = pair[0]
        lon = pair[1]
        if lat < 0:
            if lat > -10:
                lat = 'S' + '0' + str(abs(lat))
            else:
                lat = 'S' + str(abs(lat))
        else:
            if lat < 10:
                lat = 'N' + '0' + str(abs(lat))
            else:
                lat = 'N' + str(abs(lat))
        if lon < 0:
            if lon > -100:
                if lon > -10:
                    lon = 'W' + '00' + str(abs(lon))
                else:
                    lon = 'W' + '0' + str(abs(lon))
            else:
                lon = 'W' + str(abs(lon))
        else:
            if lon < 100:
                if lon < 10:
                    lon = 'E' + '00' + str(abs(lon))
                else:
                    lon = 'E' + '0' + str(abs(lon))
            else:
                lon = 'E' + str(abs(lon))
        geturl = EU_Url + lat + lon + '.hgt.zip'
        try:
            exists = os.path.isfile('dem_zips/' + lat + lon + '.hgt.zip')
            exists1 = os.path.isfile('dem_hgts/' + lat + lon + '.hgt')
            if exists and exists1:
                print("Zip file exists and has been unzipped. Skipping")
            elif not exists and exists1:
                print("Zip file does not exist, but unzipped file does. Skipping")
            elif exists and not exists1:
                EU_fname.append(lat + lon + '.hgt.zip')
            else:
                wget.download(geturl,out='dem_zips/')
                EU_fname.append(lat + lon + '.hgt.zip')
        except:
            print("Download of " + lat+lon+".hgt.zip failed. Skipping and Moving on to next")
        
    AS_fname = []
    for pair in AS_pairs:
        lat = pair[0]
        lon = pair[1]
        if lat < 0:
            if lat > -10:
                lat = 'S' + '0' + str(abs(lat))
            else:
                lat = 'S' + str(abs(lat))
        else:
            if lat < 10:
                lat = 'N' + '0' + str(abs(lat))
            else:
                lat = 'N' + str(abs(lat))
        if lon < 0:
            if lon > -100:
                if lon > -10:
                    lon = 'W' + '00' + str(abs(lon))
                else:
                    lon = 'W' + '0' + str(abs(lon))
            else:
                lon = 'W' + str(abs(lon))
        else:
            if lon < 100:
                if lon < 10:
                    lon = 'E' + '00' + str(abs(lon))
                else:
                    lon = 'E' + '0' + str(abs(lon))
            else:
                lon = 'E' + str(abs(lon))
        geturl = AS_Url + lat + lon + '.hgt.zip'
        try:
            exists = os.path.isfile('dem_zips/' + lat + lon + '.hgt.zip')
            exists1 = os.path.isfile('dem_hgts/' + lat + lon + '.hgt')
            if exists and exists1:
                print("Zip file exists and has been unzipped. Skipping")
            elif not exists and exists1:
                print("Zip file does not exist, but unzipped file does. Skipping")
            elif exists and not exists1:
                AS_fname.append(lat + lon + '.hgt.zip')
            else:
                wget.download(geturl,out='dem_zips/')
                AS_fname.append(lat + lon + '.hgt.zip')
        except:
            print("Download of " + lat+lon+".hgt.zip failed. Skipping and Moving on to next")
        
    AF_fname = []
    for pair in AF_pairs:
        lat = pair[0]
        lon = pair[1]
        if lat < 0:
            if lat > -10:
                lat = 'S' + '0' + str(abs(lat))
            else:
                lat = 'S' + str(abs(lat))
        else:
            if lat < 10:
                lat = 'N' + '0' + str(abs(lat))
            else:
                lat = 'N' + str(abs(lat))
        if lon < 0:
            if lon > -100:
                if lon > -10:
                    lon = 'W' + '00' + str(abs(lon))
                else:
                    lon = 'W' + '0' + str(abs(lon))
            else:
                lon = 'W' + str(abs(lon))
        else:
            if lon < 100:
                if lon < 10:
                    lon = 'E' + '00' + str(abs(lon))
                else:
                    lon = 'E' + '0' + str(abs(lon))
            else:
                lon = 'E' + str(abs(lon))
        geturl = AF_Url + lat + lon + '.hgt.zip'
        try:
            exists = os.path.isfile('dem_zips/' + lat + lon + '.hgt.zip')
            exists1 = os.path.isfile('dem_hgts/' + lat + lon + '.hgt')
            if exists and exists1:
                print("Zip file exists and has been unzipped. Skipping")
            elif not exists and exists1:
                print("Zip file does not exist, but unzipped file does. Skipping")
            elif exists and not exists1:
                AF_fname.append(lat + lon + '.hgt.zip')
            else:
                wget.download(geturl,out='dem_zips/')
                AF_fname.append(lat + lon + '.hgt.zip')
        except:
            print("Download of " + lat+lon+".hgt.zip failed. Skipping and Moving on to next")
            
    return [NA_fname,SA_fname,EU_fname,AS_fname,AF_fname]
if __name__ == "__main__":
    
    num = 100
    
    files = down_dem(num)
    
    for region in files:
        for dem in region:
            with ZipFile('dem_zips/' + dem,'r') as zipObj:
                zipObj.extractall('dem_hgts')
                
    
        
