# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 07:07:17 2020

@author: dmitry
"""
from osgeo import gdal
import numpy as np

def get_area(filename='', threshold=0.8):
    R = 6631000.0
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    rb = ds.GetRasterBand(1)
    img_array = rb.ReadAsArray()
    gt=ds.GetGeoTransform()
    dlat = gt[-1]
    dlon = gt[1]
    m, n = img_array.shape
    lats = np.linspace(gt[-3], gt[-3] + dlat * n, n)
    lons = np.linspace(gt[0], gt[0] + dlon * m, m)
    LON, LAT = np.meshgrid(lons, lats)
    return (R*np.cos(LAT[img_array>threshold]/180*np.pi)*R*np.abs(dlat/180*dlon/180*np.pi*np.pi)).sum()/10**6

filenames = ['pinus_0.tiff',
  'pinus__70cc26_RF_100_0.tiff',
  'pinus__70cc85_RF_100_0.tiff',
  'pinus__70mr26_RF_100_0.tiff',
  'pinus__70mr85_RF_100_0.tiff',
  'pinus__cclgm_RF_100_0.tiff',
  'pinus__mrlgm_RF_100_0.tiff']

computing_plan = [ (x, y) for x in filenames for y in [0.1, 0.5]]

for name, th in computing_plan:
    area =get_area(filename=name, threshold=th) 
    print(f"Area = {area} sq. km; filename={name}, threshold={th}.")
