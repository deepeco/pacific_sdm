import os
import subprocess


try:
    import gdal
except:
    from osgeo import gdal


SOURCE_PATH = '../wcdata/current/test_bio'
OUTPUT_PATH = '../wcdata/output'


GEOTIFF_PATTERN = '.bil'


FILEPAT = ''
TRIMMED_PAT = 'trim_'


# ------------ Crop settings  -------------
LAT_MIN = 20
LAT_MAX = 70
LON_MIN = 100
LON_MAX = 170.0
# -----------------------------------------


def update_coord(c, res):
    _ = float(c) / abs(res)
    return c - (_ - int(_)) * abs(res)

for dir_, dirnames, filenames in os.walk(SOURCE_PATH, followlinks=True):
    for filename in filenames:
        print(f"Processing {filename}...")
        command = ['gdalwarp', '-te']
        if filename.endswith(GEOTIFF_PATTERN):
            geotiff = os.path.join(dir_, filename)
            gfile = gdal.Open(geotiff)
            ulx, xres, xskew, uly, yskew, yres = gfile.GetGeoTransform()
            latmin = update_coord(LAT_MIN, yres)
            latmax = update_coord(LAT_MAX, yres)
            lonmin = update_coord(LON_MIN, xres)
            lonmax = update_coord(LON_MAX, xres)
            print("Updating bbox coords: ", ' '.join(map(str, [latmin,
                                                               latmax,
                                                               lonmin,
                                                               lonmax])))
            del gfile
            command.extend(map(str, [lonmin,latmin,lonmax,latmax, geotiff]))
            #destination = os.path.join(OUTPUT_PATH, geotiff[:-4] + '.tif')
            destination = geotiff[:-4]+'_' + '.tif'
            #if os.path.exists(destination):
                #try:
                   # os.remove(destination)
                #except:
                 #   pass
            command.append(destination)
            print("Executing the command: ", ' '.join(command), '\n')
            p = subprocess.Popen(command)
            p.wait()
            







