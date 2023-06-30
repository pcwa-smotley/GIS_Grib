import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.basemap import Basemap
from osgeo import osr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
from pyproj import Proj
from shapely.geometry import Point, Polygon
import pyproj
import shapely.ops as ops
from functools import partial
from PIL import Image, ImageFont, ImageDraw
import datetime
import platform
from urllib.request import urlopen

import os, sys
import numpy as np
import pytz

import time
#import seaborn as sb
from openpyxl import load_workbook
import pandas as pd
import fiona
import ogr
import argparse
import SNODAS_Download as sd

class Grib():
    '''
    class object for holding the grib data
    '''

    def __init__(self):
        self.model = ""
        self.date = np.array([])
        self.basin = ""
        self.basinArea = ""
        self.level = ""
        self.acc = "" #0-3 hr accumulation or 1hr forecast
        self.gribAll = ""
        self.units = ""
        self.ptVal = []
        self.displayunits = ""
        self.data = np.array([])
        self.elevation_data = np.array([])
        self.basinTotal = np.array([])
        self.basinSWE = np.array([])
        self.SMUDbasinTotal = np.array([])
        self.SMUDbasinSWE = np.array([])
        self.bbox = []

def handle_args(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                    nargs="*",
                    default='snodas')

    parser.add_argument('date',
                    nargs="*",
                    default=time.strftime("%Y%m%d"))  # This will be today's date in yyyymmdd format)

    parser.add_argument('-d', '--date2',
                        dest='date2', required=False,
                        default=None)  # This will be today's date in yyyymmdd format)

    parser.add_argument('-b','--basin',
                        dest='basin', default='French_Meadows',
                        required=False,
                        help='The basin to calculate Total SWE. Options inlude:\n'
                             'Hell_Hole, French_Meadows, or MFP')

    parser.add_argument('-u', '--displayunits',
                        dest='displayunits', default='US',
                        required=False,
                        help='Show Units In US or SI')

    parser.add_argument('-l', '--level',
                        dest='level', default='0-SFC',
                        required=False,
                        help='Variable level EX: -l 0-SFC ')

    parser.add_argument('-m', '--map',
                        dest='map', default=True, required=False,
                        help='use this option if you want to output a map')

    parser.add_argument('-p', '--plot',
                        dest='plot', default=True, required=False,
                        help='use this option if you want to output a plot')

    args = parser.parse_args()

    return args

def main():
    global inputArgs, grib, dir_path                          #Make our global vars: grib is the object that will hold our Grib Class.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    comparison_days =[0,-7]
    inputArgs = handle_args(sys.argv)               #All input arguments if run on the command line.
    for deltaDay in comparison_days:
        # MAKE SURE TO UNCOMMENT #inputArgs.date2 when putting back into production
        if deltaDay == 0:
            date2 = None
        else:
            date2 = ((datetime.datetime.now(pytz.timezone('US/Pacific'))) + datetime.timedelta(days=deltaDay)).strftime(
            "%Y%m%d")

        ##############
        # Debugging
        inputArgs.date = (datetime.datetime.today() - datetime.timedelta(days=1)).strftime("%Y%m%d")
        #inputArgs.date = time.strftime("%Y%m%d")
        #inputArgs.date = str(dayNum)
        inputArgs.date2 = date2 #Comment this out for just one date
        inputArgs.map = True  # Make the map and save png to folder.
        inputArgs.plot = False
        findValueAtPoint = False  # Find all the values at specific lat/lng points within an excel file.
        #################
        grib = Grib()                                   #Assign variable to the Grib Class.
        grib.model = inputArgs.model                    #Our model will always be "snodas" for this program
        grib.displayunits = inputArgs.displayunits
        grib.basin = inputArgs.basin                    # Basin can be "French_Meadows", "Hell_Hole", or "MFP", this gets shapefile

        # Bounding box will clip the raster to focus in on a region of interest (e.g. CA) This makes the raster MUCH smaller
        # and easier to work with. See gdal.Open -> gdal.Translate below for where this is acutally used.
        grib.bbox = [-125.0,50.0,-115.0,30.0]           #[upper left lon, upper left lat, lower left lon, lower left lat]
        grib = get_snowdas(grib, inputArgs.date, 'ASO_American_2023Jan31-Feb1_snowdepth_3m.tif')                        #Get the snodas file and save data into the object variable grib
        #pngFile = makePNG()
        #Any reprojections of grib.gribAll have already been done in get_snowdas.
        #The original projection of snodas is EPSG:4326 (lat/lng), so it has been changed to EPSG:3875 (x/y) in get_snodas
        projInfo = grib.gribAll.GetProjection()

        geoinformation = grib.gribAll.GetGeoTransform() #Get the geoinformation from the grib file.

        xres = geoinformation[1]
        yres = geoinformation[5]
        xmin = geoinformation[0]
        xmax = geoinformation[0] + (xres * grib.gribAll.RasterXSize)
        ymin = geoinformation[3] + (yres * grib.gribAll.RasterYSize)
        ymax = geoinformation[3]

        spatialRef = osr.SpatialReference()
        spatialRef.ImportFromWkt(projInfo)
        spatialRefProj = spatialRef.ExportToProj4()

        # create a grid of xy (or lat/lng) coordinates in the original projection
        xy_source = np.mgrid[xmin:xmax:xres, ymax:ymin:yres]
        xx, yy = xy_source

        # A numpy grid of all the x/y into lat/lng
        # This will convert your projection to lat/lng (it's this simple).
        lons, lats = Proj(spatialRefProj)(xx, yy, inverse=True)


        # Find the center point of each grid box.
        # This says move over 1/2 a grid box in the x direction and move down (since yres is -) in the
        # y direction. Also, the +yres (remember, yres is -) is saying the starting point of this array will
        # trim off the y direction by one row (since it's shifted off the grid)
        xy_source_centerPt = np.mgrid[xmin + (xres / 2):xmax:xres, ymax + (yres / 2):ymin:yres]
        xxC, yyC = xy_source_centerPt

        #lons_centerPt, lats_centerPt = Proj(spatialRefProj)(xxC, yyC, inverse=True)

        if grib.basin != "Hell_Hole":
            mask = createMask(xxC, yyC, spatialRefProj)
            grib.basinTotal, grib.basinSWE = calculateBasin(mask, grib, xres, yres)

        # The shape file for the Hell Hole basin includes the SMUD domain. Therefore, if we want to extract the SMUD
        # domain, then we will create another mask on top of the Hell Hole mask. This means that any grid point
        # outside of both domains will still = 0. The SMUD domain will contain it's mask AND the Hell Hole mask.
        if grib.basin == 'Hell_Hole':
            grib.basin = 'Hell_Hole_SMUD' #This is just to get the correct directory structure
            submask = createMask(xxC, yyC, spatialRefProj) # All areas in array = to 1 will be in SMUD's basin
            smud_BasinArea = grib.basinArea # Used to remove SMUD's basin area (in m^2) from Hell_Hole's basin.
            # Get SMUD's information for the SMUD submask
            grib.SMUDbasinTotal, grib.SMUDbasinSWE = calculateBasin(submask, grib, xres, yres)

            grib.basin = 'Hell_Hole'  # reset back
            mask = createMask(xxC, yyC, spatialRefProj) #This will be the entire Hell Hole basin, which includes SMUD
            hhMask = mask + submask  # Hell hole basin is now anywhere where hhMask = 1 and SMUD is anywhere it = 2
            hhMask[hhMask != 1] = 0 # Set anything outside of Hell Hole's mask = 0
            grib.basinArea = grib.basinArea - smud_BasinArea # grib.basinArea currently includes smuds basin, so remove it.

            # Get HellHoles's information from the hell hole submask
            grib.basinTotal, grib.basinSWE = calculateBasin(hhMask, grib, xres, yres)
            print("Current Basin Total: " + str(grib.basinTotal) + " SMUD Total: " + str(grib.SMUDbasinTotal))
            #grib.basinTotal = grib.basinTotal - (0.92 * smudBasinTotal[0])
            mask = hhMask # Need to do this so we can use the correct mask in compareDates and in calculateByElevation

        # # Calculate the difference between two rasters
        # if inputArgs.date2 != None:
        #     grib.basinTotal, grib.basinSWE = compareDates(mask, grib, xres, yres)
        #
        # #Need to do this after Hell_Hole's data has been manipulated (to account for SMUD)
        # elevation_bins = calculateByElevation(mask, grib, xres, yres)
        #
        # #Send data for writing to Excel File
        # if deltaDay == 0:
        #     excel_output(elevation_bins)

        # if inputArgs.plot == True:
        #     makePlot(elevation_bins, deltaDay)
        # print(elevation_bins)

        print(inputArgs.date," Basin Total: ", grib.basinTotal)

        #findValue will return a dataframe with SWE values at various lat/lng points.
        df_ptVal = None
        # if findValueAtPoint == True:
        #     df_ptVal = findPointValue(spatialRefProj, xy_source)

        if inputArgs.map == True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            m = Basemap(llcrnrlon=-122.8, llcrnrlat=37.3,
                        urcrnrlon=-119.0, urcrnrlat=40.3, ax=ax)

            #m.arcgisimage(service='World_Imagery', xpixels=2000, verbose=True)
            im = Image.open(urlopen("http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-122.8,37.3,-119.0,40.3&bboxSR=4326&imageSR=4326&size=2000,1578&dpi=96&format=png32&transparent=true&f=image"))
            m.imshow(im, origin='upper')

            #For inset
            # loc =>'upper right': 1,
            # 'upper left': 2,
            # 'lower left': 3,
            # 'lower right': 4,
            # 'right': 5,
            # 'center left': 6,
            # 'center right': 7,
            # 'lower center': 8,
            # 'upper center': 9,
            # 'center': 10
            axin = inset_axes(m.ax, width="40%", height="40%", loc=8)

            m2 = Basemap(llcrnrlon=-120.7, llcrnrlat=38.7,
                         urcrnrlon=-120.1, urcrnrlat=39.3, ax=axin)

            im2 = Image.open(urlopen("http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox=-120.7,38.7,-120.09999999999998,39.3&bboxSR=4326&imageSR=4326&size=2000,1999&dpi=96&format=png32&transparent=true&f=image"))
            m2.imshow(im2, origin='upper')

            #m2.arcgisimage(service='World_Imagery', xpixels=2000, verbose=True)
            mark_inset(ax, axin, loc1=2, loc2=4, fc="none", ec="0.5")

            ###################################DEBUGGING AREA###############################################################
            # Debugging: Test to prove a given lat/lng pair is accessing the correct grid box:

            #*********TEST 1: Test for center points
            #grib.data[0,0] = 15 #increase the variable by some arbitrary amount so it stands out.
            #xpts, ypts = m(lons_centerPt[0,0],lats_centerPt[0,0]) #This should be in the dead center of grid[0,0]
            #m.plot(xpts,ypts, 'ro')

            #*********TEST 2: Test for first grid box
            # Test to see a if the point at [x,y] is in the upper right corner of the cell (it better be!)
            #xpts, ypts = m(lons[0, 0], lats[0, 0])  # should be in upper right corner of cell
            #m.plot(xpts, ypts, 'bo')

            # *********TEST 3: Test for first grid box
            # Test to see the location of center points of each grid in polygon
            # To make this work, uncomment the variables in def create_mask
            #debug_Xpoly_center_pts, debug_Ypoly_center_pts = m(debugCenterX, debugCenterY)
            #m.plot(debug_Xpoly_center_pts, debug_Ypoly_center_pts, 'bo')

            # *********TEST 4: Test grid box size (In lat lng coords)
            # This is for use in a Basemap projection with lat/lon (e.g. EPSG:4326)
            #testX = np.array([[-120.1, -120.1], [-120.10833, -120.10833]])
            #testY = np.array([[39.0, 39.00833], [39.0, 39.00833]])
            # testVal = np.array([[4,4],[4,4]])

            # For use in basemap projection with x/y (e.g. espg:3857. In m=basemap just include the argument projection='merc')
            # testX = np.array([[500975, 500975], [(500975 + 1172), (500975 + 1172)]])
            # testY = np.array([[502363, (502363 + 1172)], [502363, (502363 + 1172)]])
            #testVal = np.array([[18, 18], [18, 18]])
            #im1 = m.pcolormesh(testX, testY, testVal, cmap=plt.cm.jet, vmin=0.1, vmax=10, latlon=False, alpha=0.5)

            # Test to see all points
            # xtest, ytest = m(lons,lats)
            # m.plot(xtest,ytest, 'bo')
            ################################################################################################################

            hr = 0
            makeMap(lons, lats, hr, m, m2,df_ptVal, deltaDay)
    return

def get_snowdas(gribObj,date,ASO_File):
    gribObj.gribAll = gdal.Open(ASO_File, GA_ReadOnly)
    # baseFolder = os.path.join(dir_path,'grib_files')
    # snowdas_dir = os.path.join(baseFolder,date,grib.model)
    # fyear, fmonth, fday = '1970', '1', '31'  # Filling with random data
    # #if not os.path.exists(snowdas_dir):
    # #    sd.main('snodas',date)
    # #if os.path.exists and os.listdir(snowdas_dir)==[]: #it's empty
    # #    sd.main('snodas', date)
    # sd.main('snodas', date) # Download no matter what..
    #
    # # After 2019 the extension names for the description files were changed from 'Hdr' to 'txt'
    # extension = 'Hdr'
    # if datetime.datetime.strptime(date, '%Y%m%d') >= datetime.datetime(2019, 7, 1):
    #     extension = 'txt'
    # for file in os.listdir(snowdas_dir):
    #     if file.endswith(extension):
    #         gribObj.gribAll = gdal.Open(ASO_File, GA_ReadOnly)
    #         #<--Extract Date Info-->
    #         if platform.system() == "Windows":
    #             f = open(snowdas_dir + './' + file, 'r')
    #         else:
    #             f = open(os.path.join(snowdas_dir,file), 'r')
    #         ' The .txt file contains a bunch of dates, the dates labeled "Start ___" is the valid time stamp for the file'
    #         year_str = 'Start year'
    #         month_str = 'Start month'
    #         day_str = 'Start day'
    #         for line in f:
    #             if year_str in line:
    #                 fyear = line[-5:]  # get last 5 charaters (one is a space)
    #             if month_str in line:
    #                 fmonth = line[-3:]
    #             if day_str in line:
    #                 fday = line[-3:]
    #         #<--Done with Date Info-->
    #         f.close()
    #         #Check to see if we are comparing two dates, if we are and this is the second date, grib.date will have a value
    #         if gribObj.date != None:
    #             gribObj.date2 = datetime.datetime(year=int(fyear), month=int(fmonth),day=int(fday))  # This will be the date in yyyymmdd format
    #         else:
    #             gribObj.date =  datetime.datetime(year=int(fyear), month=int(fmonth),day=int(fday))  # This will be the date in yyyymmdd format
    #Notes: This next section is important because:
    #       1) We are quickly reducing the size of raster using the "projWin=" parameter.
    #       2) We are transforming the SNODAS grid from latlon (EPSG:4326) to XY (EPSG:3857)
    # We MUST transform this into XY coordinates because we are making calculations on the rasters based off of
    # meters (not decimal degrees). For example, once we use gdal.Warp, it will reproject it into xy coordinates, which
    # we can then use to find the x,y resolution of each grid box in meters. Now, we will know that for every raster
    # grid cell, we can calculate any type of Volume calculation we need, like 100 mm of rainfall over a 1,000 x 1,000 m
    # grid will give us ~8 acre feet.
    #Clip the raster to a given bounding box (makes the raster much easier to work with)
    #gribObj.gribAll = gdal.Translate('/vsimem/temp.vrt', gribObj.gribAll, projWin=gribObj.bbox)
    projInfo = grib.gribAll.GetProjection()
    try:
        spatialRef = osr.SpatialReference(wkt=projInfo)
        cord_sys = spatialRef.GetAttrValue("GEOGCS|AUTHORITY",1)
    except:
        print ("NO COORDINATE SYSTEM FOUND! Transforming to XY")
        cord_sys = '4326'
    if cord_sys != '3875':
        gribObj.gribAll = gdal.Warp('/vsimem/temp.vrt', gribObj.gribAll,dstSRS='EPSG:3857')  # If you wanted to put it into x/y coords
        print("Successfully Reprojected Coordinate System From Lat/Lng to X/Y")
    band = gribObj.gribAll.GetRasterBand(1)
    data = BandReadAsArray(band)
    data[data == -9999] = 0
    gribObj.data = data
    gribObj.units = '[kg/(m^2)]' #the units are in mm, or kg/m^2
    return gribObj

def createMask(xxC,yyC,spatialRefProj):

    # Get the lat / lon of each grid box's center point.
    lons_centerPt, lats_centerPt = Proj(spatialRefProj)(xxC, yyC, inverse=True)

    # given a bunch of lat/lons in the geotiff, we can get the polygon points.
    sf = fiona.open(dir_path+'/Shapefiles/'+grib.basin+'/'+grib.basin+'.geojson')
    geoms = [feature["geometry"] for feature in sf]
    poly = Polygon(geoms[0]['coordinates'][0])  # For a simple square, this would be 4 points, but it can be thousands of points for other objects.
    grib.basinArea = ops.transform(
    partial(
    pyproj.transform,
    pyproj.Proj(init='EPSG:4326'),
    pyproj.Proj(
        proj='aea',
        lat_1=poly.bounds[1],
        lat_2=poly.bounds[3])),
    poly).area
    # NOTE: Calculating area in the mercator projection (ESPG:3587) will lead to large errors due to inaccuracies in
    # that projections distance measurements. Instead, you must convert over to 4326 or equal area coordinates (whci is best)
    # For example, if instead of proj='aea' (in code below) you used proj = 'merc' the area for French Meadows
    # increases from roughly . 30,000 acres to over 50,000 acres. Alternatively, you could calculate the area by
    # counting the grid boxes in the polygon and summing up the area of one grid where
    # xres = 0.008333 deg * 111,111 m and yres = 0.008333 deg * 111,111 m * cos(latitude of grid box).
    # The method below is the easiest way to do this.

    polyMinLon, polyMinLat, polyMaxLon, polyMaxLat = sf.bounds  # Get the bounds to speed up the process of generating a mask.

    # create a 1D numpy mask filled with zeros that is the exact same size as the lat/lon array from our projection.
    mask = np.zeros(lons_centerPt.flatten().shape)

    # Debugging: FOR CENTER POINT OF GRID BOX
    # These are test variables to see where the center points are (plot with basemaps to prove to yourself they're in the right spot).
    global debugCenterX, debugCenterY
    debugCenterX = np.zeros(lats_centerPt.flatten().shape)
    debugCenterY = np.zeros(lons_centerPt.flatten().shape)

    # Create Mask by checking whether points in the raster are within the bounds of the polygon. Instead of checking
    # every single point in the raster, just focus on points within the max/min bounds of the polygon (it's slow as hell
    # if you don't do that).
    i = 0  # counter
    for xp, yp in zip(lons_centerPt.flatten(), lats_centerPt.flatten()):
        if ((polyMinLon <= xp <= polyMaxLon) and (polyMinLat <= yp <= polyMaxLat)):
            mask[i] = (Point(xp, yp).within(poly))
            # Debugging FOR CENTER POINT OF GRID BOX: If you want to visualize the center point
            #       of each grid box that's found in the polygon,
            #       include this below and then you can put a dot (via m.plot)
            if (Point(xp, yp).within(poly)):
                debugCenterX[i] = xp
                debugCenterY[i] = yp
        i += 1

    mask = np.reshape(mask, (xxC.shape))
    return mask

def calculateBasin(mask, gribObj, xres, yres):
    basinTotal = 0
    basinSWE_in = 0
    if gribObj.units == '[kg/(m^2)]':
        value_mm = gribObj.data.copy()
        value_m = value_mm * 0.001  # mm to m.
        value_inches = gribObj.data.copy() * 0.03937  # mm to inches.
        #value_Cubed = totalPrecip_mm * abs(xres) * abs(yres)  # change each grid box to total precip in cubic meters
        #value_AF = (value_m * abs(xres) * abs(yres))/ 1233.48  # 1 AF = 1233.48 m^3
        basinSWE_in = np.sum(mask*value_inches.T) / np.sum(mask) # Taking an average (SUM of swe / total grid boxes)
        basinTotal = np.sum(mask * value_m.T)*gribObj.basinArea / (1233.48 * np.sum(mask))#1 AF = 1233.48 m^3
    return basinTotal, basinSWE_in

def makeMap(lons,lats,hr,m,m2,df,deltaDay):
    output_dir = os.path.join(dir_path, 'Images', grib.date.strftime("%Y%m%d"))
    imgtype = None
    if imgtype == 'cumulative':
        raster = sum(grib.data[0:grib.hours.index(hr)], axis=0) #cumulative
    else:
        raster = grib.data  # 1 hr forecast (not cumulative)

    if grib.displayunits == 'US' and grib.units == '[kg/(m^2)]':
        raster = raster * 0.03937
        grib.units = 'inches'

    #YOU CAN NOT PUT NAN VAULES IN BEFORE DOING scipy.ndimage.zoom
    raster[raster == 0] = np.nan #this will prevent values of zero from being plotted.

    maxVal = int(np.nanpercentile(raster, 99,  interpolation='linear'))
    minVal = int(np.nanpercentile(raster, 1,  interpolation='linear'))

    im = m.pcolormesh(lons, lats, raster.T, cmap=plt.cm.jet, vmin=minVal, vmax=maxVal) # return 50th percentile, e.g median., latlon=True)
    im2 = m2.pcolormesh(lons, lats, raster.T, cmap=plt.cm.jet, vmin=minVal,vmax=maxVal)  # return 50th percentile, e.g median., latlon=True)
    cb = m.colorbar(mappable=im, location='right', label='SWE (in.)')
    #Show user defined points on map.
    if df != None:
        for index, row in df.iterrows():
            m.plot(row['Longitude'], row['Latitude'], 'ro')
            plt.text(row['Longitude'], row['Latitude'],str(round(row['Model_Value']* 0.03937,1))+' / ' + str(row['SWE']))
            print("Modeled Value: " + str(round(row['Model_Value']* 0.03937,1))+' / Actual Value: ' + str(row['SWE']))


    #plot shapefile
    m.readshapefile(dir_path + '/Shapefiles/' + grib.basin + '/' + grib.basin + '_EPSG4326',
                    grib.basin + '_EPSG4326', linewidth=1)
    m2.readshapefile(dir_path+'/Shapefiles/'+grib.basin+'/'+grib.basin+'_EPSG4326',
                     grib.basin+'_EPSG4326', linewidth=1)
    if grib.basin == 'Hell_Hole':
        m.readshapefile(dir_path + '/Shapefiles/Hell_Hole_SMUD/Hell_Hole_SMUD' + '_EPSG4326',
                    'Hell_Hole_SMUD' + '_EPSG4326', linewidth=1)
        m2.readshapefile(dir_path + '/Shapefiles/Hell_Hole_SMUD/Hell_Hole_SMUD' + '_EPSG4326',
                     'Hell_Hole_SMUD' + '_EPSG4326', linewidth=1)

    # annotate
    m.drawcountries()
    m.drawstates()
    #m.drawrivers()
    m.drawcounties(color='darkred')
    if inputArgs.date2 != None:
        plt.suptitle(grib.basin.replace('_', " ") + ' Difference in SWE between ' + grib.date.strftime("%m/%d/%Y") +
        ' and ' + grib.date2.strftime("%m/%d/%Y") +
                '\n Total Difference in AF (calculated from SWE): ' + str("{:,}".format(int(grib.basinTotal))) + ' acre feet')
        img = Image.open(output_dir+"/"+grib.date.strftime("%Y%m%d")+"_0_"+grib.basin+'.png')
        w, h = img.size
        draw = ImageDraw.Draw(img)
        if platform.system() == 'Windows':
            font = ImageFont.truetype('micross.ttf')
        else:
            font_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images', 'fonts', 'micross.ttf')
            font = ImageFont.truetype(font_path, 120)  # Avail in C:\\Windows\Fonts

        plus_sign=''
        if grib.basinTotal > 0:
            plus_sign = "+"
        draw.text((1000,h-400),'7 Day Change from ' + grib.date2.strftime("%#m/%d") +' to ' +
                  grib.date.strftime("%#m/%d") + ': ' + plus_sign + str("{:,}".format(int(grib.basinTotal))) + ' acre feet',(0,0,0), font=font)
        img.save(output_dir+"/"+grib.date.strftime("%Y%m%d")+"_0_"+grib.basin+'.png')
    else:
        plt.suptitle(grib.basin.replace('_'," ") + ' ' + grib.date.strftime("%m/%d/%Y") +
                  '\n Total AF from SWE: ' + str("{:,}".format(int(grib.basinTotal))) +' acre feet')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir+"/"+grib.date.strftime("%Y%m%d")+"_"+str(-deltaDay)+"_"+grib.basin+'.png',dpi=775)
    print("Saved to " + dir_path+'/images/'+grib.basin+'.png')
    #plt.show()
    plt.close()

def trial():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Open the raster data using gdal
    raster_file = "ASO_American_2023Jan31-Feb1_swe_50m.tif"
    raster_data = gdal.Open(raster_file)

    # Read the first band as a numpy array
    raster_band = raster_data.GetRasterBand(1)
    raster_array = BandReadAsArray(raster_band)
    #band_data = raster_band.ReadAsArray()

    # Get the geotransform of the raster
    geo_transform = raster_data.GetGeoTransform()

    # Open the shapefile using ogr
    shape_data = ogr.Open(dir_path+'/Shapefiles/French_Meadows/French_Meadows_EPSG4326.shp')
    shape_layer = shape_data.GetLayer()

    # Get the original CRS of the shapefile
    original_crs = shape_layer.GetSpatialRef()

    # Define the target CRS (in this case, WGS 84)
    target_crs = osr.SpatialReference()
    target_crs.ImportFromEPSG(3875)

    # Create a coordinate transformation
    transform = osr.CoordinateTransformation(original_crs, target_crs)

    # Loop through each feature in the shapefile
    for feature in shape_layer:
        # Get the geometry of the feature
        geometry = feature.GetGeometryRef()

        # Transform the geometry to the target CRS
        geometry.Transform(transform)

        # Get the bounds of the transformed geometry
        minx, maxx, miny, maxy = geometry.GetEnvelope()

        # Transform the bounds to the raster's coordinate system
        transform = raster_data.GetGeoTransform()
        ulx, uly = gdal.ApplyGeoTransform(transform, minx, maxy)
        lrx, lry = gdal.ApplyGeoTransform(transform, maxx, miny)

        # Get the x and y indices of the bounds in the raster data
        xoff = int((ulx - transform[0]) / transform[1])
        yoff = int((uly - transform[3]) / transform[5])
        xcount = int((lrx - ulx) / transform[1]) + 1
        ycount = int((lry - uly) / transform[5]) + 1

        # Read the data within the bounds of the feature
        data = raster_array[yoff:yoff + ycount, xoff:xoff + xcount]

        # Sum the data within the bounds of the feature
        feature_sum = np.sum(data)
        # Do something with the sum
        print("The sum of the values in the feature is:", feature_sum)

if __name__ == "__main__":
    trial()
    main()