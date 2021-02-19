import os
import time
import datetime
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

def main(basin, date, comparison_days):
    try:
        file = os.path.join('G:/Energy Marketing/Weather', 'Daily_Output.xlsx')
        df1 = pd.read_excel(file, sheet_name='French_Meadows')
        df1.set_index(pd.DatetimeIndex(df1.Date), inplace=True)
        yesterday = (datetime.datetime.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        row_today_FM = df1.loc[yesterday]
        row_lastWeek_FM = df1.loc[(datetime.datetime.today() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')]

        df2 = pd.read_excel(file, sheet_name='Hell_Hole')
        df2.set_index(pd.DatetimeIndex(df2.Date), inplace=True)

        row_today_HH = df2.loc[yesterday]
        row_lastWeek_HH = df2.loc[(datetime.datetime.today() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')]

        combined_basin_tot = int(row_today_FM['TotalAF'] + row_today_HH['TotalAF_HellHole'])
        combined_change = (int(row_today_FM['TotalAF'] + row_today_HH['TotalAF_HellHole']) -
                           int(row_lastWeek_FM['TotalAF'] + row_lastWeek_HH['TotalAF_HellHole']))
    except Exception as err:
        print("Reading Daily_Output.xlsx Failed. Error: " + str(err))
        combined_basin_tot = -99999999
        combined_change = -99999999
    date =  (datetime.datetime.today() - datetime.timedelta(days=1)).strftime("%Y%m%d")

    dir_path = os.path.dirname(os.path.realpath(__file__))

    ofile = date+'_SWE.jpg'
    basins = ['Hell_Hole','French_Meadows']
    new_im = None
    change_im = None
    x_offset = 0
    for basin in basins:
        output_dir = os.path.join(dir_path, 'images', basin, date)

        comparison_days = [0, -7]

        mfiles=[]
        pfiles=[]
        for i, cd in enumerate(comparison_days):
            #pfiles.append(output_dir + '/' + date+'_'+str(-cd)+'_'+basin+'_plot.png')
            mfiles.append(output_dir + '/' + date+'_'+str(-cd)+'_'+basin+'.png')

        #mfiles.merge(pfiles)
        images =list(map(Image.open,mfiles))
        widths, heights = zip(*(i.size for i in images))

        #pimages=list(map(Image.open,pfiles))

        total_width = sum(widths)
        max_height = max(heights)

        #newWidth = int(widths[0] * 0.5)
        newWidth = int(widths[0])
        wpercent = (newWidth / float(images[0].size[0]))
        hsize = int((float(images[0].size[1]) * float(wpercent)))
        top_bottom_margin = 500
        if new_im == None:
            new_im = Image.new('RGB', (widths[0]*2, max_height+top_bottom_margin), (255,255,255))
            #change_im = Image.new('RGB', (widths[0] * 2, max_height + hsize))

        new_im.paste(images[0],(x_offset,int(0+(top_bottom_margin/2))))                        #Paste SWE Map on Top
        #im = pimages[0].resize((newWidth, hsize), Image.ANTIALIAS)  #Match Size of graph to SWE Map
        #new_im.paste(im, (x_offset, hsize))                         #Paste Graph

        #change_im.paste(images[1], (x_offset, 0))  # Paste SWE Map on Top
        #pim = pimages[1].resize((newWidth, hsize), Image.ANTIALIAS)  # Match Size of graph to SWE Map
        #change_im.paste(pim, (x_offset, hsize))  # Paste Graph
        x_offset = widths[0]

        draw = ImageDraw.Draw(new_im)  # This is for text
        font = ImageFont.truetype("micross.ttf", 244)  # Avail in C:\\Windows\Fonts
        plus_sign = ''
        if combined_change > 0:
            plus_sign = "+"


        tot_text = 'Combined Total Acre Feet = ' + "{:,}".format(combined_basin_tot) + " AF"
        delta_text = 'Combined 7 Day Change = ' + plus_sign + "{:,}".format(combined_change) + " AF"
        if combined_basin_tot == -99999999:
            tot_text = 'Combined Total Acre Feet = N/A'
            delta_text = 'Combined 7 Day Change = N/A'
        wt, ht = draw.textsize(tot_text) # Get the height and width of the top line of text for the total AF
        wd, hd = draw.textsize(delta_text) # Get the height and width of the bottom line of text for the change in AF

        # Put the text in the middle of the image (widths - text width)/2, and color black (0,0,0)
        draw.text(((widths[0]-wt)/2, 0), tot_text,(0, 0, 0), font=font)
        draw.text(((widths[0]-wd)/2, hsize ), delta_text,(0, 0, 0), font=font)
    new_im = new_im.resize((int(total_width/10),int(max_height/10)), Image.ANTIALIAS)
    new_im.save(os.path.join('G:/Energy Marketing/Weather', ofile))
    #change_im.save('Change.jpg')

    #x_offset = 0
    #for pim in pimages:
        #new_im.paste(pim, (x_offset,heights[0]-50))
        #x_offset += widths[0]



if __name__ == "__main__":
    main(None, None, None)
