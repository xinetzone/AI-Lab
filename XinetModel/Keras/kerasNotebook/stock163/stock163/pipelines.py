# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
#encoding: utf-8
import os
def ParseFilePath(url, id):
    # user should change this folder path
    outfolder = "e:\\data\\FinTech\\News\\Stocks\\%s" % id
    components = url.split("/")
    year = components[3]
    monthday=components[4]
    month = monthday[:2]
    day = monthday[2:]
    idx=components[5]
    page=idx+"_"+components[6]
    #folder = outfolder + "\\%s_%s_%s_" % (year, month, day)
    folder = outfolder
    if ((year=='') | ('keywords' in page)):
       filepath='xxx'
    else:
       filepath = folder + "\\%s_%s_%s_%s.txt" % (year, month, day, page) 
    filepath=filepath.replace('?', '_')
    return(folder, filepath)

class Stock163Pipeline(object):   
    def process_item(self, item, spider):
        if spider.name != "stocknews":  return item
        if item.get("news_thread", None) is None: return item
                
        url = item['news_url']
        if 'keywords' in url:
           return item
        folder, filepath = ParseFilePath(url, spider.stock_id)
        spider.counter = spider.counter+1
        counterfilepath = folder+"\\counter.txt"
        #one a single machine will is virtually no risk of race-condition
        if not os.path.exists(folder):
           os.makedirs(folder)        
        #print(filepath, counterfilepath)
        #print(spider.stats)
        fo = open(counterfilepath, "w", encoding="UTF-8")
        fo.write(str(spider.counter))
        fo.close()

        if (filepath!='xxx'):
           fo = open(filepath, 'w', encoding='utf-8')
           fo.write(str(dict(item)))
           fo.close()
        return None
        

