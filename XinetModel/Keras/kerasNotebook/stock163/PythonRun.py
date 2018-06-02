"""
# multiple spiders

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

process = CrawlerProcess(get_project_settings())

# 'stocknews' is the name of one of the spiders of the project.
for id in ['000001', '000002', '000003']:
    for page in range(5):
        print("*****************************************")
        print("*                                       *")
        print("*                                       *")
        print("* WORKING ON PAGE %s                    *" % page)
        print("*                                       *")
        print("*****************************************")
        process.crawl('stocknews', id=id, page=page)
process.start() # the script will block here until the crawling is finished
"""

from twisted.internet import reactor, defer
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings


configure_logging()
runner = CrawlerRunner()
process = CrawlerProcess(get_project_settings())

fo = open("e:\\data\\FinTech\\StockList.txt", "r")
stocklist=fo.read()
fo.close()
#now combine each file name into a list
stocklist=[s for s in stocklist.split("\n")]
@defer.inlineCallbacks
def crawl():
    for stock in stocklist[:]:
        stockid, market, misc = stock.split(".")
        if market=="SZ":
           marketid="1"
        elif market=="SH":
           marketid="2"
        else:
           pass    
        page = 0
        counter=100
        while (page <=100):
            if page>0:
               counterfile="e:\\data\\FinTech\\News\\Stocks\\%s\\counter.txt" % stockid
               fo=open(counterfile, "r")
               counter=int(fo.read())
               fo.close()
            if int(counter>1):
               print("*****************************************************")
               print("**                                                 **")
               print("**                                                 **")
               print("**                                                 **")   
               print("Currently work on Market=%s, StockID=%s, Page=%s" % (market, stockid, page))
               print("**                                                 **")
               print("**                                                 **")
               print("**                                                 **")
               print("*****************************************************")
               yield process.crawl('stocknews', id=stockid, page=str(page))   
            else:
               page=999  
            page += 1
    reactor.stop()

crawl()
reactor.run() # the script will block here until the last crawl call is finished

