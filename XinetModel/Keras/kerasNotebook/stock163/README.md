# newsCrawler


# 使用前需要先：
1. 安装scrapy
2. 创建一个scrapy项目
3. 将相应的代码拷贝进项目对应的目录

# 主要特点

- 1. 可以定制抓取某一天的网页
- 2. 抓取项目作为字典输出到单个文本文件，而不是输出到MongoDB，这样更简单
- 3. 一些旧的代码不适应新scrapy版本的地方进行的修改，不一一列出。
- 4. 将股票代码和新闻页码作为参数，可以灵活控制抓取内容

# 使用方法：
- 在..\newsCrawler\ 根目录下执行如下DOS命令：
--  ..\newsCrawler>scrapy crawl stocknews -a id='600000' -a page='02'

- 在..\newsCrawler\ 根目录下执行如下DOS命令来调用Python脚本执行抓取所有股票对应的历史新闻：
--  ..\newsCrawler>python PythonRun.py 

 
