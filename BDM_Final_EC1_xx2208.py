
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import shapely
import math
from pyproj import Transformer
from shapely.geometry import Point

# import datetime
import IPython
%matplotlib inline
IPython.display.set_matplotlib_formats('svg')
pd.plotting.register_matplotlib_converters()
sns.set_style("whitegrid")

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.types import DateType, IntegerType, MapType, StringType
from pyspark.sql.types import StructType,StructField 
from pyspark.sql.types import ArrayType


if __name__ == "__main__":
  sc = pyspark.SparkContext.getOrCreate()
  spark = SparkSession(sc)

  # functions
  def date(date1,date2,cbgs): #to filter date in range
    if date1 == '2019-03' or date2 == '2019-03':
      return [cbgs,{},{},{}]
    elif date1 == '2019-10' or date2 == '2019-10':
      return [{},cbgs,{},{}]
    elif date1 == '2020-03' or date2 == '2020-03':
      return [{},{},cbgs,{}]
    elif date1 == '2020-10' or date2 == '2020-10':
      return [{},{},{},cbgs]
    else:
      None

  def merge(x,y): #to merge the dates with the same cbgs
    output = [{},{},{},{}]
    for i in range(len(x)):
      output[i].update(x[i])
      output[i].update(y[i])
    return output

  def getcbgs(dict_in,filter_list): #to filter all cbgs that in nyc
    output = []
    for dict_ in dict_in:
      if dict_ == {}: output.append('')
      else:
        dict_out = []
        for item in dict_:
          if item in filter_list:
            dict_out.append((item,dict_[item]))
        output.append(dict_out)
    return output

  def transfer(input,transfer_list): #to transfer cbgs into points
    t = Transformer.from_crs(4326, 2263)
    if type(input) == list: 
      list_out = []
      for dict_ in input:
        if dict_ == '': list_out.append('')
        else:
          dict_out = []
          for item1 in dict_:
            for item2 in transfer_list:
              if item1[0] == item2[0]:
                dict_out.append((t.transform(item2[1],item2[2]),item1[1]))
          list_out.append(dict_out)
      return list_out
    else:
      for item in transfer_list:
        if input == item[0]:
          return t.transform(item[1],item[2])

  def distance(start_list,destination): #calculate the distances betweent cbgs and points
    output = []
    for item in start_list:
      if item == '':
        output.append('')
      else:
        distance_list=[]
        for start in item:
          distance_list.append((Point(start[0][0],start[0][1]).distance(Point(destination[0],destination[1]))/5280,start[1]))
        output.append(distance_list)
    return output

  def median_dist(input): #calculate the median distances
    output = []
    for item in input:
      if item == '':
        output.append('')
      else:
        list_ = []
        for cuple in item:
          list_ += [cuple[0] for i in range(cuple[1])]
        output.append(round(np.median(list_),2))
    return output

  fips = sc.textFile('nyc_cbg_centroids.csv').cache() #read the cbgs in nyc and extract the cgbs' list to filter weekly_patterns
  fips_header = fips.first()
  fips_filter = fips \
      .filter(lambda x: x != fips_header) \
      .map(lambda x: next(csv.reader([x]))[0]) \
      .cache().collect()

  fips_lalon = fips \
      .map(lambda x: [x.split(',')[0],x.split(',')[1],x.split(',')[2]]) \
      .cache().collect()

  placekey_filter = sc.textFile('nyc_supermarkets.csv') \
      .map(lambda x: next(csv.reader([x]))) \
      .map(lambda x: (x[9])) \
      .collect()
      
  rdd_first = sc.textFile('/tmp/bdm/weekly-patterns-nyc-2019-2020').map(lambda x: next(csv.reader([x]))) #read the weekly patterns in sample data
  header = rdd_first.first()
  rddT0 =  rdd_first \
      .filter(lambda x: x != header) \
      .map(lambda x: [x[0],
                      '-'.join(x[12].split('T')[0].split('-')[:2]),
                      '-'.join(x[13].split('T')[0].split('-')[:2]),
                      x[18],
                      json.loads(x[19])])\
      .cache()

  rddT1 = rddT0 \
      .filter(lambda x: x[0] in placekey_filter) \
      .cache() #approach 1, filter the visits in nyc supermarkets with placekey

  rddT2 = rddT1 \
      .map( lambda x: (x[3],date(x[1],x[2],x[4]))) \
      .filter(lambda x: x[1] is not None) \
      .reduceByKey(lambda x,y: merge(x,y)) \
      .cache() #approach 2, filter periods in range

  rddT3 = rddT2 \
      .map(lambda x: [x[0],getcbgs(x[1], fips_filter)]) \
      .cache() #approach 3, filter visitors' cbgs in nyc

  rddT4 = rddT3\
      .map(lambda x: [x[0],transfer(x[0],fips_lalon),transfer(x[1],fips_lalon)])\
      .map(lambda x: [x[0],distance(x[2],x[1])])\
      .cache() #approach 4, projections and calculate the distances

  result = rddT4 \
      .map(lambda x: [x[0],median_dist(x[1])]) \
      .map(lambda x: ((str(x[0]), str(x[1][0]) , str(x[1][1]), str(x[1][2]), str(x[1][3]))))\
      .toDF(['cbg_fips', '2019-03' , '2019-10' , '2020-03' , '2020-10']) #calculate the average distance of each cbgs of each months and then transfer rdd to dataframe

  result.write.csv(sys.argv[1])