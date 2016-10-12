from operator import add
from pyspark import SparkConf, SparkContext
import sys
import numpy as np
from hdfs import InsecureClient
import os

def map_line(line):
  tokens = line.split(",")
  # parse the original data line, which is (row_id, column_id, value)
  return int(tokens[0]), int(tokens[1]), float(tokens[2])

def loop_thru_one_block(tup, y_block_dim, x_block_dim, step_size):  
  H_one = tup[1][0][1]
  W_one = tup[1][0][0]
  block_it = tup[1][1]  #iterable of (blk_id,x,y,val)
  return sgd_on_one_block(block_it, H_one, y_block_dim,  W_one, x_block_dim, step_size)  

def sgd_on_one_block(block_it, H_one, y_block_dim, W_one, x_block_dim, step_size):
  for dot in block_it:  # block_it: blk_id, x, y, val
    x = dot[1]
    y = dot[2]
    val = dot[3]
    i = x/x_block_dim
    j = y/y_block_dim
    H_offeset = j*y_block_dim
    W_offeset = i*x_block_dim
    H_row = H_one[y - H_offeset]
    W_row = W_one[x - W_offeset]
    diff = val - np.dot(W_row, H_row)
    W_gradient = -2 * diff * H_row
    W_row -= step_size * W_gradient
    H_gradient = -2 * diff * W_row
    H_row -= step_size * H_gradient
  return (j,H_one), (i, W_one)

# def write_hdfs(out_path,items, hdfs_client):
#   fname = os.path.basename(out_path)
#   with open(fname) as writer:
#     out = ''
#     for tup in items:
#       blk = tup[1]
#       for line in blk:
#         output_line = str(list(line)).strip('[').strip(']').replace(' ','') + '\n'
#         out += output_line
#     writer.write(out)
#   os.system("hadoop fs -put {} {}".format(fname, out_path))
def write_hdfs(out_path,items, hdfs_client):
  with hdfs_client.write(out_path) as writer:
    out = ''
    for tup in items:
      blk = tup[1]
      for line in blk:
        output_line = str(list(line)).strip('[').strip(']').replace(' ','') + '\n'
        out += output_line
    writer.write(out)

class Config:
  input_path = ''
  output_H = ''
  rank = 0
  output_W = ''
  eta = 0.001
  eta_decay = 0.99
  N = 8 
  def __init__(self):
    self.input_path = sys.argv[1]
    self.rank = int(sys.argv[2])
    self.output_W = sys.argv[3]+'/part-00000'
    self.output_H = sys.argv[4]+'/part-00000'

    os.system("hadoop fs -mkdir {}".format(sys.argv[3]))
    os.system("hadoop fs -mkdir {}".format(sys.argv[4]))

    pass


if __name__ == '__main__':
  
  num_iterations = 10

  conf = SparkConf().setAppName('my sgd')
  sc= SparkContext(conf = conf)

  brd_conf = Config()

  N = brd_conf.N
  matrix_rdd = sc.textFile(brd_conf.input_path, N).map(map_line).persist()

  max_x_id = matrix_rdd.map(lambda x: x[0]).max()
  max_y_id = matrix_rdd.map(lambda x: x[1]).max()

  matrix_rdd.unpersist()


  # assume the id starts from 0
  x_block_dim = sc.broadcast(int((max_x_id + N) / N))
  y_block_dim = sc.broadcast(int((max_y_id + N) / N))

  rank = brd_conf.rank

  #initialize H, M
  np.random.seed(1)
  H_rdd = sc.parallelize(range(N)).map(lambda x : (x,np.random.rand( y_block_dim.value, rank))).partitionBy(N)
  np.random.seed(1)
  W_rdd = sc.parallelize(range(N)).map(lambda x : (x,np.random.rand( x_block_dim.value, rank))).partitionBy(N)

  H_dict = dict(H_rdd.collect())
  W_dict = dict(W_rdd.collect())

  #matrix_rdd_array = {}

  # for i in xrange(N):
  #   for j in xrange(N):
  #     matrix_rdd_array{i,j} = matrix_rdd.filter(lambda x: x[0]/x_block_dim == i && x[1]/y_block_dim == j)

  #matrix_rdd = matrix_rdd.map(lambda x: (str(x[0]/x_block_dim.value) + ':'+ str(x[1]/y_block_dim.value), x[0],  x[1], x[2])).groupBy(lambda x:x[0],numPartitions=N*N)

  matrix_rdd = matrix_rdd.map(lambda x: ( (x[0]/x_block_dim.value) + (x[1]/y_block_dim.value)*N, x[0],  x[1], x[2])).groupBy(lambda x:x[0],numPartitions=N*N).cache()


  # for i in range(0, num_iterations):
  #   matrix_rdd.foreach(lambda x:loop_thru_blocks(x, H_rdd, W_rdd, y_block_dim.value, x_block_dim.value, brd_conf.eta))
  #   brd_conf.eta *= brd_conf.eta_decay

  for i in range(0, num_iterations):
    for delta in range(N):
      matrix_part = matrix_rdd.filter(lambda line: ((line[0]%N - line[0]/N == delta) or (line[0]/N - line[0]%N == N-delta)) ).map(lambda line: (line[0]/N, line[1])).partitionBy(N)
      W_shift = W_rdd.map(lambda x: ((x[0] - delta)%N, x[1])).partitionBy(N)

      MHW_union = W_shift.join(H_rdd).join(matrix_part).partitionBy(N)
      updated_HM_rdd = MHW_union.map(lambda line: loop_thru_one_block(line, y_block_dim.value, x_block_dim.value, brd_conf.eta)).persist()

      H_updated = updated_HM_rdd.map(lambda x: x[0]).collect()
      W_updated = updated_HM_rdd.map(lambda x: x[1]).collect()

      H_dict.update(dict(H_updated))
      W_dict.update(dict(W_updated))

      updated_HM_rdd.unpersist()

      W_rdd = sc.parallelize(W_dict.items()).partitionBy(N)
      H_rdd = sc.parallelize(H_dict.items()).partitionBy(N)

    brd_conf.eta *= brd_conf.eta_decay


  sc.stop()
  W_items = sorted(W_dict.items())
  H_items = sorted(H_dict.items())
  hdfs_client = InsecureClient('http://ec2-52-3-241-11.compute-1.amazonaws.com:50070', user='hadoop')

  write_hdfs(brd_conf.output_W,W_items, hdfs_client)

  write_hdfs(brd_conf.output_H,H_items, hdfs_client)


