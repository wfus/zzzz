from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from com.yahoo.ml.tf import TFCluster, TFNode
from datetime import datetime


def main_fun(argv, ctx):
    """Main function entrance for spark. Make sure that all imports are done here,
    or spark will try to serialize libraries when they are placed outside
    for each executor, and we don't want that! ~WFU"""
    import tensorflow as tf


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args, rem = parser.parse_known_args()

    sc = SparkContext(conf=SparkConf().setAppName("Nacho"))
    num_executors = int(sc._conf.get("spark.executor.instances"))
    num_processes = 1
    use_tensorboard = False

    cluster = TFCluster.run(sc, main_fun, sys.argv, num_executors, num_processes,
                            use_tensorboard, TFCluster.InputMode.TENSORFLOW)
    cluster.shutdown()


