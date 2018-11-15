import shutil
import os

file = '/opt/spark-2.2.0-bin-hadoop2.6/work'
def del_file(path):
    shutil.rmtree(path)
    os.mkdir(path)

del_file(file)
