from ether.instance.format.fcvs import csvReader

debug = True
filePath = r'E:\VirtualDesktop\nnet\csv\reddit-comments-2015-08.csv'
model_fname = 'nltk-rnn'

csv_reader = csvReader(filePath, attrSelected=[0])
db = csv_reader.read_all()
r = 1
