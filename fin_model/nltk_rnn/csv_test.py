from ether.instance.format.neoCsv import neoCsvReader

debug = True
filePath = r'E:\VirtualDesktop\nnet\csv\reddit-comments-2015-08.csv'
model_fname = 'nltk-rnn'

csv_reader = neoCsvReader(filePath)
db = csv_reader.read_all()
r = 1
