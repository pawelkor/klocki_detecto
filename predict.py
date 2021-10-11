from pathlib import Path
from glob import glob
import pandas as pd
from detecto.utils import read_image
from detecto.core import Model
from object_detection.utils import label_map_util  # part of tensorflow?
import os, sys

expt_dir = sys.argv[1]
eval_name = sys.argv[2]
batch_dir = './'
klocki_home = os.path.expanduser('~/klocki/')
eval_dir = klocki_home + 'data/sliced/' + eval_name
recursive = False
model_file = 'experiments/' + expt_dir + '/' + expt_dir + '.pth'
label_file = klocki_home + 'data/label_map.pbtxt'
# end of 'params'
label_map = label_map_util.get_label_map_dict(label_map_util.load_labelmap(label_file))
classes = [*label_map]
classes = ['c' + c for c in classes]

model = Model.load(model_file, classes)
files = glob(eval_dir + '/*.jpg', recursive=recursive) + glob(eval_dir + '/*.png', recursive=recursive)
files.sort()
image_list = [read_image(f) for f in files]

# import gc
# gc.collect()
# torch.cuda.empty_cache()
llist=[]
for i, image in enumerate(image_list):
    img_predictions = model.predict(image)
    print(files[i])
    p = Path(files[i])
    filename = p.parts[-1]
    labels, boxes, scores = img_predictions
    for l, label in enumerate(labels):
        # print()
        # print(labels[l])
        # print(boxes[l])
        # print(scores[l])
        # print()
        llist.append((filename,
                      boxes[l][0].tolist(), boxes[l][1].tolist(), boxes[l][2].tolist(), boxes[l][3].tolist(),
                      label[1:],
                      scores[l].tolist(),
                      eval_name
        ))

column_names = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class', 'score', 'source']
df = pd.DataFrame(llist, columns=column_names)
df.to_csv('experiments/' + expt_dir + '/predictions.csv', index=False)
