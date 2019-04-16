import os
from glob import glob
from pathlib import Path
import numpy as np
import csv

PATH = "/Users/stephanielew/Projects/iot-sec/code/vggvox-speaker-identification/data/wav/enroll"
result = [ y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.wav'))]
ids = [Path(Path(r).parent).parent.name for r in result]
enroll_list = np.column_stack((result,ids))
print(enroll_list.shape)

with open("cfg/enroll_list.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows([['filename,speaker']])
    writer.writerows(enroll_list)