#!/usr/bin/env python3

import json
import logparse as lp
import numpy as np
import sys

from click import echo, style
from collections import deque
from derplog.derplog import Derplog
from keras.preprocessing.sequence import pad_sequences

weight_file='checkpoints/weights.08-0.863.hdf5'
with open('scratch/events.json', 'r') as fd:
    events = json.load(fd)
events = [ lp.LogEvent(*e) for e in events ]

d = Derplog(epocs=10, weight_file=weight_file)

# print(d.lcsmap)

context_size = 0
context_queue = deque(maxlen=context_size)

for i, e in enumerate(events):
    e = lp.LogEvent(*e)
    start = max(0, i-d.W+1)
    window = [ e.key for e in events[start:i] ]
    window = pad_sequences([window], maxlen=d.W-1)[0]

    classes = d.predict_top_classes(window, tol=0.1)

    if e.key not in classes:

        for c in context_queue:
            print(c.log, end='')

        context_queue.clear()

        echo(style('*** ' + e.log, bold=True), err=False, nl=False)

    else:
        context_queue.append(e)
