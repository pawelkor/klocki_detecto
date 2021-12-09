#!/usr/bin/python
from experiments import Experiments
import os, sys, subprocess
import json
import new_experiment
import train

CONFIG_FILE = 'src/detecto/config.json'

run_eval_everything = True if '-e' in sys.argv else False

f = open(CONFIG_FILE, 'r')
config = json.load(f)
exps = Experiments(config)

#exps.delete_unfinished()
for e in exps.needed():
    print(e)
    batch, pp, template = e
    exp = f"{batch}_{pp}_{template}"
    new_experiment.create(exps, exp, batch, pp, template)
    r = train.run(exp)
    if 0 != r:  exit(1)
    exps.finish(exp)

if run_eval_everything:
    if 0 != subprocess.call(['bash', 'scripts/eval_everything.sh']):
        exit(1)
