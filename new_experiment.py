from experiments import Experiments
import pandas as pd
import os
import json
import click
import shutil
import re

CONFIG_FILE = 'config.json'
with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

def create(exps, name, batch, pp, template):
    if name is None:
        name = '_'.join([batch, pp, template])

    exps.add({
        'exp': name,
        'batch': batch,
        'pp': pp,
        'template': template,
        'finished': 0
    })

    exps.set_num_steps(name,10000)
    p_path = os.getcwd()
    epath = p_path + '/experiments/' + name
    with open(epath + '/pipeline.csv', 'w') as f:
        f.write(batch + '\n')
        f.write(pp + '\n')
        f.write(template + '\n')

@click.command()
@click.option('--name', help='Experiment name')
@click.option('--batch', default=config['BATCH'][0],
              help='Training batch')
@click.option('--pp', default=config['PP'][0],
              help='Preprocessing type')
@click.option('--template', default=config['TEMPLATE'][0],
              help='pipeline.config template')
def run(*args, **kwargs):
    exps = Experiments(config)
    create(exps, *args, **kwargs)

if __name__ == '__main__':
    run()
