import os
import sys
import argparse
import re
from tensorboardX import SummaryWriter

def parse_log(log_file):
    iterations = []
    loss_values = []
    map_iterations = []
    map_values = []

    loss_pattern = re.compile(r'(\d+)/\d+: loss=([\d.]+)')
    map_pattern = re.compile(r'(\d+)/\d+: loss=[\d.]+ map=([\d.]+)')

    with open(log_file, 'r') as f:
        for line in f:
            loss_match = loss_pattern.search(line)
            map_match = map_pattern.search(line)

            if loss_match:
                iteration, loss = map(float, loss_match.groups())
                iterations.append(iteration)
                loss_values.append(loss)

            if map_match:
                iteration, map_value = map(float, map_match.groups())
                map_iterations.append(iteration)
                map_values.append(map_value)

    return iterations, loss_values, map_iterations, map_values


def write_tensorboard(iterations, loss_values, map_iterations, map_values, log_dir):
    writer = SummaryWriter(log_dir=log_dir)

    for i, (iteration, loss) in enumerate(zip(iterations, loss_values)):
        writer.add_scalar('Loss/train', loss, iteration)

    for i, (iteration, mAP) in enumerate(zip(map_iterations, map_values)):
        writer.add_scalar('mAP@0.5/train', mAP, iteration)

    writer.close()


def main(args):
    for log_file in args.log_files:
        log_name = os.path.splitext(os.path.basename(log_file))[0]
        log_dir = os.path.join(args.log_dir, log_name)

        iterations, loss_values, map_iterations, map_values = parse_log(log_file)
        write_tensorboard(iterations, loss_values, map_iterations, map_values, log_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_files', type=str, nargs='+', help='Paths to Darknet log files')
    parser.add_argument('--log_dir', type=str, default='runs', help='Path to TensorBoard log directory')
    args = parser.parse_args()

    main(args)