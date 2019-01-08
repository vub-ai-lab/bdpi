#!/usr/bin/python3
import sys
import math

if len(sys.argv) < 3:
    print("Usage: %s <column> <prefix> [file...]" % sys.argv[0])
    sys.exit(0)

# Open all files
col = int(sys.argv[1])
prefix = sys.argv[2]
files = [open(f, 'r') for f in sys.argv[3:]]

# Read and average each files
N = float(len(files))
i = 0
running_mean = None
running_err = None
elems = [0.0] * len(files)
timesteps = [0] * len(files)
elements = [None] * len(files)

running_mean = None
running_err = 0.0
running_coeff = 0.95

while True:
    # Read a line from every file
    ok = False

    for j, f in enumerate(files):
        while True:
            elements[j] = f.readline().strip().replace(',', ' ')

            if elements[j].startswith(prefix) or len(elements[j]) == 0:
                break

        if len(elements[j]) > 0:
            ok = True

    if not ok:
        # No more file
        break

    try:
        # Plot lines
        N = 0

        for j in range(len(files)):
            if len(elements[j]) > 0:
                parts = elements[j].split()

                elems[j] = float(parts[col])
                try:
                    timesteps[j] = int(parts[2])
                except:
                    timesteps[j] = 0

                N += 1
            else:
                elems[j] = 0.0
                timesteps[j] = 0

        mean = sum(elems) / N
        var = sum([(e - mean)**2 for e in elems if e != 0.0])
        std = math.sqrt(var)
        err = std / N

        if running_mean is None:
            running_mean = mean
            running_err = err
        else:
            running_mean = running_coeff * running_mean + (1.0 - running_coeff) * mean
            running_err = running_coeff * running_err + (1.0 - running_coeff) * err

        if i % 16 == 0:
            print(i / 1000., running_mean, running_mean + running_err, running_mean - running_err, sum(timesteps) / N)

        i += 1
    except Exception:
        pass
