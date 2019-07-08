#!/usr/bin/python3
# This file is part of Bootstrapped Dual Policy Iteration
#
# Copyright 2018, Vrije Universiteit Brussel (http://vub.ac.be)
#     authored by Denis Steckelmacher <dsteckel@ai.vub.ac.be>
#
# BDPI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BDPI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BDPI.  If not, see <http://www.gnu.org/licenses/>.
import subprocess
import sys
import os

import psutil

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['LANG'] = 'C'

ENVS = ['table', 'frozenlake', 'largegrid', 'lunarlander']

def log(s):
    print('\033[32m' + s + '\033[0m')

def run_env(command, cores, num_cpus, name, multithread):
    """ Run an environment on several cores
    """
    f = open(name, 'w')
    rs = []

    # Launch the processes, each process has an affinity to one core
    processes = []

    if not multithread:
        # Launch replicas of the process
        for i in range(cores):
            p = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            processes.append(p)

            thread_index = i

            psutil.Process(p.pid).cpu_affinity([thread_index])
    else:
        # Launch one process, it will use the requested threads
        p = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        processes.append(p)

        psutil.Process(p.pid).cpu_affinity(list(range(cores)))


    # Read the lines
    batches_seen = 0

    while True:
        # Start perf-stat
        perf = subprocess.Popen(['perf', 'stat', '-a'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Read the lines
        lines = []

        while len(lines) < 1:
            for p in processes:
                for line in p.stdout:
                    if line.startswith(b'Learned') and b'0 steps' not in line:
                        lines.append(str(line, 'ascii'))
                        break

        batch_size = int(lines[0].split()[1])

        # Get statistics about the number of instructions per second
        insns = 0.0

        perf.send_signal(subprocess.signal.SIGINT)
        perf.wait()

        for line in perf.stdout:
            if b'insn per cycle' in line:
                insns = float(line.split()[3])

        # Compute the total speed of the processes
        s = 0.0

        for l in lines:
            s += float(l.split()[6])

        # Summary of the statistics
        rs.append((s, insns))

        print('%i/256 %f t/s, ipc %f' % (batch_size, s, insns), end='\r')
        sys.stdout.flush()

        # Write the statistics
        if batch_size == 256:
            batches_seen += 1

        if batches_seen < (64 if multithread else 8):
            # First 256 batches, where JIT happens. Discard it
            pass
        else:
            f.write('%i %i %f %f\n' % (cores, batch_size, s, insns))
            break

    # Kill the processes
    for p in processes:
        p.kill()

    # Return the statistics, for easy debugging in the console
    f.close()

    return rs

def main():
    if len(sys.argv) < 2:
        print('Usage:', sys.argv[0], '<num_cpus> [run_cpu]')
        return

    num_cpus = int(sys.argv[1])

    # Prepare the command file for all the environments
    commands = []

    log('WARNING: When comparing processors, use the same num_cpus for all of them, and use run_cpu to set the actual number of cores. Otherwise, results cannot be compared.')
    log('')
    log('Preparing commands...')

    for e in ENVS:
        os.system('bash experiments_gym.sh ' + e)

        # Get the benchmark command
        with open('commands_' + e + '.sh', 'r') as f:
            for line in f:
                if 'rp-16critics' in line and 'epochs20x16-qloops4-1' in line:
                    # Remove the "2> log*"
                    parts = line.split()[:-2]
                    parts[19] = str(num_cpus)           # Force --loops to be the number of CPUs, to get meaningul speed measurements for high thread counts
                    commands.append(' '.join(parts))
                    break
    # All tests
    if len(sys.argv) >= 3:
        all_cpus = [int(sys.argv[2])]
    else:
        all_cpus = [1] + list(range(4, num_cpus+1, 4))

    # Multi-thread test
    for c, e in zip(commands, ENVS):
        for threads in reversed(all_cpus):
            log('Benchmark multi-thread x%i on %s...' % (threads, e))

            c = c + ' --threads %i' % threads
            rs = run_env(c, threads, num_cpus, 'out-%s-%ix-threads.csv' % (e, threads), multithread=True)

            print('    %i t/s, %f instructions per cycle' % (rs[-1][0], rs[-1][1]))

    # Multi-process test
    for c, e in zip(commands, ENVS):
        for threads in reversed(all_cpus):
            log('Benchmark multi-process x%i on %s...' % (threads, e))

            rs = run_env(c, threads, num_cpus, 'out-%s-%ix.csv' % (e, threads), multithread=False)

            print('    %i t/s, %f instructions per cycle' % (rs[-1][0], rs[-1][1]))

if __name__ == '__main__':
    main()
