#!/bin/bash
RUNS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
PYTHON="python3"
MAIN_VARIANT="rp"

if [ "$1" = "largegrid" ]
then
    ENV=largegrid
    ENVN=LargeGrid-v0
    LEN=2000
    HIDDEN=32
elif [ "$1" = "frozenlake" ]
then
    ENV=frozenlake
    ENVN=FrozenLake8x8-v0
    LEN=2000
    HIDDEN=32
elif [ "$1" = "table" ]
then
    ENV=table
    ENVN=Table-v0
    LEN=2000
    HIDDEN=32
elif [ "$1" = "tablernd" ]
then
    ENV=tablernd
    ENVN=TableRandom-v0
    LEN=4000
    HIDDEN=32
else
    echo "No environment given, give largegrid, frozenlake or table"
    exit 1
fi

> commands_$ENV.sh

# BDPI
if true
then
    parallel --header : echo $PYTHON main.py \
        --name "parallel-$ENV-{variant}-{ac}critics-er{er}-epochs{epochs}x{loops}-qloops{qloops}-{run}" \
        --pursuit-variant "{variant}" \
        --env $ENVN \
        --episodes $LEN \
        --hidden $HIDDEN \
        --lr 0.0001 \
        --er {er} \
        --erfreq 1 \
        --loops {loops} \
        --actor-count {ac} \
        --q-loops {qloops} \
        --epochs {epochs} \
        --erpoolsize 20000 "\"2>\"" "log-parallel-$ENV-{variant}-{ac}critics-er{er}-epochs{epochs}x{loops}-qloops{qloops}-{run}" ">>" commands_$ENV.sh \
        ::: er 512 \
        ::: loops 1 16 \
        ::: ac 4 16 \
        ::: epochs 20 \
        ::: qloops 1 2 4 \
        ::: variant rp ri generalized \
        ::: run $RUNS
fi

# Bootstrapped DQN
if true
then
    parallel --header : echo $PYTHON main.py \
        --name "bootstrapped-$ENV-{ac}critics-er{er}-epochs{epochs}x{loops}-qloops{qloops}-{run}" \
        --learning-algo egreedy \
        --temp 0.0 \
        --env $ENVN \
        --episodes $LEN \
        --hidden $HIDDEN \
        --lr 0.0001 \
        --er {er} \
        --erfreq 1 \
        --loops {loops} \
        --actor-count {ac} \
        --q-loops {qloops} \
        --epochs {epochs} \
        --erpoolsize 20000 "\"2>\"" "log-bootstrapped-$ENV-{ac}critics-er{er}-epochs{epochs}x{loops}-qloops{qloops}-{run}" ">>" commands_$ENV.sh \
        ::: er 512 \
        ::: loops 1 16 \
        ::: ac 16 \
        ::: epochs 1 20 \
        ::: qloops 1 4 \
        ::: run $RUNS
fi

# Run all commands
echo -e "\033[37mGNU Parallel\033[0m: cat commands_$ENV.sh | parallel -jCORES"
