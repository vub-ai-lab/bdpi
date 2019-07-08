#!/bin/bash
RUNS="1 2 3 4 5 6 7 8"
PYTHON="python3"
MAIN_VARIANT="rp"

LR=1e-4
EPOCHS=20

if [ "$1" = "largegrid" ]
then
    ENV=largegrid
    ENVN=LargeGrid-v0
    LEN=2000
    HIDDEN=32
    NOISE=0.2
elif [ "$1" = "frozenlake" ]
then
    ENV=frozenlake
    ENVN=FrozenLake8x8-v0
    LEN=2000
    HIDDEN=32
    NOISE=0.2
elif [ "$1" = "table" ]
then
    ENV=table
    ENVN=Table-v0
    LEN=500
    HIDDEN=32
    NOISE=0.05
elif [ "$1" = "tablernd" ]
then
    ENV=tablernd
    ENVN=TableRandom-v0
    LEN=4000
    HIDDEN=128
    NOISE=0.05
elif [ "$1" = "lunarlander" ]
then
    ENV=lunarlander
    ENVN=LunarLander-v2
    LEN=1000
    HIDDEN=256
    NOISE=0.2
elif [ "$1" = "hallway" ]
then
    ENV=hallway
    ENVN=MiniWorld-Hallway-v0
    LEN=500
    HIDDEN=256
    NOISE=0.2
    LR=1e-5
    EPOCHS=1
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
        --lr $LR \
        --er {er} \
        --erfreq 1 \
        --loops {loops} \
        --actor-count {ac} \
        --q-loops {qloops} \
        --aepochs {epochs} \
        --cepochs {epochs} \
        --erpoolsize 20000 "\"2>\"" "log-parallel-$ENV-{variant}-{ac}critics-er{er}-epochs{epochs}x{loops}-qloops{qloops}-{run}" ">>" commands_$ENV.sh \
        ::: er 256 \
        ::: loops 8 16 32 \
        ::: ac 8 16 32 \
        ::: epochs $EPOCHS \
        ::: qloops 1 2 4 \
        ::: variant rp mimic \
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
        --lr $LR \
        --er {er} \
        --erfreq 1 \
        --loops {loops} \
        --actor-count {ac} \
        --q-loops {qloops} \
        --aepochs {epochs} \
        --cepochs {epochs} \
        --erpoolsize 20000 "\"2>\"" "log-bootstrapped-$ENV-{ac}critics-er{er}-epochs{epochs}x{loops}-qloops{qloops}-{run}" ">>" commands_$ENV.sh \
        ::: er 256 \
        ::: loops 1 16 \
        ::: ac 16 \
        ::: epochs 1 $EPOCHS \
        ::: qloops 1 4 \
        ::: run $RUNS
fi

# BDPI with off-policy noise
if true
then
    parallel --header : echo $PYTHON main.py \
        --name "parallel-noise-$ENV-{variant}-{ac}critics-er{er}-epochs{epochs}x{loops}-qloops{qloops}-{run}" \
        --offpolicy-noise $NOISE \
        --pursuit-variant "{variant}" \
        --env $ENVN \
        --episodes $LEN \
        --hidden $HIDDEN \
        --lr $LR \
        --temp 0 \
        --er {er} \
        --erfreq 1 \
        --loops {loops} \
        --actor-count {ac} \
        --q-loops {qloops} \
        --aepochs {epochs} \
        --cepochs {epochs} \
        --erpoolsize 20000 "\"2>\"" "log-parallel-noise-$ENV-{variant}-{ac}critics-er{er}-epochs{epochs}x{loops}-qloops{qloops}-{run}" ">>" commands_$ENV.sh \
        ::: er 256 \
        ::: loops 16 \
        ::: ac 16 \
        ::: epochs $EPOCHS \
        ::: qloops 4 \
        ::: variant rp \
        ::: run $RUNS
fi

# Bootstrapped DQN with off-policy noise
if true
then
    parallel --header : echo $PYTHON main.py \
        --name "bootstrapped-noise-$ENV-{ac}critics-er{er}-epochs{epochs}x{loops}-qloops{qloops}-{run}" \
        --offpolicy-noise $NOISE \
        --learning-algo egreedy \
        --temp 0 \
        --env $ENVN \
        --episodes $LEN \
        --hidden $HIDDEN \
        --lr 0.0001 \
        --er {er} \
        --erfreq 1 \
        --loops {loops} \
        --actor-count {ac} \
        --q-loops {qloops} \
        --aepochs {epochs} \
        --cepochs {epochs} \
        --erpoolsize 20000 "\"2>\"" "log-bootstrapped-noise-$ENV-{ac}critics-er{er}-epochs{epochs}x{loops}-qloops{qloops}-{run}" ">>" commands_$ENV.sh \
        ::: er 256 \
        ::: loops 1 \
        ::: ac 16 \
        ::: epochs $EPOCHS \
        ::: qloops 1 \
        ::: run $RUNS
fi

# Run all commands
echo -e "\033[37mGNU Parallel\033[0m: cat commands_$ENV.sh | parallel -jCORES"
