set terminal pdf size 6.75in,7.5cm font "Times, 11"
set output 'plot.pdf'

set multiplot layout 3,3 columnsfirst

set style fill transparent solid 0.33 noborder
set border back
set macros

# Macros
NOXTICS = "set format x ''; unset xlabel"
NOYTICS = "unset ylabel"
YTICS = "set ylabel 'Return per episode'"
XTICS = "set format x '%1.1fK'"

#V0MARGIN = "set tmargin at screen 0.980; set bmargin at screen 0.555"
#V1MARGIN = "set tmargin at screen 0.555; set bmargin at screen 0.130"
V0MARGIN = "set tmargin at screen 0.950; set bmargin at screen 0.677"
V1MARGIN = "set tmargin at screen 0.677; set bmargin at screen 0.403"
V2MARGIN = "set tmargin at screen 0.403; set bmargin at screen 0.130"
H0MARGIN = "set lmargin at screen 0.060; set rmargin at screen 0.347"
H1MARGIN = "set lmargin at screen 0.387; set rmargin at screen 0.653"
H2MARGIN = "set lmargin at screen 0.693; set rmargin at screen 0.980"

# Algos
table_ppi = "<(python3 ../avg_stats.py 0 '' out-parallel-table-rp-16critics-er256-epochs20x16-qloops4-?)"
table_aabdqn = "<(python3 ../avg_stats.py 0 '' out-bootstrapped-table-16critics-er256-epochs20x16-qloops4-?)"
table_bdqn_opt = "<(python3 ../avg_stats.py 0 '' out-bootstrapped-table-16critics-er256-epochs20x1-qloops1-?)"
table_mimic = "<(python3 ../avg_stats.py 0 '' out-parallel-table-mimic-16critics-er256-epochs20x16-qloops4-?)"
table_ppo_opt = "<(python3 ../avg_stats.py 0 '' table-256-*/0.monitor.csv)"
table_acktr = "<(python3 ../avg_stats.py 0 '' table-acktr-*/monitor.csv)"

lunarlander_ppi = "<(python3 ../avg_stats.py 0 '' out-parallel-lunarlander-rp-16critics-er256-epochs20x16-qloops4-?)"
lunarlander_aabdqn = "<(python3 ../avg_stats.py 0 '' out-bootstrapped-lunarlander-16critics-er256-epochs20x16-qloops4-?)"
lunarlander_bdqn_opt = "<(python3 ../avg_stats.py 0 '' out-bootstrapped-lunarlander-16critics-er256-epochs20x1-qloops1-?)"
lunarlander_mimic = "<(python3 ../avg_stats.py 0 '' out-parallel-lunarlander-mimic-16critics-er256-epochs20x16-qloops4-?)"
lunarlander_ppo_opt = "<(python3 ../avg_stats.py 0 '' lunarlander-ppo-*/0.monitor.csv)"
lunarlander_acktr = "<(python3 ../avg_stats.py 0 '' lunarlander-acktr-*/monitor.csv)"

frozenlake_ppi = "<(python3 ../avg_stats.py 0 '' out-parallel-frozenlake-rp-16critics-er256-epochs20x16-qloops4-?)"
frozenlake_aabdqn = "<(python3 ../avg_stats.py 0 '' out-bootstrapped-frozenlake-16critics-er256-epochs20x16-qloops4-?)"
frozenlake_bdqn_opt = "<(python3 ../avg_stats.py 0 '' out-bootstrapped-frozenlake-16critics-er256-epochs20x1-qloops1-?)"
frozenlake_mimic = "<(python3 ../avg_stats.py 0 '' out-parallel-frozenlake-mimic-16critics-er256-epochs20x16-qloops4-?)"
frozenlake_ppo_opt = "<(python3 ../avg_stats.py 0 '' frozenlake-256-*/0.monitor.csv)"
frozenlake_acktr = "<(python3 ../avg_stats.py 0 '' frozenlake-acktr-*/0.monitor.csv)"

# Compare number of critics and critics updated per time-step
table_ppi_8critics_8loops = "<(python3 ../avg_stats.py 0 '' out-parallel-table-rp-8critics-er256-epochs20x8-qloops4-?)"
table_ppi_16critics_8loops = "<(python3 ../avg_stats.py 0 '' out-parallel-table-rp-16critics-er256-epochs20x8-qloops4-?)"
table_ppi_16critics_16loops = table_ppi
table_ppi_32critics_8loops = "<(python3 ../avg_stats.py 0 '' out-parallel-table-rp-32critics-er256-epochs20x8-qloops4-?)"
table_ppi_32critics_16loops = "<(python3 ../avg_stats.py 0 '' out-parallel-table-rp-32critics-er256-epochs20x16-qloops4-?)"
table_ppi_32critics_32loops = "<(python3 ../avg_stats.py 0 '' out-parallel-table-rp-32critics-er256-epochs20x32-qloops4-?)"

lunarlander_ppi_8critics_8loops = "<(python3 ../avg_stats.py 0 '' out-parallel-lunarlander-rp-8critics-er256-epochs20x8-qloops4-?)"
lunarlander_ppi_16critics_8loops = "<(python3 ../avg_stats.py 0 '' out-parallel-lunarlander-rp-16critics-er256-epochs20x8-qloops4-?)"
lunarlander_ppi_16critics_16loops = lunarlander_ppi
lunarlander_ppi_32critics_8loops = "<(python3 ../avg_stats.py 0 '' out-parallel-lunarlander-rp-32critics-er256-epochs20x8-qloops4-?)"
lunarlander_ppi_32critics_16loops = "<(python3 ../avg_stats.py 0 '' out-parallel-lunarlander-rp-32critics-er256-epochs20x16-qloops4-?)"
lunarlander_ppi_32critics_32loops = "<(python3 ../avg_stats.py 0 '' out-parallel-lunarlander-rp-32critics-er256-epochs20x32-qloops4-?)"

frozenlake_ppi_8critics_8loops = "<(python3 ../avg_stats.py 0 '' out-parallel-frozenlake-rp-8critics-er256-epochs20x8-qloops4-?)"
frozenlake_ppi_16critics_8loops = "<(python3 ../avg_stats.py 0 '' out-parallel-frozenlake-rp-16critics-er256-epochs20x8-qloops4-?)"
frozenlake_ppi_16critics_16loops = frozenlake_ppi
frozenlake_ppi_32critics_8loops = "<(python3 ../avg_stats.py 0 '' out-parallel-frozenlake-rp-32critics-er256-epochs20x8-qloops4-?)"
frozenlake_ppi_32critics_16loops = "<(python3 ../avg_stats.py 0 '' out-parallel-frozenlake-rp-32critics-er256-epochs20x16-qloops4-?)"
frozenlake_ppi_32critics_32loops = "<(python3 ../avg_stats.py 0 '' out-parallel-frozenlake-rp-32critics-er256-epochs20x32-qloops4-?)"

# Compare qloops
table_ppi_16loops_1qloops = "<(python3 ../avg_stats.py 0 '' out-parallel-table-rp-16critics-er256-epochs20x16-qloops1-?)"
table_ppi_16loops_2qloops = "<(python3 ../avg_stats.py 0 '' out-parallel-table-rp-16critics-er256-epochs20x16-qloops2-?)"
table_ppi_16loops_4qloops = table_ppi

lunarlander_ppi_16loops_1qloops = "<(python3 ../avg_stats.py 0 '' out-parallel-lunarlander-rp-16critics-er256-epochs20x16-qloops1-?)"
lunarlander_ppi_16loops_2qloops = "<(python3 ../avg_stats.py 0 '' out-parallel-lunarlander-rp-16critics-er256-epochs20x16-qloops2-?)"
lunarlander_ppi_16loops_4qloops = lunarlander_ppi

frozenlake_ppi_16loops_1qloops = "<(python3 ../avg_stats.py 0 '' out-parallel-frozenlake-rp-16critics-er256-epochs20x16-qloops1-?)"
frozenlake_ppi_16loops_2qloops = "<(python3 ../avg_stats.py 0 '' out-parallel-frozenlake-rp-16critics-er256-epochs20x16-qloops2-?)"
frozenlake_ppi_16loops_4qloops = frozenlake_ppi


# Table
set ytics ('-40' -40, '0' 0, '50' 50, '100' 100)
@V0MARGIN; @H0MARGIN; @NOXTICS; @YTICS
set key bottom left at 0.042,113 maxrows 1 box linewidth 1 opaque
plot [0:0.5] [-50:110] \
    table_acktr using 1:3:4 with filledcu notitle lc rgb "#ff8888", table_acktr using 1:2 with lines title 'ACKTR' lc "#880000" dt 2, \
    table_ppo_opt using 1:3:4 with filledcu notitle lc rgb "#ff8888", table_ppo_opt using 1:2 with lines title 'PPO' lc "#ff0000", \
    table_bdqn_opt using 1:3:4 with filledcu notitle lc rgb "#8888ff", table_bdqn_opt using 1:2 with lines title 'BDQN' lc "#0000ff", \
    table_mimic using 1:3:4 with filledcu notitle lc rgb "#8888ff", table_mimic using 1:2 with lines title 'BDPI w/ AM' lc "#000088" dt 5, \
    table_aabdqn using 1:3:4 with filledcu notitle lc rgb "#888888", table_aabdqn using 1:2 with lines title ' ABCDQN (ours)' lc "#000000" dt 2, \
    table_ppi using 1:3:4 with filledcu notitle lc rgb "#888888", table_ppi using 1:2 with lines title 'BDPI (ours)' lc "#000000"

@V1MARGIN; @H0MARGIN; @NOXTICS; @YTICS
set key bottom right at 0.45,-40 maxrows 5 box linewidth 1 opaque
plot [0:0.5] [-50:110] \
    table_ppi_8critics_8loops   using 1:3:4 with filledcu notitle lc rgb "#888888", table_ppi_8critics_8loops   using 1:2 with lines title '8x8' lc "#000000" dt 5, \
    table_ppi_16critics_8loops  using 1:3:4 with filledcu notitle lc rgb "#888888", table_ppi_16critics_8loops  using 1:2 with lines title '16x8' lc "#000000" dt 2, \
    table_ppi_16critics_16loops using 1:3:4 with filledcu notitle lc rgb "#888888", table_ppi_16critics_16loops using 1:2 with lines title '16x16' lc "#000000", \
    table_ppi_32critics_8loops  using 1:3:4 with filledcu notitle lc rgb "#8888ff", table_ppi_32critics_8loops  using 1:2 with lines title '32x8' lc "#0000ff" dt 5, \
    table_ppi_32critics_16loops using 1:3:4 with filledcu notitle lc rgb "#8888ff", table_ppi_32critics_16loops using 1:2 with lines title '32x16' lc "#0000ff" dt 2, \
    table_ppi_32critics_32loops using 1:3:4 with filledcu notitle lc rgb "#8888ff", table_ppi_32critics_32loops using 1:2 with lines title '32x32' lc "#0000ff"

@V2MARGIN; @H0MARGIN; @XTICS; @YTICS
set xlabel "Episode (Table)"
set key bottom right at 0.45,-40 maxrows 5 box linewidth 1 opaque
plot [0:0.5] [-50:110] \
    table_ppi_16loops_1qloops using 1:3:4 with filledcu notitle lc rgb "#8888ff", table_ppi_16loops_1qloops using 1:2 with lines title '1 iteration' lc "#0000ff" dt 5, \
    table_ppi_16loops_2qloops using 1:3:4 with filledcu notitle lc rgb "#8888ff", table_ppi_16loops_2qloops using 1:2 with lines title '2 iterations' lc "#0000ff" dt 2, \
    table_ppi_16loops_4qloops using 1:3:4 with filledcu notitle lc rgb "#888888", table_ppi_16loops_4qloops using 1:2 with lines title '4 iterations' lc "#000000"

# LunarLander
set ytics ('-400' -400, '-200' -200, '0' 0, '200' 200)
@V0MARGIN; @H1MARGIN; @NOXTICS; @NOYTICS
set xlabel ""
set nokey
plot [0:1] [-560:300] \
    lunarlander_acktr using 1:3:4 with filledcu notitle lc rgb "#ff8888", lunarlander_acktr using 1:2 with lines title 'ACKTR' lc "#880000" dt 2, \
    lunarlander_ppo_opt using 1:3:4 with filledcu notitle lc rgb "#ff8888", lunarlander_ppo_opt using 1:2 with lines title 'PPO' lc "#ff0000", \
    lunarlander_bdqn_opt using 1:3:4 with filledcu notitle lc rgb "#8888ff", lunarlander_bdqn_opt using 1:2 with lines title 'BDQN' lc "#0000ff", \
    lunarlander_mimic using 1:3:4 with filledcu notitle lc rgb "#8888ff", lunarlander_mimic using 1:2 with lines title 'BDPI Actor-Mimic loss' lc "#000088" dt 5, \
    lunarlander_aabdqn using 1:3:4 with filledcu notitle lc rgb "#888888", lunarlander_aabdqn using 1:2 with lines title ' ABCDQN (ours)' lc "#000000" dt 2, \
    lunarlander_ppi using 1:3:4 with filledcu notitle lc rgb "#888888", lunarlander_ppi using 1:2 with lines title 'BDPI (ours)' lc "#000000"

@V1MARGIN; @H1MARGIN; @NOXTICS; @NOYTICS
set nokey
plot [0:1] [-560:300] \
    lunarlander_ppi_8critics_8loops   using 1:3:4 with filledcu notitle lc rgb "#888888", lunarlander_ppi_8critics_8loops   using 1:2 with lines title '8x8' lc "#000000" dt 5, \
    lunarlander_ppi_16critics_8loops  using 1:3:4 with filledcu notitle lc rgb "#888888", lunarlander_ppi_16critics_8loops  using 1:2 with lines title '16x8' lc "#000000" dt 2, \
    lunarlander_ppi_16critics_16loops using 1:3:4 with filledcu notitle lc rgb "#888888", lunarlander_ppi_16critics_16loops using 1:2 with lines title '16x16' lc "#000000", \
    lunarlander_ppi_32critics_8loops  using 1:3:4 with filledcu notitle lc rgb "#8888ff", lunarlander_ppi_32critics_8loops  using 1:2 with lines title '32x8' lc "#0000ff" dt 5, \
    lunarlander_ppi_32critics_16loops using 1:3:4 with filledcu notitle lc rgb "#8888ff", lunarlander_ppi_32critics_16loops using 1:2 with lines title '32x16' lc "#0000ff" dt 2, \
    lunarlander_ppi_32critics_32loops using 1:3:4 with filledcu notitle lc rgb "#8888ff", lunarlander_ppi_32critics_32loops using 1:2 with lines title '32x32' lc "#0000ff"

@V2MARGIN; @H1MARGIN; @XTICS; @NOYTICS
set xlabel "Episode (LunarLander)"
set nokey
plot [0:1] [-560:300] \
    lunarlander_ppi_16loops_1qloops using 1:3:4 with filledcu notitle lc rgb "#8888ff", lunarlander_ppi_16loops_1qloops using 1:2 with lines title '16x1' lc "#0000ff" dt 5, \
    lunarlander_ppi_16loops_2qloops using 1:3:4 with filledcu notitle lc rgb "#8888ff", lunarlander_ppi_16loops_2qloops using 1:2 with lines title '16x2' lc "#0000ff" dt 2, \
    lunarlander_ppi_16loops_4qloops using 1:3:4 with filledcu notitle lc rgb "#888888", lunarlander_ppi_16loops_4qloops using 1:2 with lines title '16x4' lc "#000000"

# FrozenLake
set ytics ('0.0' 0, '0.2' 0.2, '0.4' 0.4, '0.6' 0.6, '0.8' 0.8)
@V0MARGIN; @H2MARGIN; @NOXTICS; @NOYTICS
set nokey
plot [0:2] [0:1] \
    frozenlake_acktr using 1:3:4 with filledcu notitle lc rgb "#ff8888", frozenlake_acktr using 1:2 with lines title 'ACKTR' lc "#880000" dt 2, \
    frozenlake_ppo_opt using 1:3:4 with filledcu notitle lc rgb "#ff8888", frozenlake_ppo_opt using 1:2 with lines title 'PPO' lc "#ff0000", \
    frozenlake_bdqn_opt using 1:3:4 with filledcu notitle lc rgb "#8888ff", frozenlake_bdqn_opt using 1:2 with lines title 'BDQN' lc "#0000ff", \
    frozenlake_mimic using 1:3:4 with filledcu notitle lc rgb "#8888ff", frozenlake_mimic using 1:2 with lines title 'BDPI Actor-Mimic loss' lc "#000088" dt 5, \
    frozenlake_aabdqn using 1:3:4 with filledcu notitle lc rgb "#888888", frozenlake_aabdqn using 1:2 with lines title ' ABCDQN (ours)' lc "#000000" dt 2, \
    frozenlake_ppi using 1:3:4 with filledcu notitle lc rgb "#888888", frozenlake_ppi using 1:2 with lines title 'BDPI (ours)' lc "#000000"

@V1MARGIN; @H2MARGIN; @NOXTICS; @NOYTICS
set nokey
plot [0:2] [0:1] \
    frozenlake_ppi_8critics_8loops   using 1:3:4 with filledcu notitle lc rgb "#888888", frozenlake_ppi_8critics_8loops   using 1:2 with lines title '8x8' lc "#000000" dt 5, \
    frozenlake_ppi_16critics_8loops  using 1:3:4 with filledcu notitle lc rgb "#888888", frozenlake_ppi_16critics_8loops  using 1:2 with lines title '16x8' lc "#000000" dt 2, \
    frozenlake_ppi_16critics_16loops using 1:3:4 with filledcu notitle lc rgb "#888888", frozenlake_ppi_16critics_16loops using 1:2 with lines title '16x16' lc "#000000", \
    frozenlake_ppi_32critics_8loops  using 1:3:4 with filledcu notitle lc rgb "#8888ff", frozenlake_ppi_32critics_8loops  using 1:2 with lines title '32x8' lc "#0000ff" dt 5, \
    frozenlake_ppi_32critics_16loops using 1:3:4 with filledcu notitle lc rgb "#8888ff", frozenlake_ppi_32critics_16loops using 1:2 with lines title '32x16' lc "#0000ff" dt 2, \
    frozenlake_ppi_32critics_32loops using 1:3:4 with filledcu notitle lc rgb "#8888ff", frozenlake_ppi_32critics_32loops using 1:2 with lines title '32x32' lc "#0000ff"

@V2MARGIN; @H2MARGIN; @XTICS; @NOYTICS
set xlabel "Episode (FrozenLake)"
set nokey
plot [0:2] [0:1] \
    frozenlake_ppi_16loops_1qloops using 1:3:4 with filledcu notitle lc rgb "#8888ff", frozenlake_ppi_16loops_1qloops using 1:2 with lines title '16x1' lc "#0000ff" dt 5, \
    frozenlake_ppi_16loops_2qloops using 1:3:4 with filledcu notitle lc rgb "#8888ff", frozenlake_ppi_16loops_2qloops using 1:2 with lines title '16x2' lc "#0000ff" dt 2, \
    frozenlake_ppi_16loops_4qloops using 1:3:4 with filledcu notitle lc rgb "#888888", frozenlake_ppi_16loops_4qloops using 1:2 with lines title '16x4' lc "#000000"
