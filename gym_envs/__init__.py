# This file is part of Bootstrapped Dual Policy Iteration
#
# Copyright 2018-2019, Vrije Universiteit Brussel (http://vub.ac.be)
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

from gym.envs.registration import registry, register, make, spec

register(
    id='LargeGrid-v0',
    entry_point='gym_envs.myGrid:myGrid',
    kwargs={'y': 29, 'x': 27}
)

register(
    id='Table-v0',
    entry_point='gym_envs.table:Table',
    kwargs={'rnd': False, 'backup': False}
)

register(
    id='TableBackup-v0',
    entry_point='gym_envs.table:Table',
    kwargs={'rnd': False, 'backup': True}
)

register(
    id='TableRandom-v0',
    entry_point='gym_envs.table:Table',
    kwargs={'rnd': True, 'backup': False}
)
