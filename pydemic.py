"""Agent-base pandemic simulator in Python.

Based on an original idea by Georg Wohlfahrt and loosely adapted from his
MATLAB code.
"""

import time
from collections import Counter
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt


class GameMaster:
    """The GameMaster class handles all things related to luck, fate and politics.

    A little bit like God, if you like. There is only one instance per game.
    """

    def __init__(self, t0=0, seed=None, npix=100):
        """Instantiate the object.

        Parameters
        ----------
        t0 : int
            initial game time (should be 0 in most case)
        seed : int
            if you want to repeat random simulation results, you'll want
            to set a seed.
        npix : int
            domain size.
        """
        self.t = t0
        self.rng = default_rng(seed=seed)
        self.npix = npix

    def get_random_pos(self):
        """Get a random (x, y) position in the simulation domain.

        Returns
        -------
        tuple of (x, y) positions.
        """
        return tuple(self.rng.integers(low=0, high=self.npix, size=2, endpoint=True))

    def get_random_move(self, movemax):
        """Get a random agent move in the x and y direction.

        The agent is in charge of not walking out of the domain.

        Parameters
        ----------
        movemax : int
            the maximum number of pixels that the agent wants to travel.

        Returns
        -------
        [mx, my] a random move.
        """
        return self.rng.integers(low=-movemax, high=movemax, size=2, endpoint=True)

    def get_random_integer(self, low, high=None, size=None):
        """Get random integer(s) between bounds.

        Parameters
        ----------
        low : int
            Lowest (signed) integers to be drawn from the distribution
            (unless high=None, in which case this parameter is 0 and this value
            is used for high).
        high : int
            If provided, one above the largest (signed) integer to be drawn
            from the distribution (see above for behavior if high=None).
            If array-like, must contain integer values
        size : int or tuple of ints
            Output shape. If the given shape is, e.g., (m, n, k), then
            m * n * k samples are drawn. Default is None, in which case a
            single value is returned.

        Returns
        -------
        size-shaped array of random integers from the appropriate distribution,
        or a single such random int if size not provided.
        """
        return self.rng.integers(low, high=high, size=size)

    def get_random_float(self, size=None):
        """Return random floats in the half-open interval [0.0, 1.0).

        Parameters
        ----------
        size : int or tuple of ints
            Output shape. If the given shape is, e.g., (m, n, k), then
            m * n * k samples are drawn. Default is None, in which case a
            single value is returned.

        Returns
        -------
        Array of random floats of shape size (unless size=None, in which case
        a single float is returned).
        """
        return self.rng.random(size=size)


class Agent:
    """The Agent moves around and eventually gets sick.

    It all depends on the gamemaster.
    """

    def __init__(self,
                 gamemaster,
                 movemax=5,
                 time_until_recovery=10,
                 mortality_rate=0.01,
                 probability_of_infection=0.5,
                 ):

        self.gamemaster = gamemaster
        self.movemax = movemax
        self.immune = False
        self.infected = False
        self.deceased = False
        self.time_of_infection = None
        self.probability_of_infection = probability_of_infection
        self.time_until_recovery = time_until_recovery
        self.mortality_rate = mortality_rate
        self.pos = self.gamemaster.get_random_pos()

    def move(self):
        """We move in the domain

        Returns
        -------
        the (x, y) position.
        """

        if self.deceased:
            # Dead agents are out
            return -1, -1

        # Tell me where to go
        mx, my = self.gamemaster.get_random_move(self.movemax)
        # Abs is to reflect on the lower bound
        new_x = abs(self.pos[0] + mx)
        new_y = abs(self.pos[1] + my)
        # Reflect also on the higher bound
        if new_x >= self.gamemaster.npix:
            new_x = 2 * self.gamemaster.npix - new_x - 2
        if new_y >= self.gamemaster.npix:
            new_y = 2 * self.gamemaster.npix - new_y - 2
        self.pos = (new_x, new_y)
        return self.pos

    def contact(self):
        """Agent in contact with an infectious person. Decide what happens."""
        if self.immune or self.infected:
            # Nothing happens there
            return

        # Let fate decide
        v = self.gamemaster.get_random_float()
        if v < self.probability_of_infection:
            self.infected = True
            self.time_of_infection = self.gamemaster.t

    @property
    def contagious(self):
        """Is the agent contagious?"""
        if self.deceased or not self.infected:
            # Nope
            return False

        # Maybe
        if self.gamemaster.t > (self.time_of_infection + self.time_until_recovery):
            # Ok we are recovered
            self.infected = False
            self.immune = True
            # Maybe dead
            v = self.gamemaster.get_random_float()
            if v < self.mortality_rate:
                self.deceased = True

        return self.infected


def game(seed=None,  # Repeat random results?
         n_agents=10000,  # Number of agents in the domain
         npix=100,  # Domain size
         nt=100,  # number time steps (days) in the simulation
         stop_when_extinct=False,  # stop the simulation if the virus is extinct
         n_initial_infected=5,  # Number of initially infected agents
         movemax=5,  # Agents max movements
         time_until_recovery=10,  # Agents days until recovery after infection
         mortality_rate=0.01,  # Mortality rate at the end of the infection
         probability_of_infection=0.5,  # Probability of being infected if one other agent is infected
         ):
    """Run a pandemic simulation.

    Returns
    -------
    the gamemaster, with some results attached to it.
    """
    # Timer
    start_time = time.time()

    # Prepare the players
    gamemaster = GameMaster(seed=seed, npix=npix)
    agents = []
    for _ in range(n_agents):
        agents.append(Agent(gamemaster,
                            movemax=movemax,
                            time_until_recovery=time_until_recovery,
                            mortality_rate=mortality_rate,
                            probability_of_infection=probability_of_infection)
                      )

    # Fate: who is infected
    for idx in gamemaster.get_random_integer(0, n_agents-1, n_initial_infected):
        agents[idx].infected = True
        agents[idx].time_of_infection = gamemaster.t

    # Output
    gamemaster.n_infected = []
    gamemaster.n_deceased = []
    gamemaster.n_immune = []

    # Data containers (its faster to create them only once before the game)
    is_contagious = np.zeros(n_agents, dtype=bool)
    is_deceased = np.zeros(n_agents, dtype=bool)
    is_immune = np.zeros(n_agents, dtype=bool)

    # Lets go
    while gamemaster.t <= nt:

        pos_infected = []
        for i, a in enumerate(agents):
            p = a.move()
            c = a.contagious
            if c:
                pos_infected.append(p)

            is_contagious[i] = c
            is_deceased[i] = a.deceased
            is_immune[i] = a.immune

        pos_infected = Counter(pos_infected)
        for a in agents:
            if a.pos in pos_infected:
                # The probably to get infected is higher if there are more
                # agents infected in one cell
                for _ in range(pos_infected[a.pos]):
                    a.contact()

        gamemaster.n_infected.append(is_contagious.sum())
        gamemaster.n_deceased.append(is_deceased.sum())
        gamemaster.n_immune.append(is_immune.sum())

        print(f"Day {gamemaster.t:3d}. "
              f"N infected: {gamemaster.n_infected[-1]:5d}. "
              f"N immune: {gamemaster.n_immune[-1]:5d}. "
              f"N deceased: {gamemaster.n_deceased[-1]:5d}.")

        if stop_when_extinct and gamemaster.n_infected[-1] == 0:
            # Nothing left to compute
            break
        gamemaster.t += 1

    total_time = time.time() - start_time
    print(f"Simulation done. Total time: {total_time:.1f}s")

    return gamemaster


if __name__ == '__main__':

    # Demo game without measures and with measures
    gm1 = game(seed=1)
    gm2 = game(seed=1, movemax=2)

    plt.plot(gm1.n_infected, color='C0', label='Infected')
    plt.plot(gm2.n_infected, color='C0', linestyle='--')
    plt.plot(gm1.n_immune, color='C1', label='Immune')
    plt.plot(gm2.n_immune, color='C1', linestyle='--')
    plt.plot(gm1.n_deceased, color='C3', label='Deceased (cum)')
    plt.plot(gm2.n_deceased, color='C3', linestyle='--')
    plt.legend()
    plt.show()
