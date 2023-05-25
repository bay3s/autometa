from dataclasses import dataclass

from autometa.randomization.randomization_bound import RandomizationBound
from autometa.randomization.randomization_parameter import RandomizationParameter


@dataclass
class RandomizationBoundary:
    """
    Describes the boundary sampled during Auto DR by the parameter and bound .
    """

    parameter: RandomizationParameter
    bound: RandomizationBound
