"""by lyuwenyu
"""

from .solver import BaseSolver
from .det_solver import DetSolver
from .det_solver_for_video import DetVideoSolver
from .det_solver_mm import DetSolverMM



from typing import Dict 

TASKS :Dict[str, BaseSolver] = {
    'detection': DetSolver,
    'detection video': DetVideoSolver,
    'detection mm': DetSolverMM,
}