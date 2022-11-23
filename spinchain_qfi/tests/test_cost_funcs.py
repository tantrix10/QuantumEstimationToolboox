import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from lin_space import expect_stays_linear as lin_D
from exp_space import D as exp_D 
from numpy import isclose

def test_zero_con():
    assert isclose(lin_D(0,0,1,4), exp_D(0,0,1,4)).all()

@pytest.mark.parametrize("f, g", [(1,1),(1,0),(0,1), (0.001,0.5),(102,234)])
def test_con(f,g):
    assert isclose(lin_D(f,g,1,4),exp_D(f,g,1,4)).all()