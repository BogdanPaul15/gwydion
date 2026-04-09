import pytest
import numpy as np
from gwydion.actions import MultiDiscreteAdapter, DiscreteAdapter, build_action_space

class TestMultiDiscreteAdapter:
    def test_decode(self):
        adapter = MultiDiscreteAdapter(num_deployments=2, num_actions=15)
        assert adapter.decode(np.array([0, 0])) == (0, 0)
        assert adapter.decode(np.array([0, 7])) == (0, 7)
        assert adapter.decode(np.array([1, 0])) == (1, 0)
        assert adapter.decode(np.array([1, 14])) == (1, 14)

    def test_encode(self):
        adapter = MultiDiscreteAdapter(num_deployments=2, num_actions=15)
        assert list(adapter.encode(0, 0)) == [0, 0]
        assert list(adapter.encode(0, 7)) == [0, 7]
        assert list(adapter.encode(1, 0)) == [1, 0]
        assert list(adapter.encode(1, 14)) == [1, 14]

    def test_roundtrip(self):
        adapter = MultiDiscreteAdapter(num_deployments=11, num_actions=15)
        for deployment in range(11):
            for action in range(15):
                assert adapter.decode(adapter.encode(deployment, action)) == (deployment, action)

    def test_gym_space_shape(self):
        assert list(MultiDiscreteAdapter(2, 15).gym_space.nvec) == [2, 15]

class TestDiscreteAdapter:
    def test_decode(self):
        adapter = DiscreteAdapter(num_deployments=2, num_actions=15)
        assert adapter.decode(0) == (0, 0)
        assert adapter.decode(7) == (0, 7)
        assert adapter.decode(15) == (1, 0)
        assert adapter.decode(22) == (1, 7)
        assert adapter.decode(29) == (1, 14)

    def test_encode(self):
        adapter = DiscreteAdapter(num_deployments=2, num_actions=15)
        assert adapter.encode(0, 0) == 0
        assert adapter.encode(0, 7) == 7
        assert adapter.encode(1, 0) == 15
        assert adapter.encode(1, 14) == 29

    def test_roundtrip(self):
        adapter = DiscreteAdapter(num_deployments=11, num_actions=15)
        for deployment in range(11):
            for action in range(15):
                assert adapter.decode(adapter.encode(deployment, action)) == (deployment, action)

    def test_gym_space_size(self):
        assert DiscreteAdapter(2, 15).gym_space.n == 30
        assert DiscreteAdapter(11, 15).gym_space.n == 165

class TestBuildActionSpace:
    def test_multi_discrete(self):
        assert isinstance(build_action_space("multi_discrete", 2, 15), MultiDiscreteAdapter)

    def test_discrete(self):
        assert isinstance(build_action_space("discrete", 2, 15), DiscreteAdapter)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown action space type"):
            build_action_space("other", 2, 15)
