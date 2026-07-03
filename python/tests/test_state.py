"""
Tests for python/src/state.py  --  the Model ABC.

We test the ABC itself plus a minimal concrete subclass
(TinyModel) that implements all four required methods.
TinyModel is defined only in this test file and is never
used outside it.  Its forward_batch does Euler integration
of a trivial 2-state linear ODE:
    dz/dt = -0.1 * z
so we can compute the expected output analytically.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from python.src.state import FloatArray, Model


# =============================================================================
# MINIMAL CONCRETE SUBCLASS FOR TESTING
# =============================================================================

class TinyModel(Model):
    """
    A minimal Model subclass used only in this test file.

    State: z = [z0, z1]                   state_dim = 2
    Params: [decay_rate, offset]           param_dim = 2
    ODE:    dz/dt = -decay_rate * z + offset
    Euler:  z_new = z + dt * (-decay_rate * z + offset)

    This is simple enough that we can compute the exact expected
    output by hand, which lets us verify forward_batch precisely.
    """

    @property
    def state_dim(self) -> int:
        return 2

    @property
    def param_dim(self) -> int:
        return 2

    def param_names(self) -> list[str]:
        return ["decay_rate", "offset"]

    def initial_state(self) -> FloatArray:
        # Both state variables start at 1.0
        return np.array([1.0, 1.0], dtype=np.float64)

    def forward_batch(
        self,
        state: FloatArray,   # shape (N, 2)
        params: FloatArray,  # shape (N, 2)
        dt: float,
    ) -> FloatArray:
        # Extract parameter columns using NumPy broadcasting.
        # params[:, 0] is decay_rate for all N particles at once.
        # params[:, 1] is offset for all N particles at once.
        decay = params[:, 0:1]   # shape (N, 1) -- broadcasts over state columns
        offset = params[:, 1:2]  # shape (N, 1)

        # Euler step: z_new = z + dt * (-decay * z + offset)
        # Every operation here works on the full (N, 2) array at once.
        result: FloatArray = state + dt * (-decay * state + offset)
        return result


# =============================================================================
# TEST CLASS 1: ABC CANNOT BE INSTANTIATED
# =============================================================================

class TestModelABC:
    """Tests that enforce the ABC contract."""

    def test_cannot_instantiate_Model_directly(self) -> None:
        """
        Model is abstract.  Trying to instantiate it directly must
        raise TypeError.  This is Python enforcing the interface contract.
        """
        with pytest.raises(TypeError):
            Model()  # type: ignore[abstract]

    def test_subclass_missing_state_dim_is_rejected(self) -> None:
        """
        A subclass that forgets to define state_dim cannot be instantiated.
        Python raises TypeError before any __init__ code runs.
        """
        class BrokenModel(Model):
            # state_dim is missing -- forgot to implement it
            @property
            def param_dim(self) -> int: return 1
            def param_names(self) -> list[str]: return ["p"]
            def initial_state(self) -> FloatArray:
                return np.array([0.0])
            def forward_batch(
                self, s: FloatArray, p: FloatArray, dt: float
            ) -> FloatArray:
                return s

        with pytest.raises(TypeError):
            BrokenModel()  # type: ignore[abstract]

    def test_subclass_missing_forward_batch_is_rejected(self) -> None:
        """
        A subclass that forgets forward_batch cannot be instantiated.
        """
        class BrokenModel(Model):
            @property
            def state_dim(self) -> int: return 1
            @property
            def param_dim(self) -> int: return 1
            def param_names(self) -> list[str]: return ["p"]
            def initial_state(self) -> FloatArray:
                return np.array([0.0])
            # forward_batch is missing

        with pytest.raises(TypeError):
            BrokenModel()  # type: ignore[abstract]

    def test_complete_subclass_can_be_instantiated(self) -> None:
        """
        A subclass that implements all four methods can be instantiated
        without error.
        """
        m = TinyModel()
        assert m is not None

    def test_repr_is_readable(self) -> None:
        """
        __repr__ must return a string containing the class name,
        state_dim, and param_dim.
        """
        m = TinyModel()
        r = repr(m)
        assert "TinyModel" in r
        assert "state_dim=2" in r
        assert "param_dim=2" in r


# =============================================================================
# TEST CLASS 2: PROPERTIES
# =============================================================================

class TestModelProperties:
    """Tests for state_dim, param_dim, param_names, initial_state."""

    def setup_method(self) -> None:
        """Create a TinyModel before each test."""
        self.model = TinyModel()

    def test_state_dim_is_correct(self) -> None:
        assert self.model.state_dim == 2

    def test_param_dim_is_correct(self) -> None:
        assert self.model.param_dim == 2

    def test_param_names_length_matches_param_dim(self) -> None:
        """param_names() must return exactly param_dim names."""
        names = self.model.param_names()
        assert len(names) == self.model.param_dim

    def test_param_names_are_strings(self) -> None:
        """Every element of param_names() must be a non-empty string."""
        for name in self.model.param_names():
            assert isinstance(name, str)
            assert len(name) > 0

    def test_param_names_content(self) -> None:
        assert self.model.param_names() == ["decay_rate", "offset"]

    def test_initial_state_shape(self) -> None:
        """initial_state() must return a 1-D array of length state_dim."""
        s0 = self.model.initial_state()
        assert s0.shape == (self.model.state_dim,)

    def test_initial_state_dtype(self) -> None:
        """initial_state() must return float64."""
        s0 = self.model.initial_state()
        assert s0.dtype == np.float64

    def test_initial_state_values(self) -> None:
        s0 = self.model.initial_state()
        assert_allclose(s0, [1.0, 1.0])


# =============================================================================
# TEST CLASS 3: VALIDATE_PARAMS
# =============================================================================

class TestValidateParams:
    """Tests for the validate_params() helper method."""

    def setup_method(self) -> None:
        self.model = TinyModel()

    def test_correct_shape_passes(self) -> None:
        """A (N, param_dim) array must pass without raising."""
        params = np.ones((10, 2), dtype=np.float64)
        self.model.validate_params(params)   # no exception

    def test_1d_array_raises_ValueError(self) -> None:
        """A 1-D array must raise ValueError with a helpful message."""
        params = np.ones(2)
        with pytest.raises(ValueError, match="2-D"):
            self.model.validate_params(params)

    def test_wrong_number_of_columns_raises_ValueError(self) -> None:
        """An array with wrong param_dim must raise ValueError."""
        params = np.ones((10, 5))   # 5 columns but param_dim == 2
        with pytest.raises(ValueError, match="2"):
            self.model.validate_params(params)

    def test_error_message_contains_class_name(self) -> None:
        """The error message must name the class so the user
        knows where it came from."""
        params = np.ones(2)
        with pytest.raises(ValueError, match="TinyModel"):
            self.model.validate_params(params)


# =============================================================================
# TEST CLASS 4: VALIDATE_STATE
# =============================================================================

class TestValidateState:
    """Tests for the validate_state() helper method."""

    def setup_method(self) -> None:
        self.model = TinyModel()

    def test_correct_shape_passes(self) -> None:
        state = np.ones((10, 2), dtype=np.float64)
        self.model.validate_state(state)   # no exception

    def test_1d_array_raises_ValueError(self) -> None:
        state = np.ones(2)
        with pytest.raises(ValueError, match="2-D"):
            self.model.validate_state(state)

    def test_wrong_number_of_columns_raises_ValueError(self) -> None:
        state = np.ones((10, 7))   # 7 columns but state_dim == 2
        with pytest.raises(ValueError, match="2"):
            self.model.validate_state(state)


# =============================================================================
# TEST CLASS 5: FORWARD_BATCH
# =============================================================================

class TestForwardBatch:
    """
    Tests for forward_batch().

    TinyModel implements: z_new = z + dt * (-decay * z + offset)
    With decay=0.1, offset=0.0, z=1.0, dt=1.0:
      z_new = 1.0 + 1.0 * (-0.1 * 1.0 + 0.0) = 0.9
    We can verify this analytically.
    """

    def setup_method(self) -> None:
        self.model = TinyModel()

    def test_output_shape_matches_input_shape(self) -> None:
        """forward_batch must return shape (N, state_dim)."""
        N = 50
        state = np.ones((N, 2), dtype=np.float64)
        params = np.ones((N, 2), dtype=np.float64)
        new_state = self.model.forward_batch(state, params, dt=0.1)
        assert new_state.shape == (N, self.model.state_dim)

    def test_output_dtype_is_float64(self) -> None:
        N = 10
        state = np.ones((N, 2), dtype=np.float64)
        params = np.ones((N, 2), dtype=np.float64)
        new_state = self.model.forward_batch(state, params, dt=0.1)
        assert new_state.dtype == np.float64

    def test_euler_step_is_mathematically_correct(self) -> None:
        """
        With decay=0.1, offset=0.0, z=1.0, dt=1.0:
          z_new = z + dt * (-decay * z + offset)
               = 1.0 + 1.0 * (-0.1 * 1.0 + 0.0)
               = 0.9
        """
        state = np.array([[1.0, 1.0]])           # shape (1, 2)
        params = np.array([[0.1, 0.0]])           # decay=0.1, offset=0.0
        new_state = self.model.forward_batch(state, params, dt=1.0)
        assert_allclose(new_state, [[0.9, 0.9]], rtol=1e-12)

    def test_offset_shifts_equilibrium(self) -> None:
        """
        With decay=0.1, offset=1.0, z=0.0, dt=1.0:
          z_new = 0.0 + 1.0 * (-0.1 * 0.0 + 1.0) = 1.0
        """
        state = np.array([[0.0, 0.0]])
        params = np.array([[0.1, 1.0]])           # decay=0.1, offset=1.0
        new_state = self.model.forward_batch(state, params, dt=1.0)
        assert_allclose(new_state, [[1.0, 1.0]], rtol=1e-12)

    def test_dt_scaling(self) -> None:
        """
        Halving dt should halve the step size.
        With decay=0.1, offset=0.0, z=1.0, dt=0.5:
          z_new = 1.0 + 0.5 * (-0.1 * 1.0) = 0.95
        """
        state = np.array([[1.0, 1.0]])
        params = np.array([[0.1, 0.0]])
        new_state = self.model.forward_batch(state, params, dt=0.5)
        assert_allclose(new_state, [[0.95, 0.95]], rtol=1e-12)

    def test_vectorised_over_N_particles(self) -> None:
        """
        Each of N particles can have different parameter values.
        Particle 0: decay=0.1, offset=0.0, z=1.0 -> z_new = 0.9
        Particle 1: decay=0.5, offset=0.0, z=1.0 -> z_new = 0.5
        """
        state = np.array([[1.0, 1.0],
                          [1.0, 1.0]], dtype=np.float64)
        params = np.array([[0.1, 0.0],
                           [0.5, 0.0]], dtype=np.float64)
        new_state = self.model.forward_batch(state, params, dt=1.0)
        assert_allclose(new_state[0], [0.9, 0.9], rtol=1e-12)
        assert_allclose(new_state[1], [0.5, 0.5], rtol=1e-12)

    def test_large_batch_N_5000(self) -> None:
        """
        N=5000 is the production batch size.
        All particles identical here -- just verify shape and no crash.
        """
        N = 5000
        state = np.ones((N, 2), dtype=np.float64)
        params = np.ones((N, 2), dtype=np.float64) * 0.1
        new_state = self.model.forward_batch(state, params, dt=0.01)
        assert new_state.shape == (N, 2)
        assert not np.any(np.isnan(new_state))
        assert not np.any(np.isinf(new_state))
