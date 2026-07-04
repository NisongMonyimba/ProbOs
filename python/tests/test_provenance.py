"""
Tests for ProvenanceTracker and ProvenanceNode.

20 tests covering:
  - Recording param and state nodes
  - Ancestor queries
  - P05 particle query
  - P05 param values query
  - Serialisation
  - record_from_result convenience method
"""

from __future__ import annotations

import numpy as np
import pytest

from python.src.battery_model import BatteryModel2Cell
from python.src.monte_carlo import MCResult, MonteCarloEngine
from python.src.parameter_priors import build_battery_priors
from python.src.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model() -> BatteryModel2Cell:
    return BatteryModel2Cell()


@pytest.fixture(scope="module")
def result(model: BatteryModel2Cell) -> MCResult:
    priors = build_battery_priors()
    return MonteCarloEngine(
        model, priors, N=200, n_steps=10, seed=42
    ).run()


@pytest.fixture(scope="module")
def tracker(
    model: BatteryModel2Cell, result: MCResult
) -> ProvenanceTracker:
    t = ProvenanceTracker()
    t.record_from_result(result, model.param_names())
    return t


# ---------------------------------------------------------------------------
# TestProvenanceRecording
# ---------------------------------------------------------------------------

class TestProvenanceRecording:

    def test_record_param_returns_string(self) -> None:
        t = ProvenanceTracker()
        nid = t.record_param("Ea_SEI", 135080.0, particle_id=0)
        assert isinstance(nid, str)

    def test_record_param_node_id_format(self) -> None:
        t = ProvenanceTracker()
        nid = t.record_param("Ea_SEI", 135080.0, particle_id=7)
        assert nid == "param_Ea_SEI_p7"

    def test_record_state_returns_string(self) -> None:
        t = ProvenanceTracker()
        pid = t.record_param("Ea_SEI", 135080.0, particle_id=0)
        nid = t.record_state("T1", 403.15, timestep=5,
                             particle_id=0, parent_ids=[pid])
        assert isinstance(nid, str)

    def test_record_state_node_id_format(self) -> None:
        t = ProvenanceTracker()
        nid = t.record_state("T1", 403.15, timestep=5,
                             particle_id=3, parent_ids=[])
        assert nid == "state_T1_t5_p3"

    def test_record_from_result_correct_node_count(
        self, tracker: ProvenanceTracker,
        model: BatteryModel2Cell, result: MCResult
    ) -> None:
        # N=200 particles * (15 params + 8 states) = 4600 nodes
        expected = result.n_particles * (model.param_dim + model.state_dim)
        assert len(tracker) == expected

    def test_param_nodes_have_no_parents(
        self, model: BatteryModel2Cell, result: MCResult
    ) -> None:
        t = ProvenanceTracker()
        t.record_from_result(result, model.param_names())
        nid   = "param_Ea_SEI_p0"
        nodes = t.query_ancestors(nid)
        param_node = next(n for n in nodes if n.node_id == nid)
        assert param_node.parent_ids == []

    def test_state_nodes_have_param_parents(
        self, model: BatteryModel2Cell, result: MCResult
    ) -> None:
        t = ProvenanceTracker()
        t.record_from_result(result, model.param_names())
        nid   = f"state_state_0_t{result.n_steps}_p0"
        nodes = t.query_ancestors(nid)
        state_node = next(n for n in nodes if n.node_id == nid)
        assert len(state_node.parent_ids) == model.param_dim


# ---------------------------------------------------------------------------
# TestAncestorQuery
# ---------------------------------------------------------------------------

class TestAncestorQuery:

    def test_query_returns_list(
        self, tracker: ProvenanceTracker, result: MCResult
    ) -> None:
        nid  = f"state_state_0_t{result.n_steps}_p0"
        ancs = tracker.query_ancestors(nid)
        assert isinstance(ancs, list)

    def test_query_includes_target_node(
        self, tracker: ProvenanceTracker, result: MCResult
    ) -> None:
        nid  = f"state_state_0_t{result.n_steps}_p0"
        ancs = tracker.query_ancestors(nid)
        ids  = [n.node_id for n in ancs]
        assert nid in ids

    def test_query_includes_param_parents(
        self, tracker: ProvenanceTracker,
        model: BatteryModel2Cell, result: MCResult
    ) -> None:
        nid  = f"state_state_0_t{result.n_steps}_p0"
        ancs = tracker.query_ancestors(nid)
        # Should contain the state node + all 15 param parents = 16 nodes
        assert len(ancs) == model.param_dim + 1

    def test_query_nonexistent_node_returns_empty(
        self, tracker: ProvenanceTracker
    ) -> None:
        assert tracker.query_ancestors("nonexistent_node_id") == []

    def test_query_param_node_returns_only_itself(
        self, tracker: ProvenanceTracker
    ) -> None:
        nid  = "param_Ea_SEI_p0"
        ancs = tracker.query_ancestors(nid)
        assert len(ancs) == 1
        assert ancs[0].node_id == nid


# ---------------------------------------------------------------------------
# TestP05Query
# ---------------------------------------------------------------------------

class TestP05Query:

    def test_p05_returns_list(
        self, tracker: ProvenanceTracker, result: MCResult
    ) -> None:
        ids = tracker.query_particles_causing_p05(result, state_idx=0)
        assert isinstance(ids, list)

    def test_p05_count_near_5pct(
        self, tracker: ProvenanceTracker, result: MCResult
    ) -> None:
        ids = tracker.query_particles_causing_p05(result, state_idx=0)
        # Should be approximately 5% of N=200 -> ~10 particles
        assert 1 <= len(ids) <= 30

    def test_p05_all_values_le_threshold(
        self, tracker: ProvenanceTracker, result: MCResult
    ) -> None:
        """
        All returned particles must have T1 <= P05 threshold.
        P05 threshold = result.percentiles[0, -1, 0]
        """
        ids           = tracker.query_particles_causing_p05(result, state_idx=0)
        threshold     = result.percentiles[0, -1, 0]
        final_values  = result.trajectories[ids, -1, 0]
        assert np.all(final_values <= threshold + 1e-10)

    def test_p05_different_state_idx(
        self, tracker: ProvenanceTracker, result: MCResult
    ) -> None:
        ids0 = tracker.query_particles_causing_p05(result, state_idx=0)
        ids2 = tracker.query_particles_causing_p05(result, state_idx=2)
        # Different state variables -> different particle sets
        assert set(ids0) != set(ids2)

    def test_p05_param_values_returns_dict(
        self, tracker: ProvenanceTracker,
        model: BatteryModel2Cell, result: MCResult
    ) -> None:
        d = tracker.query_p05_param_values(result, model.param_names())
        assert isinstance(d, dict)
        assert "Ea_SEI" in d
        assert len(d) == model.param_dim


# ---------------------------------------------------------------------------
# TestSerialisation
# ---------------------------------------------------------------------------

class TestSerialisation:

    def test_to_dict_returns_dict(
        self, tracker: ProvenanceTracker
    ) -> None:
        assert isinstance(tracker.to_dict(), dict)

    def test_to_dict_has_n_nodes(
        self, tracker: ProvenanceTracker
    ) -> None:
        d = tracker.to_dict()
        assert "n_nodes" in d
        assert d["n_nodes"] == len(tracker)

    def test_to_dict_has_nodes_list(
        self, tracker: ProvenanceTracker
    ) -> None:
        d = tracker.to_dict()
        assert "nodes" in d
        assert isinstance(d["nodes"], list)

    def test_to_dict_nodes_have_required_fields(
        self, tracker: ProvenanceTracker
    ) -> None:
        d     = tracker.to_dict()
        nodes = d["nodes"]
        assert isinstance(nodes, list)
        node  = nodes[0]
        assert isinstance(node, dict)
        required = {
            "node_id", "node_type", "name",
            "value", "timestep", "particle_id", "parent_ids"
        }
        assert required.issubset(node.keys())

    def test_empty_tracker_to_dict(self) -> None:
        t = ProvenanceTracker()
        d = t.to_dict()
        assert d["n_nodes"] == 0
        assert d["nodes"] == []
