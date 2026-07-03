// =============================================================================
// cpp/bindings/probos_bindings.cpp
// ProbOS Month 2 Week 6 -- pybind11 bindings
//
// Exposes the Week 4 C++ kernel (BatteryCell + MonteCarloEngineOMP) to
// Python as the `probos_cpp` extension module. This is the point where
// Month 1's "two parallel implementations" (Python MonteCarloEngine and
// C++ MonteCarloEngineOMP) stop being separate codebases and become one
// callable interface -- the C++ 7x-faster path becomes directly usable
// from Python without a subprocess or file-based handoff.
//
// DESIGN CHOICE: expose MCResult as a plain Python object with NumPy
// array attributes, not as opaque C++ objects. This keeps the Python
// side idiomatic (result.percentiles is a real (3, STATE_DIM) NumPy
// array, not a wrapped std::vector<double> requiring manual reshaping
// on every access) at the one-time cost of a copy from std::vector into
// a NumPy buffer at binding time. Given N is typically in the thousands
// and STATE_DIM=8, this copy is negligible next to the OpenMP MC run
// itself.
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "kernel/battery_cell.hpp"
#include "kernel/monte_carlo_omp.hpp"

namespace py = pybind11;
using namespace probos::kernel;

// -----------------------------------------------------------------------
// Helper: wrap a flat std::vector<double> as a NumPy array with the
// given shape, WITHOUT copying -- pybind11 keeps the vector alive via
// a capsule tied to the array's lifetime.
// -----------------------------------------------------------------------
static py::array_t<double> vector_to_numpy(
    std::vector<double> vec,
    std::vector<py::ssize_t> shape
) {
    auto* data_ptr = new std::vector<double>(std::move(vec));
    py::capsule free_when_done(data_ptr, [](void* p) {
        delete reinterpret_cast<std::vector<double>*>(p);
    });
    return py::array_t<double>(shape, data_ptr->data(), free_when_done);
}

PYBIND11_MODULE(probos_cpp, m) {
    m.doc() = "ProbOS C++ kernel bindings (Month 2 Week 6)";

    // ------------------------------------------------------------------
    // Module-level constants -- mirror python/src/battery_model.py's
    // state_dim/param_dim so callers can validate array shapes without
    // hardcoding 8 and 15 on the Python side.
    // ------------------------------------------------------------------
    m.attr("STATE_DIM") = STATE_DIM;
    m.attr("PARAM_DIM") = PARAM_DIM;

    // ------------------------------------------------------------------
    // BatteryCell -- static methods only, exposed as module functions
    // rather than a class (there is no per-instance state to bind).
    // ------------------------------------------------------------------
    m.def("battery_nominal_params",
        []() {
            Param p = BatteryCell::nominal_params();
            return py::array_t<double>(PARAM_DIM, p.data());
        },
        "Nominal battery parameters from Kim et al. (2007), shape (15,)"
    );

    m.def("battery_initial_state",
        []() {
            State s = BatteryCell::initial_state();
            return py::array_t<double>(STATE_DIM, s.data());
        },
        "Initial battery state (both cells at onset temp), shape (8,)"
    );

    m.def("battery_forward_step",
        [](py::array_t<double> state, py::array_t<double> params, double dt) {
            auto s_buf = state.unchecked<1>();
            auto p_buf = params.unchecked<1>();
            if (s_buf.shape(0) != STATE_DIM) {
                throw std::invalid_argument(
                    "state must have shape (8,), got shape (" +
                    std::to_string(s_buf.shape(0)) + ",)"
                );
            }
            if (p_buf.shape(0) != PARAM_DIM) {
                throw std::invalid_argument(
                    "params must have shape (15,), got shape (" +
                    std::to_string(p_buf.shape(0)) + ",)"
                );
            }
            State s{};
            Param p{};
            for (int i = 0; i < STATE_DIM; ++i) s[i] = s_buf(i);
            for (int i = 0; i < PARAM_DIM; ++i) p[i] = p_buf(i);
            State ns = BatteryCell::forward_step(s, p, dt);
            return py::array_t<double>(STATE_DIM, ns.data());
        },
        py::arg("state"), py::arg("params"), py::arg("dt"),
        "Single-particle Euler step. state: shape (8,), params: shape (15,)."
    );

    // ------------------------------------------------------------------
    // MCResult -- exposed as a lightweight class with NumPy-array
    // properties, reshaped from the flat std::vector<double> storage.
    // ------------------------------------------------------------------
    py::class_<MCResult>(m, "MCResult",
        "Result of a MonteCarloEngineOMP.run() call.")
        .def_property_readonly("final_state",
            [](const MCResult& r) {
                return vector_to_numpy(
                    r.final_state,
                    {static_cast<py::ssize_t>(r.n_particles), STATE_DIM}
                );
            },
            "shape (N, STATE_DIM) -- final state of every particle"
        )
        .def_property_readonly("percentiles",
            [](const MCResult& r) {
                return vector_to_numpy(
                    r.percentiles, {3, STATE_DIM}
                );
            },
            "shape (3, STATE_DIM) -- [P05, P50, P95] per state variable"
        )
        .def_property_readonly("convergence",
            [](const MCResult& r) {
                return vector_to_numpy(
                    r.convergence, {STATE_DIM}
                );
            },
            "shape (STATE_DIM,) -- sigma/sqrt(N) per state variable"
        )
        .def_readonly("n_particles", &MCResult::n_particles)
        .def_readonly("n_steps",     &MCResult::n_steps)
        .def_readonly("dt",          &MCResult::dt)
        .def_readonly("wall_time_ms", &MCResult::wall_time_ms)
        .def("__repr__", [](const MCResult& r) {
            return "<probos_cpp.MCResult N=" + std::to_string(r.n_particles) +
                   " n_steps=" + std::to_string(r.n_steps) +
                   " wall_time_ms=" + std::to_string(r.wall_time_ms) + ">";
        });

    // ------------------------------------------------------------------
    // MonteCarloEngineOMP -- the 7x-faster-than-Python-serial engine
    // from Week 4 Tuesday, now callable directly from Python.
    // ------------------------------------------------------------------
    py::class_<MonteCarloEngineOMP>(m, "MonteCarloEngineOMP",
        "OpenMP-parallel Monte Carlo engine for BatteryCell. "
        "7x faster than the pure-Python MonteCarloEngine on the same "
        "hardware (measured Week 4 Tuesday benchmark).")
        .def(py::init<int, int, double, int>(),
            py::arg("N"),
            py::arg("n_steps"),
            py::arg("dt") = 1.0,
            py::arg("N_threads") = 0,
            "N_threads=0 means use all available cores."
        )
        .def("run", &MonteCarloEngineOMP::run,
            py::arg("seed") = 42,
            "Draw parameters, advance N particles for n_steps, "
            "return an MCResult."
        )
        .def_property_readonly("N", &MonteCarloEngineOMP::N)
        .def_property_readonly("n_steps", &MonteCarloEngineOMP::n_steps)
        .def_property_readonly("N_threads", &MonteCarloEngineOMP::N_threads)
        .def("__repr__", [](const MonteCarloEngineOMP& e) {
            return "<probos_cpp.MonteCarloEngineOMP N=" +
                   std::to_string(e.N()) +
                   " n_steps=" + std::to_string(e.n_steps()) + ">";
        });
}
