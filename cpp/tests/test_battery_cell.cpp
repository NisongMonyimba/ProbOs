// =============================================================================
// cpp/tests/test_battery_cell.cpp
// Native GoogleTest tests for probos::kernel::BatteryCell
// =============================================================================

#include <gtest/gtest.h>
#include <cmath>
#include "kernel/battery_cell.hpp"

using namespace probos::kernel;

TEST(BatteryCellNominal, NominalParamsHasCorrectSize) {
    Param p = BatteryCell::nominal_params();
    EXPECT_EQ(p.size(), static_cast<size_t>(PARAM_DIM));
}

TEST(BatteryCellNominal, NominalParamsMatchKim2007) {
    Param p = BatteryCell::nominal_params();
    EXPECT_DOUBLE_EQ(p[P_EA_SEI], 1.3508e5);
    EXPECT_DOUBLE_EQ(p[P_A_SEI],  1.667e15);
    EXPECT_DOUBLE_EQ(p[P_M_CELL], 0.045);
    EXPECT_DOUBLE_EQ(p[P_T_ONSET], 403.15);
}

TEST(BatteryCellInitialState, HasCorrectSizeAndValues) {
    State s = BatteryCell::initial_state();
    EXPECT_EQ(s.size(), static_cast<size_t>(STATE_DIM));
    EXPECT_DOUBLE_EQ(s[T1], 403.15);
    EXPECT_DOUBLE_EQ(s[T2], 403.15);
    for (int idx : {C_SEI1, C_SEI2, C_AN1, C_AN2, C_CA1, C_CA2}) {
        EXPECT_DOUBLE_EQ(s[idx], 1.0);
    }
}

TEST(BatteryCellForwardStep, TemperatureIncreasesAtOnsetConditions) {
    State s = BatteryCell::initial_state();
    Param p = BatteryCell::nominal_params();
    State ns = BatteryCell::forward_step(s, p, 1.0);
    EXPECT_GT(ns[T1], s[T1]);
    EXPECT_GT(ns[T2], s[T2]);
}

TEST(BatteryCellForwardStep, ConcentrationsDecreaseOrStaySame) {
    State s = BatteryCell::initial_state();
    Param p = BatteryCell::nominal_params();
    State ns = BatteryCell::forward_step(s, p, 1.0);
    for (int idx : {C_SEI1, C_SEI2, C_AN1, C_AN2, C_CA1, C_CA2}) {
        EXPECT_LE(ns[idx], s[idx]);
    }
}

TEST(BatteryCellForwardStep, TemperatureNeverExceedsMaxBound) {
    State s = BatteryCell::initial_state();
    Param p = BatteryCell::nominal_params();
    State ns = s;
    for (int i = 0; i < 1000; ++i) {
        ns = BatteryCell::forward_step(ns, p, 10.0);
        EXPECT_LE(ns[T1], T_MAX);
        EXPECT_LE(ns[T2], T_MAX);
        EXPECT_GE(ns[T1], T_MIN);
        EXPECT_GE(ns[T2], T_MIN);
    }
}

TEST(BatteryCellForwardStep, ConcentrationsNeverGoNegativeOrAboveOne) {
    State s = BatteryCell::initial_state();
    Param p = BatteryCell::nominal_params();
    State ns = s;
    for (int i = 0; i < 1000; ++i) {
        ns = BatteryCell::forward_step(ns, p, 10.0);
        for (int idx : {C_SEI1, C_SEI2, C_AN1, C_AN2, C_CA1, C_CA2}) {
            EXPECT_GE(ns[idx], 0.0);
            EXPECT_LE(ns[idx], 1.0);
        }
    }
}

TEST(BatteryCellForwardStep, ZeroDtProducesNoChange) {
    State s = BatteryCell::initial_state();
    Param p = BatteryCell::nominal_params();
    State ns = BatteryCell::forward_step(s, p, 0.0);
    for (size_t i = 0; i < s.size(); ++i) {
        EXPECT_DOUBLE_EQ(ns[i], s[i]);
    }
}

TEST(BatteryCellForwardStep, DeterministicGivenSameInputs) {
    State s = BatteryCell::initial_state();
    Param p = BatteryCell::nominal_params();
    State ns1 = BatteryCell::forward_step(s, p, 1.0);
    State ns2 = BatteryCell::forward_step(s, p, 1.0);
    for (size_t i = 0; i < ns1.size(); ++i) {
        EXPECT_DOUBLE_EQ(ns1[i], ns2[i]);
    }
}
