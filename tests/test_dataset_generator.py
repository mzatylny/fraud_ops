import pytest

from src.dataset_generator import (
    SimulationConfig,
    generate_customer_profiles,
    generate_terminal_profiles,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_customers": 0},
        {"n_terminals": -1},
        {"n_days": 6},
        {"radius": 0},
        {"radius": True},
        {"random_state": -1},
        {"max_transactions": 0},
        {"target_max_fraud_rate": 1.0},
        {"start_date": "not-a-date"},
    ],
)
def test_simulation_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SimulationConfig(**kwargs)


def test_profile_generation_is_reproducible():
    customers_a = generate_customer_profiles(10, random_state=7)
    customers_b = generate_customer_profiles(10, random_state=7)
    terminals_a = generate_terminal_profiles(12, random_state=7)
    terminals_b = generate_terminal_profiles(12, random_state=7)
    assert customers_a.equals(customers_b)
    assert terminals_a.equals(terminals_b)
