from benchmarks.official_baselines.adapters import anomaly_transformer as adapter


def test_released_epoch_schedule_preserves_first_two_epoch_rates():
    # Upstream changes the rate after each epoch using (epoch_number - 1).
    assert [adapter._epoch_learning_rate(1e-4, index) for index in range(4)] == [
        1e-4, 1e-4, 5e-5, 2.5e-5]
