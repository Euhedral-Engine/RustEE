use super::*;

const MS: Duration = Duration::from_millis(1);

/// A recorder with a 10 ms window and alpha 0.5, whose window starts at the returned instant.
fn recorder(window: Option<Duration>) -> (FlowRecorder, Instant) {
    let mut recorder = FlowRecorder::with_window(window, 0.5);
    let base = Instant::now();
    recorder.reset_at(base);
    (recorder, base)
}

fn at(base: Instant, millis: u64) -> Instant {
    base + MS * millis as u32
}

#[test]
fn defaults() {
    let recorder = FlowRecorder::new();
    assert_eq!(recorder.window(), Some(DEFAULT_WINDOW));
    assert_eq!(recorder.alpha(), DEFAULT_ALPHA);
    assert!(!recorder.is_measured());
    assert_eq!(recorder.min_units(), None);
    assert_eq!(recorder.last_recording(), None);
}

#[test]
#[should_panic(expected = "alpha")]
fn alpha_must_be_a_weight() {
    FlowRecorder::with_window(None, 0.0);
}

#[test]
fn a_zero_window_is_unbounded() {
    assert_eq!(
        FlowRecorder::with_window(Some(Duration::ZERO), 0.5).window(),
        None
    );
}

#[test]
fn the_first_recording_only_starts_the_clock() {
    let (mut recorder, base) = recorder(Some(10 * MS));
    recorder.record_units_at(at(base, 1), 7);
    assert_eq!(recorder.last_recording(), Some(at(base, 1)));
    assert_eq!(recorder.last_recorded_units(), 7);
    assert_eq!(recorder.rolling_sum(), 7);
    assert_eq!(recorder.current_window_count(), 1);
    assert!(!recorder.is_measured());
}

#[test]
fn the_second_recording_initializes_the_statistics() {
    let (mut recorder, base) = recorder(Some(10 * MS));
    recorder.record_units_at(at(base, 1), 7);
    recorder.record_units_at(at(base, 3), 10);

    assert!(recorder.is_measured());
    assert_eq!(recorder.last_interval(), 2 * MS);
    assert_eq!(recorder.average_units(), 10.0);
    assert_eq!(recorder.average_interval(), 2e6);
    assert_eq!(recorder.min_units(), Some(10));
    assert_eq!(recorder.max_units(), Some(10));
    assert_eq!(recorder.min_interval(), Some(2_000_000));
    assert_eq!(recorder.min_units_over_time(), Some(10.0 / 2e6));
    assert_eq!(recorder.rolling_sum(), 14, "round(0.5 * 7 + 10)");
}

#[test]
fn the_rate_average_starts_from_its_first_sample() {
    let (mut recorder, base) = recorder(None);
    recorder.record_units_at(at(base, 0), 4);
    recorder.record_units_at(at(base, 2), 4);
    assert_eq!(recorder.average_units_over_time(), 0.0);
    recorder.record_units_at(at(base, 4), 4);
    assert_eq!(recorder.average_units_over_time(), 4.0 / 2e6);
}

#[test]
fn a_recording_no_later_than_the_last_is_ignored() {
    let (mut recorder, base) = recorder(Some(10 * MS));
    recorder.record_units_at(at(base, 2), 1);
    recorder.record_units_at(at(base, 2), 5);
    recorder.record_units_at(at(base, 1), 5);
    assert_eq!(recorder.last_recorded_units(), 1);
    assert_eq!(recorder.current_window_count(), 1);
}

#[test]
fn steady_samples_have_no_variance_or_trend() {
    let (mut recorder, base) = recorder(None);
    for step in 0..50 {
        recorder.record_units_at(at(base, step), 8);
    }
    assert_eq!(recorder.average_units(), 8.0);
    assert_eq!(recorder.average_interval(), 1e6);
    assert_eq!(recorder.unit_variance(), 0.0);
    assert_eq!(recorder.unit_trend(), 0.0);
    assert_eq!(recorder.unit_cv(), 0.0);
    assert_eq!(recorder.interval_cv(), 0.0);
    assert!((recorder.average_units_over_time() - 8e-6).abs() < 1e-15);
    assert_eq!(recorder.units_over_time_variance(), 0.0);
}

#[test]
fn averages_follow_a_step_and_trend_upward() {
    let (mut recorder, base) = recorder(None);
    for step in 0..10 {
        recorder.record_units_at(at(base, step), 10);
    }
    recorder.record_units_at(at(base, 10), 30);
    assert_eq!(recorder.average_units(), 20.0);
    assert!(recorder.unit_variance() > 0.0);
    assert!(recorder.unit_trend() > 0.0);
    assert_eq!(recorder.max_units(), Some(30));
    assert!(recorder.unit_cv() > 0.0);
    assert!(recorder.unit_standard_deviation() > 0.0);
}

#[test]
fn extremes_decay_to_within_three_standard_deviations() {
    let (mut recorder, base) = recorder(None);
    recorder.record_units_at(at(base, 0), 0);
    recorder.record_units_at(at(base, 1), 1_000);
    for step in 2..60 {
        recorder.record_units_at(at(base, step), 10);
    }
    let bound = recorder.average_units() + 3.0 * recorder.unit_standard_deviation();
    assert!((recorder.max_units().unwrap() as f64) <= bound.ceil() + 10.0);
    assert!(recorder.max_units().unwrap() < 1_000);
}

#[test]
fn an_interval_jump_beyond_the_window_restarts_the_recorder() {
    let (mut recorder, base) = recorder(Some(10 * MS));
    recorder.record_units_at(at(base, 0), 1);
    recorder.record_units_at(at(base, 1), 1);
    recorder.record_units_at(at(base, 2), 1);
    assert!(recorder.is_measured());

    recorder.record_units_at(at(base, 20), 1);
    assert!(!recorder.is_measured());
    assert_eq!(recorder.last_recording(), None);
    assert_eq!(recorder.current_window_count(), 0);

    recorder.record_units_at(at(base, 21), 3);
    assert_eq!(
        recorder.current_window_count(),
        1,
        "the next recording starts afresh"
    );
}

#[test]
fn without_a_window_jumps_are_absorbed() {
    let (mut recorder, base) = recorder(None);
    recorder.record_units_at(at(base, 0), 1);
    recorder.record_units_at(at(base, 1), 1);
    recorder.record_units_at(at(base, 1_000), 1);
    assert!(recorder.is_measured());
    assert_eq!(recorder.current_window_count(), 3);
    assert_eq!(recorder.effective_window_count_at(at(base, 5_000)), 3.0);
}

#[test]
fn windows_roll_over_and_carry_their_count() {
    let (mut recorder, base) = recorder(Some(10 * MS));
    for step in 0..=5 {
        recorder.record_units_at(at(base, step * 2), 1);
    }
    assert_eq!(recorder.current_window_count(), 6);
    recorder.record_units_at(at(base, 12), 1);
    assert_eq!(
        recorder.current_window_count(),
        0,
        "the window rolled at 12 ms"
    );
    recorder.record_units_at(at(base, 14), 1);
    assert_eq!(recorder.current_window_count(), 1);
}

#[test]
fn the_effective_count_blends_the_previous_window_by_what_remains() {
    let (mut recorder, base) = recorder(Some(10 * MS));
    for step in 0..=6 {
        recorder.record_units_at(at(base, step * 2), 1);
    }
    recorder.record_units_at(at(base, 14), 1);
    recorder.record_units_at(at(base, 16), 1);
    assert_eq!(recorder.current_window_count(), 2);

    assert_eq!(
        recorder.effective_window_count_at(at(base, 12)),
        7.0,
        "prev at the roll"
    );
    let halfway = recorder.effective_window_count_at(at(base, 17));
    assert!((halfway - (7.0 * 0.5 + 2.0)).abs() < 1e-9, "got {halfway}");
    let late = recorder.effective_window_count_at(at(base, 21));
    assert!((late - (7.0 * 0.1 + 2.0)).abs() < 1e-9, "got {late}");
    assert_eq!(
        recorder.effective_window_count_at(at(base, 23)),
        0.0,
        "stale"
    );
}

#[test]
fn the_effective_count_is_the_current_count_before_any_roll() {
    let (mut recorder, base) = recorder(Some(10 * MS));
    recorder.record_units_at(at(base, 1), 1);
    recorder.record_units_at(at(base, 2), 1);
    assert_eq!(recorder.effective_window_count_at(at(base, 3)), 2.0);
    assert_eq!(
        recorder.effective_window_count_at(base),
        0.0,
        "prev at the window start"
    );
}

#[test]
fn the_rolling_average_divides_the_decayed_sum_by_the_effective_count() {
    let (mut recorder, base) = recorder(Some(10 * MS));
    recorder.record_units_at(at(base, 1), 4);
    recorder.record_units_at(at(base, 2), 4);
    assert_eq!(recorder.rolling_sum(), 6);
    assert_eq!(recorder.rolling_average_at(at(base, 3)), 3.0);
    assert_eq!(recorder.rolling_average_at(at(base, 30)), 0.0);
}

#[test]
fn reset_forgets_everything() {
    let (mut recorder, base) = recorder(Some(10 * MS));
    for step in 0..5 {
        recorder.record_units_at(at(base, step), 3);
    }
    recorder.reset_at(at(base, 5));
    assert!(!recorder.is_measured());
    assert_eq!(recorder.average_units(), 0.0);
    assert_eq!(recorder.unit_variance(), 0.0);
    assert_eq!(recorder.rolling_sum(), 0);
    assert_eq!(recorder.current_window_count(), 0);
    assert_eq!(recorder.max_units(), None);
    assert_eq!(recorder.rolling_average_at(at(base, 6)), 0.0);
}

#[test]
fn a_clone_is_a_snapshot() {
    let (mut recorder, base) = recorder(None);
    recorder.record_units_at(at(base, 0), 2);
    recorder.record_units_at(at(base, 1), 2);
    let snapshot = recorder.clone();
    recorder.record_units_at(at(base, 2), 50);
    assert_eq!(snapshot.average_units(), 2.0);
    assert_ne!(recorder.average_units(), 2.0);
}

#[test]
fn the_minimum_rate_decays_back_toward_the_average() {
    let (mut recorder, base) = recorder(None);
    for step in 0..4 {
        recorder.record_units_at(at(base, step), 10);
    }
    recorder.record_units_at(at(base, 4), 0);
    assert_eq!(recorder.min_units_over_time(), Some(0.0));
    for step in 5..45 {
        recorder.record_units_at(at(base, step), 10);
    }
    assert!(recorder.min_units_over_time().unwrap() > 0.9e-5);
    assert!(recorder.max_units_over_time().unwrap() <= 1e-5 + 1e-12);
}

#[test]
fn a_fading_previous_window_still_counts_as_one_recording() {
    let (mut recorder, base) = recorder(Some(10 * MS));
    for step in 0..=6 {
        recorder.record_units_at(at(base, step * 2), 1);
    }
    assert_eq!(recorder.current_window_count(), 0);
    let fading = base + Duration::from_micros(21_500);
    assert_eq!(recorder.effective_window_count_at(fading), 1.0);
}
