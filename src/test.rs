use crate::{Rating, WhrBuilder};

#[test]
fn test_whr_output() {
    env_logger::init();
    let whr = WhrBuilder::default()
        .with_game(0, 1, Some(1), 1, None)
        .with_game(0, 1, Some(0), 2, None)
        .with_game(0, 1, Some(0), 3, None)
        .with_game(0, 1, Some(0), 4, None)
        .with_game(0, 1, Some(0), 4, None)
        .with_iterations(50)
        .build();

    let result_0 = whr
        .get(&0)
        .unwrap()
        .iter()
        .map(|r| (r.timestep, r.elo().0.round() as i64))
        .collect::<Vec<_>>();
    let result_1 = whr
        .get(&1)
        .unwrap()
        .iter()
        .map(|r| (r.timestep, r.elo().0.round() as i64))
        .collect::<Vec<_>>();

    assert_eq!(&result_0, &[(1, 92), (2, 94), (3, 95), (4, 96)]);
    assert_eq!(&result_1, &[(1, -92), (2, -94), (3, -95), (4, -96)]);
}
