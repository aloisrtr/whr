use crate::{MatchRecord, WhrBuilder};

#[test]
fn test_whr_output() {
    let whr = WhrBuilder::default()
        .with_matches([
            MatchRecord::new(0, 1, Some(1), 1, None).unwrap(),
            MatchRecord::new(0, 1, Some(0), 2, None).unwrap(),
            MatchRecord::new(0, 1, Some(0), 3, None).unwrap(),
            MatchRecord::new(0, 1, Some(0), 4, None).unwrap(),
            MatchRecord::new(0, 1, Some(0), 4, None).unwrap(),
        ])
        .build();

    let result_0: Vec<_> = whr
        .get_player_ratings(&0)
        .unwrap()
        .iter()
        .map(|r| (r.timestep, r.elo().round() as i64))
        .collect();
    let result_1: Vec<_> = whr
        .get_player_ratings(&1)
        .unwrap()
        .iter()
        .map(|r| (r.timestep, r.elo().round() as i64))
        .collect();

    assert_eq!(&result_0, &[(1, 92), (2, 94), (3, 95), (4, 96)]);
    assert_eq!(&result_1, &[(1, -92), (2, -94), (3, -95), (4, -96)]);
}
