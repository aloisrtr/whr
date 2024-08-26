use crate::WhrBuilder;

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

    println!(
        "{:?}",
        whr.get(&0)
            .unwrap()
            .read()
            .unwrap()
            .get_ratings()
            .collect::<Vec<_>>()
    );
    println!(
        "{:?}",
        whr.get(&1)
            .unwrap()
            .read()
            .unwrap()
            .get_ratings()
            .collect::<Vec<_>>()
    )
}
