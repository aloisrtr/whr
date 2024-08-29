#[derive(Clone, PartialEq, PartialOrd)]
pub struct TimeStep {
    pub timestep: usize,

    pub won_games: Vec<usize>,
    pub lost_games: Vec<usize>,
    pub drawn_games: Vec<usize>,

    // Cached information
    pub won_game_terms: Vec<[f64; 4]>,
    pub lost_game_terms: Vec<[f64; 4]>,
    pub drawn_game_terms: Vec<[f64; 4]>,
}
impl TimeStep {
    pub fn new(time: usize) -> Self {
        Self {
            timestep: time,
            won_games: vec![],
            lost_games: vec![],
            drawn_games: vec![],
            won_game_terms: vec![],
            lost_game_terms: vec![],
            drawn_game_terms: vec![],
        }
    }
}
