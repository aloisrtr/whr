#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub struct Game {
    pub p1: usize,
    pub p2: usize,
    pub winner: Option<usize>,
    pub handicap: f64,
}
impl Game {
    pub fn opponent(&self, player: usize) -> usize {
        if player == self.p1 {
            self.p2
        } else {
            self.p1
        }
    }
}
