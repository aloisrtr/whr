use std::sync::{Arc, RwLock};

use crate::{
    player_info::PlayerInfo,
    timestep::{Rating, TimeStamp, TimeStep},
};

#[derive(Clone, Debug)]
pub struct Game<P> {
    pub time: TimeStamp,
    pub p1: P,
    pub p2: P,
    pub p1_timestep: Arc<RwLock<TimeStep<P>>>,
    pub p2_timestep: Arc<RwLock<TimeStep<P>>>,
    pub winner: Option<P>,
    pub p2_advantage: f64,
}
impl<P> Game<P>
where
    P: Eq + PartialEq + std::fmt::Display,
{
    pub fn opponent_adjusted_gamma_rating(&self, player: &P) -> Rating {
        let opponent_elo = if player == &self.p1 {
            self.p2_timestep.read().unwrap().elo_rating() + self.p2_advantage
        } else {
            self.p1_timestep.read().unwrap().elo_rating() - self.p2_advantage
        };
        10f64.powf(opponent_elo / 400f64)
    }
}
