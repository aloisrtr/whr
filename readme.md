# Whole History Rating
Implementation of Rémi Coulom's [Whole History Rating (WHR) algorithm](https://www.remi-coulom.fr/WHR/)
as a Rust library.

It notably supports:
- handicaps (first player advantage)
- draws

WHR is used to rate players in competitive games, akin to the [Elo](https://en.wikipedia.org/wiki/Elo_rating_system) 
or [TrueSkill](https://en.wikipedia.org/wiki/TrueSkill) systems. It can estimate
the probability of winning between two players even if they have never competed
against one another. It is more accurate than say the Elo system, at the cost
of requiring more computation.

WHR is notably used in Go, Warcraft 3, Renju, and is even used in some sports!

## Future work
I hope to generalize the library further, notably by supporting:
- games with more than two players
- teams
as was stated to be possible in Rémi Coulom's paper introducing WHR.

## Installation
The library can be used in any Cargo project by running:
```sh
cargo add whr
```
or by adding the following to your `Cargo.toml`:
```toml
[dependencies]
whr = "0.1"
```
## Exemple usage
The library works following the [builder pattern](https://rust-unofficial.github.io/patterns/patterns/creational/builder.html).
```rust
use whr::WhrBuilder;

let whr = WhrBuilder::default()
  .with_iterations(50) // Maximum number of iterations for the algorithm to converge.
  .with_epsilon(1e-5) // Target stability between iterations
  // Register games, with:
  // - two named players,
  // - an optional winner, 
  // - a timestep,
  // - and optional handicap (first player advantage)
  .with_game("alice", "bob", Some("bob"), 1, None)
  .with_game("alice", "bob", None, 2, None)
  .with_game("bob", "alice", Some("alice"), 2, None)
  .build();
```
See the [documentation]() for more parameters and information.

## Benchmarks

