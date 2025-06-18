# Whole History Rating
Implementation of Rémi Coulom's [Whole History Rating (WHR) algorithm](https://www.remi-coulom.fr/WHR/)
as a Rust library, with support for draws and handicaps.

WHR is used to rate players in competitive games, akin to the [Elo](https://en.wikipedia.org/wiki/Elo_rating_system) 
or [TrueSkill](https://en.wikipedia.org/wiki/TrueSkill) systems. It can estimate
the probability of winning between two players even if they have never competed
against one another. It is more accurate than other rating systems because it uses
the entire match history of players instead of updating estimations incrementally.
This comes at the cost of higher computational costs, but most game servers should
be able to support it without issue.

WHR is notably used in Go, Warcraft 3, Renju, and is even used in some sports!

## Future work
I hope to generalize the library further, notably by supporting games with:
- more than two players
- teams

It's also difficult for me to assess how well the structure of this library matches
the needs of larger projects. If you'd like to use it and have trouble making it
work without friction, feel free to open up an issue.

## Installation
The library can be used in any Cargo project by running:
```sh
cargo add whr
```
or by adding the following to your `Cargo.toml`:
```toml
[dependencies]
whr = "0.3"
```
## Usage
Whole History Rating, as the name suggests, derives ratings from the entire history
of each player. When using this library, this is done in two steps:
- building an history of the matches that have been played.
- using that information to compute the ratings, which can then be accessed.

### Building the history
This is done using the [builder pattern](https://rust-unofficial.github.io/patterns/patterns/creational/builder.html).
```ignorerust
use whr::{WhrBuilder, MatchRecord};

let whr = WhrBuilder::default()
  // Register matches, with:
  // - two named players,
  // - an optional winner, 
  // - a timestep,
  // - and optional handicap (first player advantage)
  .with_match(MatchRecord::new("alice", "bob", Some("bob"), 1, None)?)
  .with_match(MatchRecord::new("alice", "bob", None, 2, None)?)
  .with_match(MatchRecord::new("bob", "alice", Some("alice"), 2, None)?)

  // You can even add multiple games at once from an iterator
  .with_matches([
    MatchRecord::new("bob", "alice", Some("alice"), 1, None)?,
    MatchRecord::new("alice", "charlie", Some("charlie"), 4, None)?
  ]);
```

Timesteps are discrete values indicating an order for the matches. This may be matched
with tournaments, for example. `WhrBuilder` is generic over both the player's identification
(so that you can use whatever your infrastructure uses: strings, integers, etc)
and timesteps (as long as they are ordered values).

### Computing ratings
Once the matches have been added to the history, you can call `WhrBuilder::build`
to compute a `Whr` structure. This structure allows you to access individual players'
ratings over timesteps, along with the uncertainety of this rating.

The computation step can be configured within `WhrBuilder`, by setting the number
of iterations to stabilize ratings, a maximum duration, batch size, etc.

All of these parameters and the methods used to set them are described in the
[documentation](https://docs.rs/whr/latest/whr/struct.WhrBuilder.html).

