//! # Gymnarium Random Agents
//!
//! `gymnarium_agents_random` contains agents for the `gymnarium_libraries`
//! choosing their actions based on randomness.

extern crate gymnarium_base;
extern crate num_traits;
extern crate rand;
extern crate rand_chacha;

use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use gymnarium_base::{ActionSpace, Agent, AgentAction, EnvironmentState, Seed};

use num_traits::{Float, PrimInt};

use rand::distributions::{Distribution, Uniform};
use rand::SeedableRng;

use rand_chacha::ChaCha20Rng;

/// Possible errors occurring within this library.
///
/// Currently there are none and I don't think some will be here in the future.
#[derive(Debug)]
pub enum RandomAgentError {}

impl std::fmt::Display for RandomAgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

impl std::error::Error for RandomAgentError {}

/// Agent which chooses his actions through random number generation.
pub struct RandomAgent<D: PrimInt + Debug, C: Float + Debug> {
    action_spaces: ActionSpace<D, C>,
    rng: Arc<Mutex<ChaCha20Rng>>,
}

impl<D: PrimInt + Debug, C: Float + Debug> RandomAgent<D, C> {
    /// Creates a new RandomAgent with the provided ActionSpace.
    pub fn with(action_spaces: ActionSpace<D, C>) -> Self {
        Self {
            action_spaces,
            rng: Arc::new(Mutex::new(ChaCha20Rng::from_entropy())),
        }
    }

    // Extracted because this struct will be implemented for various variations of PrimInt and Float
    fn internal_reset(&mut self, random_seed: Option<Seed>) -> Result<(), RandomAgentError> {
        if let Some(seed) = random_seed {
            self.rng = Arc::new(Mutex::new(ChaCha20Rng::from_seed(seed.into())));
        } else {
            self.rng = Arc::new(Mutex::new(ChaCha20Rng::from_entropy()));
        }
        Ok(())
    }
}

impl Agent<RandomAgentError, i64, f64> for RandomAgent<i64, f64> {
    fn reset(&mut self, random_seed: Option<Seed>) -> Result<(), RandomAgentError> {
        self.internal_reset(random_seed)
    }

    /// ```
    /// # use gymnarium_agents_random::RandomAgent;
    /// use gymnarium_base::{DimensionBoundaries, Seed, Agent, DimensionValue, ActionSpace, EnvironmentState, SpacePosition, DimensionValueI64F64};
    /// let mut random_agent = RandomAgent::with(ActionSpace::from(vec!(
    ///     DimensionBoundaries::DISCRETE { minimum: 1, maximum: 2 },
    ///     DimensionBoundaries::CONTINUOUS { minimum: 2.0, maximum: 2.0 }
    /// )));
    /// random_agent.reset(Some(Seed::from(0))).unwrap();
    /// let chosen_actions: Vec<DimensionValue<i64, f64>> = random_agent.choose_action(&EnvironmentState::default()).unwrap().into();
    /// assert_eq!(2, chosen_actions.len());
    /// assert_eq!(DimensionValue::DISCRETE(2), chosen_actions[0]);
    /// assert_eq!(DimensionValue::CONTINUOUS(2.0), chosen_actions[1]);
    /// ```
    fn choose_action(
        &mut self,
        _: &EnvironmentState<i64, f64>,
    ) -> Result<AgentAction<i64, f64>, RandomAgentError> {
        let rng_a = Arc::clone(&self.rng);
        let rng_b = Arc::clone(&self.rng);

        Ok(self.action_spaces.sample(
            &move |min, max| Uniform::new_inclusive(min, max).sample(&mut *rng_a.lock().unwrap()),
            &move |min, max| Uniform::new_inclusive(min, max).sample(&mut *rng_b.lock().unwrap()),
        ))
    }

    fn process_reward(
        &mut self,
        _: &EnvironmentState<i64, f64>,
        _: &EnvironmentState<i64, f64>,
        _: f64,
        _: bool,
    ) -> Result<(), RandomAgentError> {
        Ok(())
    }

    fn close(&mut self) -> Result<(), RandomAgentError> {
        Ok(())
    }
}
