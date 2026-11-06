//! Why a run was rejected.

use saans_scene::SceneError;
use std::fmt;

/// A rejected scene or configuration. Every variant names what is wrong;
/// a person is named `people[i]`, never by a scene id (§3.1).
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A §3.2 parameter has a value the simulator cannot honour.
    Config { param: String, reason: String },
    /// The configuration text is not JSON, or not a JSON object.
    ConfigJson(String),
    /// `saans_scene::Scene::validate` rejects the scene.
    Scene(Vec<SceneError>),
    /// One of the extra scene rejections of §3.1.
    SceneRule(String),
    /// A person cannot be placed or moved as configured (§8.2, §9).
    Person { index: usize, reason: String },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config { param, reason } => write!(f, "config `{param}`: {reason}"),
            Self::ConfigJson(e) => write!(f, "config JSON: {e}"),
            Self::Scene(errors) => {
                write!(f, "the scene is invalid:")?;
                for e in errors {
                    write!(f, " {e};")?;
                }
                Ok(())
            }
            Self::SceneRule(r) => write!(f, "scene rejected by the simulator: {r}"),
            Self::Person { index, reason } => write!(f, "people[{index}]: {reason}"),
        }
    }
}

impl std::error::Error for Error {}
