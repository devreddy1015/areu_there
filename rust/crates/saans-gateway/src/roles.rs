//! Who may see what (SAANS_PLAN.md §10, ADR-005).
//!
//! Every output interface answers one question before it emits anything:
//! may this principal see this space, and at what level of detail? The
//! answer lives here, once, so the HTTP API, the WebSocket, MQTT and alert
//! messages cannot drift apart.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Configuration, consent, audit log, everything.
    Admin,
    /// Gateway and sensor health: links, calibration, metrics. Occupancy,
    /// but never vital signs or health-alert evidence.
    Ops,
    /// Lab-safety officers and wardens: full state and alerts for their
    /// spaces, vital signs only where the space has consented.
    Safety,
    /// Facilities and security: occupied / empty / unknown per space. No
    /// vital signs, no health alerts.
    Occupancy,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Admin => "admin",
            Role::Ops => "ops",
            Role::Safety => "safety",
            Role::Occupancy => "occupancy",
        }
    }
}

/// An authenticated user or API token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Principal {
    pub name: String,
    pub role: Role,
    /// Space ids this principal is limited to; `None` = all spaces.
    pub spaces: Option<BTreeSet<String>>,
}

impl Principal {
    /// The implicit principal when authentication is off (loopback only).
    pub fn local_admin() -> Self {
        Self { name: "local".into(), role: Role::Admin, spaces: None }
    }

    pub fn can_see_space(&self, space_id: &str) -> bool {
        self.spaces.as_ref().is_none_or(|s| s.contains(space_id))
    }

    /// Vital signs need both the space's consent and a role allowed to
    /// read them. Consent is checked by the privacy gate upstream of here
    /// too; this is the second, role-based lock.
    pub fn can_see_vitals(&self, space_id: &str, space_vitals: bool) -> bool {
        space_vitals && matches!(self.role, Role::Admin | Role::Safety) && self.can_see_space(space_id)
    }

    /// Health alerts (pauses, inactivity, collapse) and their evidence.
    pub fn can_see_health_alerts(&self, space_id: &str) -> bool {
        matches!(self.role, Role::Admin | Role::Safety) && self.can_see_space(space_id)
    }

    /// Acknowledge / resolve alerts.
    pub fn can_act_on_alerts(&self, space_id: &str) -> bool {
        self.can_see_health_alerts(space_id)
    }

    /// Links, sensor health, calibration, metrics.
    pub fn can_see_system(&self) -> bool {
        matches!(self.role, Role::Admin | Role::Ops)
    }

    pub fn can_see_audit(&self) -> bool {
        self.role == Role::Admin
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(role: Role, spaces: Option<&[&str]>) -> Principal {
        Principal { name: "x".into(), role, spaces: spaces.map(|s| s.iter().map(|x| x.to_string()).collect()) }
    }

    #[test]
    fn vitals_need_consent_role_and_scope() {
        assert!(p(Role::Safety, Some(&["lab3"])).can_see_vitals("lab3", true));
        assert!(!p(Role::Safety, Some(&["lab3"])).can_see_vitals("lab3", false), "no consent, no vitals");
        assert!(!p(Role::Safety, Some(&["lab3"])).can_see_vitals("lab4", true), "out of scope");
        assert!(!p(Role::Ops, None).can_see_vitals("lab3", true), "ops never sees vitals");
        assert!(!p(Role::Occupancy, None).can_see_vitals("lab3", true));
    }

    #[test]
    fn occupancy_role_sees_no_alerts_or_system() {
        let o = p(Role::Occupancy, None);
        assert!(!o.can_see_health_alerts("lab3"));
        assert!(!o.can_see_system());
        assert!(o.can_see_space("lab3"));
    }
}
