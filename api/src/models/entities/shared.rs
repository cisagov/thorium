//! Contains models shared between multiple entity kinds

use strum::EnumString;

/// A critical sector that an entity is associated with
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    EnumString,
    strum::Display,
)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum CriticalSector {
    Chemical,
    CommercialFacilities,
    Communications,
    CriticalManufacturing,
    Dams,
    DefenseIndustrialBase,
    EmergencyServices,
    Energy,
    FinancialServices,
    FoodAgriculture,
    GovernmentServicesFacilities,
    HealthcarePublicHealth,
    InformationTechnology,
    NuclearReactorsMaterialsWaste,
    TransportSystems,
    WaterWasteWater,
}
