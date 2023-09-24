use crate::math::mat::{RealToProjective, RealToReal};

/// Model space coordinate basis.
#[derive(Copy, Clone, Debug, Default)]
pub struct Model;

/// World space coordinate basis.
#[derive(Copy, Clone, Debug, Default)]
pub struct World;

/// View (camera) space coordinate basis.
#[derive(Copy, Clone, Debug, Default)]
pub struct View;

/// <abbr title="Normalized device coordinates">NDC</abbr> space coordinate basis.
#[derive(Copy, Clone, Debug, Default)]
pub struct Ndc;

/// Screen space coordinate basis.
#[derive(Copy, Clone, Debug, Default)]
pub struct Screen;

/// Mapping from model space to view space.
pub type ModelToView = RealToReal<3, Model, View>;

/// Mapping from model space to view space.
pub type ModelToProjective = RealToProjective<Model>;

/// Mapping from view space to projective space.
pub type ViewToProjective = RealToProjective<View>;

/// Mapping from NDC space to screen space.
pub type NdcToScreen = RealToReal<3, Ndc, Screen>;
