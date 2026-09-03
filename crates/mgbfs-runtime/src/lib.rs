//! Native scheduling contracts. CPU models are verification tools, not GPU fallbacks.
pub mod archive;
#[cfg(feature = "cuda")]
pub mod dense_device;
pub mod exchange;
pub mod jobs;
#[cfg(feature = "cuda")]
pub mod native;
pub mod owner;
#[cfg(feature = "cuda")]
pub mod pinned_archive;
pub mod receipts;
pub mod ring;
pub mod simulation;
pub mod transport;
