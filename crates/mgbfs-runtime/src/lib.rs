//! Native scheduling contracts. CPU models are verification tools, not GPU fallbacks.
pub mod admitted_buffers;
pub mod archive;
pub mod benchmark;
pub mod bootstrap;
pub mod reference_launch;
pub mod session_protocol;
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub mod session_worker;
#[cfg(feature = "cuda")]
mod session_cache;
pub mod byte_admission;
mod control_admission;
pub mod control_connection;
pub mod control_handshake;
pub mod control_outbox;
pub mod control_pump;
pub mod control_wire;
#[cfg(feature = "cuda")]
pub mod dense_device;
pub mod dense_frames;
pub mod distributed_memory;
#[cfg(feature = "cuda")]
pub mod distributed_native;
#[cfg(feature = "cuda")]
mod batch_graph;
pub mod epoch_coordinator;
pub mod event_generation;
pub mod exchange;
pub mod failure;
pub mod group_commit;
#[cfg(feature = "cuda")]
pub mod hash_first_exchange;
pub mod jobs;
#[cfg(feature = "library-owner")]
pub mod library_native;
pub mod library_owner;
pub mod macro_history_window;
#[cfg(feature = "cuda")]
pub mod macro_native;
pub mod macro_owner;
pub mod macro_simulation;
#[cfg(feature = "cuda")]
pub mod native;
pub mod owner;
pub mod owner_results;
pub mod parent_batches;
pub mod payload_lease;
#[cfg(feature = "cuda")]
pub mod pinned_archive;
pub mod rank_epochs;
pub mod receipts;
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub mod reference_bench;
pub mod ring;
pub mod route_count;
pub mod scatter_admission;
pub mod simulation;
pub mod source_banks;
pub mod topology;
pub mod transport;

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub mod cuda_loading;

pub mod generic_memory;
mod generic_gemm;

#[cfg(feature="cuda")]
pub mod generic_native;

#[cfg(all(feature="cuda",target_os="linux"))]
pub mod generic_run;

pub mod generic_route;

pub mod generic_distributed_memory;

#[cfg(feature="cuda")]
pub mod generic_distributed_native;

#[cfg(all(feature="cuda",target_os="linux"))]
pub mod generic_distributed_run;

#[cfg(feature="cuda")]
mod generic_sorted_native;
