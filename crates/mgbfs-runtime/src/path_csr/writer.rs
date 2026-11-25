use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use mgbfs_core::Result;

use super::TransitionSpool;

/// The caller submits a buffer only after CUDA has stopped writing to it.
/// Until the caller reclaims it, the writer owns the buffer and its credit.
pub(crate) struct TransitionWriter<T> {
    input: Option<SyncSender<T>>,
    output: Receiver<T>,
    worker: Option<JoinHandle<TransitionSpool>>,
    failure: Arc<OnceLock<String>>,
    slots: usize,
    outstanding: usize,
}

impl<T: Send + 'static> TransitionWriter<T> {
    pub(crate) fn new(
        spool: TransitionSpool,
        slots: usize,
        mut write: impl FnMut(&mut TransitionSpool, &mut T) -> Result<()> + Send + 'static,
    ) -> Result<Self> {
        if slots == 0 {
            return Err("PATH_LINEAGE_WRITER_SLOTS".into());
        }
        let (input, ready) = mpsc::sync_channel::<T>(slots);
        let (done, output) = mpsc::sync_channel(slots);
        let failure = Arc::new(OnceLock::new());
        let worker_failure = Arc::clone(&failure);
        let worker = thread::Builder::new()
            .name("mgbfs-csr-writer".into())
            .spawn(move || {
                let mut spool = spool;
                for mut item in ready {
                    if worker_failure.get().is_none() {
                        if let Err(error) = write(&mut spool, &mut item) {
                            let _ = worker_failure.set(error);
                        }
                    }
                    if done.send(item).is_err() {
                        break;
                    }
                }
                spool
            })
            .map_err(super::io_error)?;
        Ok(Self {
            input: Some(input),
            output,
            worker: Some(worker),
            failure,
            slots,
            outstanding: 0,
        })
    }

    /// A submission that fails retains ownership of the unchanged item.
    pub(crate) fn submit(&mut self, item: T) -> std::result::Result<(), (String, T)> {
        if let Some(error) = self.failure() {
            return Err((error, item));
        }
        if self.outstanding == self.slots {
            return Err(("PATH_LINEAGE_WRITER_FULL".into(), item));
        }
        let input = self.input.as_ref().expect("live writer has an input");
        match input.try_send(item) {
            Ok(()) => {
                self.outstanding += 1;
                Ok(())
            }
            Err(TrySendError::Full(item)) => Err(("PATH_LINEAGE_WRITER_FULL".into(), item)),
            Err(TrySendError::Disconnected(item)) => {
                Err(("PATH_LINEAGE_WRITER_DISCONNECTED".into(), item))
            }
        }
    }

    pub(crate) fn failure(&self) -> Option<String> {
        self.failure.get().cloned()
    }

    pub(crate) fn discard(&mut self) {
        let _ = self.failure.set("PATH_LINEAGE_WRITER_ABANDONED".into());
    }

    pub(crate) fn try_reclaim(&mut self) -> Option<T> {
        let item = self.output.try_recv().ok()?;
        self.outstanding -= 1;
        Some(item)
    }

    pub(crate) fn reclaim_timeout(&mut self, timeout: Duration) -> Option<T> {
        if self.outstanding == 0 {
            return None;
        }
        let item = self.output.recv_timeout(timeout).ok()?;
        self.outstanding -= 1;
        Some(item)
    }

    pub(crate) fn finish(mut self) -> Result<(TransitionSpool, Option<String>, Vec<T>)> {
        drop(self.input.take());
        // Every accepted item still holds one of `slots` credits until reclaimed.
        // The completion queue can therefore hold every result during the join.
        let worker = self.worker.take().expect("live writer has a worker");
        let spool = worker
            .join()
            .map_err(|_| "PATH_LINEAGE_WRITER_PANIC".to_owned())?;
        let returned: Vec<_> = self.output.try_iter().collect();
        if returned.len() != self.outstanding {
            return Err("PATH_LINEAGE_WRITER_LOST_CREDIT".into());
        }
        Ok((spool, self.failure.get().cloned(), returned))
    }
}

impl<T> Drop for TransitionWriter<T> {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            let _ = self.failure.set("PATH_LINEAGE_WRITER_ABANDONED".into());
            drop(self.input.take());
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path_csr::Transition;

    #[test]
    fn preserves_fifo_credits_and_returns_buffers_after_spool_failure() {
        let directory =
            std::env::temp_dir().join(format!("mgbfs-csr-writer-{}", std::process::id()));
        let spool = TransitionSpool::new(directory.clone(), 1024).unwrap();
        let (resume, paused) = mpsc::channel();
        let mut writer = TransitionWriter::new(spool, 3, move |spool, item| match *item {
            1 => {
                paused.recv_timeout(Duration::from_secs(5)).unwrap();
                spool.add(&mut [Transition {
                    child: 0,
                    parent_rank: 0,
                    parent: 0,
                    generator: 5,
                }])?;
                *item = 11;
                Ok(())
            }
            2 => Err("LINEAGE_LIMIT".into()),
            _ => panic!("write after first failure"),
        })
        .unwrap();
        assert!(writer.try_reclaim().is_none());
        for item in 1..=3 {
            writer.submit(item).unwrap();
        }
        assert_eq!(
            writer.submit(4),
            Err(("PATH_LINEAGE_WRITER_FULL".into(), 4))
        );
        resume.send(()).unwrap();
        assert_eq!(writer.reclaim_timeout(Duration::from_secs(5)), Some(11));
        let (spool, failure, returned) = writer.finish().unwrap();
        assert_eq!(failure.as_deref(), Some("LINEAGE_LIMIT"));
        assert_eq!(returned, [2, 3]);
        assert_eq!(spool.pending.len(), 1);
        drop(spool);
        assert!(!directory.exists());
    }
}
