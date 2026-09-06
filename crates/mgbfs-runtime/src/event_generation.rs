//! Generation-bound event lifecycle; caller supplies native record/query calls.
use mgbfs_core::Result;
#[derive(Default)]
pub struct EventGeneration {
    active: Option<u64>,
    retired: Option<u64>,
    ready: bool,
    failed: bool,
}
impl EventGeneration {
    fn apply<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if self.failed {
            return Err("EVENT_FAILED".into());
        }
        let result = f(self);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    /// Submit native EventRecord after this generation's protected writes.
    /// No later recording of the underlying handle may bypass this guard.
    pub fn record(&mut self, generation: u64, submit: impl FnOnce() -> Result<()>) -> Result<()> {
        self.apply(|s| {
            if s.active.is_some() || s.retired.is_some_and(|old| generation <= old) {
                return Err("EVENT_GENERATION".into());
            }
            submit()?;
            s.active = Some(generation);
            s.ready = false;
            Ok(())
        })
    }
    /// Native query maps success to true, not-ready to false, all other statuses
    /// to errors. One poll never waits for GPU work or allocates on success.
    pub fn poll(&mut self, generation: u64, query: impl FnOnce() -> Result<bool>) -> Result<bool> {
        self.apply(|s| {
            if s.active != Some(generation) {
                return Err("EVENT_GENERATION".into());
            }
            if !s.ready {
                s.ready = query()?;
            }
            Ok(s.ready)
        })
    }
    /// Completion alone does not free data: the caller must first retire all
    /// consumers/stream waits referring to this recorded generation.
    pub fn retire(&mut self, generation: u64) -> Result<()> {
        self.apply(|s| {
            if s.active != Some(generation) || !s.ready {
                return Err("EVENT_NOT_RETIRED".into());
            }
            s.retired = s.active.take();
            s.ready = false;
            Ok(())
        })
    }
}
