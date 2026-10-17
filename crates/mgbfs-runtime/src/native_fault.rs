//! Per-thread native failure context. No new CUDA synchronization is introduced.
use std::cell::RefCell;
use mgbfs_core::Result;

struct Context {
    rank: Option<u32>, device: Option<u32>, layer: Option<u64>, completed: Option<u64>,
    call: &'static str, source: &'static str, line: u32, ordinal: u64,
    phase: String, state: &'static str, error: Option<serde_json::Value>, last_error: Option<serde_json::Value>, secondary_errors: u64, trace: bool,
}
impl Default for Context {
    fn default() -> Self { Self {
        rank: None, device: None, layer: None, completed: None, call: "unlabelled",
        source: "unknown", line: 0, ordinal: 0, phase: "preparing".into(),
        state: "idle", error: None, last_error: None, secondary_errors: 0, trace: false,
    }}
}
thread_local! { static CONTEXT: RefCell<Context> = RefCell::new(Context::default()); }

pub fn reset(rank: Option<u32>, device: Option<u32>) {
    CONTEXT.with(|v| *v.borrow_mut() = Context {
        rank, device, trace: std::env::var("MGBFS_TRACE_NATIVE_CALLS").as_deref() == Ok("1"),
        ..Context::default()
    });
}
pub fn failed() -> bool { CONTEXT.with(|v| v.borrow().error.is_some()) }
pub fn phase(value: &str) { CONTEXT.with(|v| v.borrow_mut().phase = value.into()); }
pub fn layer(target: u64, completed: u64) {
    CONTEXT.with(|v| { let mut v = v.borrow_mut(); v.layer = Some(target); v.completed = Some(completed); });
    persist();
}
pub fn completed(value: u64) { CONTEXT.with(|v| v.borrow_mut().completed = Some(value)); }
pub fn snapshot() -> serde_json::Value {
    CONTEXT.with(|v| { let v = v.borrow(); serde_json::json!({
        "rank":v.rank,"device":v.device,"layer":v.layer,"last_completed_layer":v.completed,
        "native_call":v.call,"native_call_source":v.source,"native_call_line":v.line,
        "native_call_sequence":v.ordinal,"native_call_state":v.state,"native_error":v.error,
        "native_last_error":v.last_error,"native_secondary_error_count":v.secondary_errors,
    }) })
}
fn persist() {
    #[cfg(all(feature="cuda",target_os="linux"))]
    if let Err(error) = crate::job_phase::checkpoint() { eprintln!("MGBFS_TELEMETRY_WRITE_FAILED {error}"); }
}
pub fn begin(call: &'static str, source: &'static str, line: u32) {
    let trace = CONTEXT.with(|v| { let mut v = v.borrow_mut();
        v.call = call; v.source = source; v.line = line; v.ordinal += 1; v.state = "entered"; v.trace
    });
    if trace { eprintln!("MGBFS_NATIVE_CALL {}", snapshot()); persist(); }
}
pub fn finish(code: i32) -> Result<()> {
    if code == 0 {
        let trace = CONTEXT.with(|v| { let mut v = v.borrow_mut(); v.state = "returned"; v.trace });
        if trace { eprintln!("MGBFS_NATIVE_CALL {}", snapshot()); persist(); }
        return Ok(());
    }
    let error = CONTEXT.with(|v| { let mut v = v.borrow_mut();
        let error = serde_json::json!({"code":code,"call":v.call,"rank":v.rank,"device":v.device,
            "layer":v.layer,"last_completed_layer":v.completed,"phase":v.phase,
            "source":v.source,"line":v.line,"call_sequence":v.ordinal,
            "interpretation":"API reporting the error; asynchronous origin may precede this call"});
        if v.error.is_none() { v.error = Some(error.clone()); }
        else { v.secondary_errors = v.secondary_errors.saturating_add(1); }
        v.last_error = Some(error.clone()); v.state = "failed"; error
    });
    eprintln!("MGBFS_NATIVE_ERROR {error}"); persist();
    let display = |name: &str| error[name].as_u64().map(|v|v.to_string()).unwrap_or_else(||"unknown".into());
    Err(format!("GENERIC_CUDA_{code} call={} rank={} device={} layer={} last_completed_layer={} source={}:{} call_sequence={}",
        error["call"].as_str().unwrap_or("unknown"),display("rank"),display("device"),display("layer"),display("last_completed_layer"),
        error["source"].as_str().unwrap_or("unknown"),display("line"),display("call_sequence")))
}
