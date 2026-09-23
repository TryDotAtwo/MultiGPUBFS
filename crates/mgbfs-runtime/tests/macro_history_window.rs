use mgbfs_runtime::macro_history_window::MacroHistoryWindow;

#[test]
fn old_layer_is_readable_during_settlement_and_cannot_be_rebound_early() {
    let mut window = MacroHistoryWindow::new(2).unwrap();
    for depth in 0..4 {
        window.settled(depth).unwrap();
        window.publish(depth).unwrap();
        window.archive_copied(depth).unwrap();
    }
    assert_eq!(window.slot_depth(0).unwrap(), Some(0));
    assert_eq!(window.slot_depth(3).unwrap(), Some(3));
    window.hold_reader(0).unwrap();
    window.settled(4).unwrap();
    assert_eq!(window.publish(4).unwrap_err(), "MACRO_HISTORY_SLOT_BUSY");
    assert_eq!(window.slot_depth(0).unwrap(), Some(0));
    window.release_reader(0).unwrap();
    window.publish(4).unwrap();
    assert_eq!(window.slot_depth(0).unwrap(), Some(4));
    assert_eq!(window.slot_depth(1).unwrap(), Some(1));
    assert!(window.hold_reader(0).is_err());
    assert_eq!(window.hold_reader(4).unwrap(), 0);
}

#[test]
fn archive_copy_lease_and_depth_order_guard_reuse() {
    let mut window = MacroHistoryWindow::new(1).unwrap();
    assert!(window.publish(0).is_err());
    window.settled(0).unwrap();
    window.publish(0).unwrap();
    window.settled(1).unwrap();
    window.publish(1).unwrap();
    window.archive_copied(1).unwrap();
    window.settled(2).unwrap();
    assert_eq!(window.publish(2).unwrap_err(), "MACRO_HISTORY_SLOT_BUSY");
    window.archive_copied(0).unwrap();
    window.publish(2).unwrap();
    assert_eq!(window.slot_depth(0).unwrap(), Some(2));
    assert_eq!(window.slot_depth(1).unwrap(), Some(1));
    assert_eq!(
        window.archive_copied(0).unwrap_err(),
        "MACRO_HISTORY_STALE_DEPTH"
    );
    assert_eq!(window.settled(4).unwrap_err(), "MACRO_HISTORY_DEPTH_ORDER");
}

#[test]
fn checked_window_size_and_lease_underflow_are_explicit() {
    assert!(MacroHistoryWindow::new(0).is_err());
    assert!(MacroHistoryWindow::new(u32::MAX).is_err());
    let mut window = MacroHistoryWindow::new(1).unwrap();
    window.settled(0).unwrap();
    window.publish(0).unwrap();
    assert_eq!(
        window.release_reader(0).unwrap_err(),
        "MACRO_HISTORY_READER_UNDERFLOW"
    );
    window.archive_copied(0).unwrap();
    assert_eq!(
        window.archive_copied(0).unwrap_err(),
        "MACRO_HISTORY_ARCHIVE_UNDERFLOW"
    );
}
