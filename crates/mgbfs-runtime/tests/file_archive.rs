#![cfg(target_os = "linux")]
use mgbfs_runtime::archive::{verify, Archive, FileExtent};
#[test]
fn linux_archive_reserves_real_extent_and_never_overwrites_an_existing_file() {
    let path = std::env::temp_dir().join(format!(
        "mgbfs-archive-{}-{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let file = FileExtent::create_new(&path).unwrap();
    assert!(FileExtent::create_new(&path).is_err());
    let mut archive = Archive::new(file, 4096, 4, [17; 32]).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 4096);
    archive.records(0, &[1, 0, 0, 1], &[[1; 4]]).unwrap();
    archive.layer_commit(0, 1).unwrap();
    archive.run_commit().unwrap();
    drop(archive);
    verify(&std::fs::read(&path).unwrap()).unwrap();
    std::fs::remove_file(path).unwrap();
}
