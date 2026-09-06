use mgbfs_runtime::{bootstrap::BootstrapRecord, control_handshake::RunIdentity};

fn record() -> BootstrapRecord {
    BootstrapRecord {
        world: 2,
        identity: RunIdentity {
            config_digest: [7; 32],
            run_id: [9; 16],
        },
        endpoint: "127.0.0.1:12345".parse().unwrap(),
        nccl_id: [11; 128],
    }
}

#[test]
fn publication_never_overwrites_existing_run() {
    let root = std::env::temp_dir().join(format!(
        "mgbfs-bootstrap-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let path = root.join("bootstrap");
    let original = record();
    original.publish(&path).unwrap();
    assert_eq!(
        BootstrapRecord::read(&path, 2, original.identity).unwrap(),
        original
    );
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        BootstrapRecord::decode(&bytes, 2, original.identity).unwrap(),
        original
    );
    let mut replacement = original.clone();
    replacement.nccl_id.fill(17);
    assert!(replacement.publish(&path).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    let malformed = root.join("malformed");
    std::fs::write(&malformed, &bytes[..199]).unwrap();
    assert!(BootstrapRecord::read(&malformed, 2, original.identity).is_err());
    let mut oversized = bytes.clone();
    oversized.push(0);
    std::fs::write(&malformed, &oversized).unwrap();
    assert!(BootstrapRecord::read(&malformed, 2, original.identity).is_err());
    for entry in std::fs::read_dir(&root).unwrap() {
        std::fs::remove_file(entry.unwrap().path()).unwrap();
    }
    std::fs::remove_dir(root).unwrap();
}

#[test]
fn frozen_record_preserves_endpoint_identity_and_nccl_id() {
    let r = record();
    let bytes = r.encode().unwrap();
    assert_eq!(&bytes[..16], b"MGBBOOT1\x01\x00\x00\x00\x02\x00\x00\x00");
    assert_eq!(&bytes[16..48], &[7; 32]);
    assert_eq!(&bytes[48..64], &[9; 16]);
    assert_eq!(&bytes[64..72], &[127, 0, 0, 1, 57, 48, 0, 0]);
    assert_eq!(&bytes[72..200], &[11; 128]);
    assert_eq!(BootstrapRecord::decode(&bytes, 2, r.identity).unwrap(), r);
}

#[test]
fn stale_identity_corrupt_header_and_nonlocal_endpoint_fail_closed() {
    let r = record();
    let bytes = r.encode().unwrap();
    for offset in [0, 8, 12, 16, 48, 64, 70, 71] {
        let mut bad = bytes;
        bad[offset] ^= 1;
        assert!(
            BootstrapRecord::decode(&bad, 2, r.identity).is_err(),
            "offset {offset}"
        );
    }
    assert!(BootstrapRecord::decode(&bytes[..199], 2, r.identity).is_err());
    assert!(BootstrapRecord::decode(&bytes, 3, r.identity).is_err());
    let mut invalid = r;
    invalid.endpoint = "0.0.0.0:12345".parse().unwrap();
    assert!(invalid.encode().is_err());
    invalid.endpoint = "127.0.0.1:0".parse().unwrap();
    assert!(invalid.encode().is_err());
}
