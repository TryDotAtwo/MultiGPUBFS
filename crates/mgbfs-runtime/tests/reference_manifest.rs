use mgbfs_runtime::reference_launch::load_matrix_manifest;

fn with_manifest(text: &str, check: impl FnOnce(&std::path::Path)) {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "mgbfs-manifest-{}-{nonce}.json",
        std::process::id()
    ));
    std::fs::write(&path, text).unwrap();
    check(&path);
    std::fs::remove_file(path).unwrap();
}

const VALID: &str = r#"{"schema":1,"rows":2,"cols":2,"modulus":3,
"start":[1,1,0,1],"generators":[[1,1,0,1],[1,2,0,1]],
"inverse_map":[1,0],"expected_max_unique_states":3}"#;

#[test]
fn loader_preserves_nonidentity_start_and_inverse_closed_generators() {
    with_manifest(VALID, |path| {
        let (label, graph) = load_matrix_manifest(path).unwrap();
        assert!(label.starts_with("matrix-"));
        assert_eq!(graph.start, [1, 1, 0, 1]);
        assert_eq!(graph.successor(&graph.start, 0).unwrap(), [1, 2, 0, 1]);
        assert_eq!(graph.successor(&graph.start, 1).unwrap(), [1, 0, 0, 1]);
    });
}

#[test]
fn identity_depends_on_validated_content_not_path_or_json_whitespace() {
    with_manifest(VALID, |first| {
        with_manifest(&format!("\n{VALID}\n"), |second| {
            assert_eq!(
                load_matrix_manifest(first).unwrap().0,
                load_matrix_manifest(second).unwrap().0
            );
        });
        with_manifest(
            &VALID.replace("\"start\":[1,1,0,1]", "\"start\":[1,0,0,1]"),
            |second| {
                assert_ne!(
                    load_matrix_manifest(first).unwrap().0,
                    load_matrix_manifest(second).unwrap().0
                );
            },
        );
    });
}

#[test]
fn invalid_manifest_never_substitutes_a_reference_graph() {
    for text in [
        "{",
        &VALID.replace("\"inverse_map\":[1,0]", "\"inverse_map\":[0,1]"),
        &VALID.replace("\"start\":[1,1,0,1]", "\"start\":[1,3,0,1]"),
    ] {
        with_manifest(text, |path| assert!(load_matrix_manifest(path).is_err()));
    }
}
