use mgbfs_core::wire::{FrameHeader, FrameKind};
use mgbfs_runtime::{
    control_wire::Plane, dense_frames::decode_prefix, scatter_admission::TicketKey,
};

#[test]
fn weighted_prefix_cannot_be_decoded_as_unit_cost_dense() {
    use mgbfs_core::wire::{FrameHeader, FrameKind};
    use mgbfs_runtime::dense_frames::decode_macro_prefix;
    let key = TicketKey { depth: 3, epoch: 9, source: 0, plane: Plane::Candidate, generation: 3 };
    let header = FrameHeader { kind: FrameKind::MacroDense, run_tag: 5, sequence: 9,
        batch: 3, depth: 7, source: 0, destination: 1, count: 2 };
    let mut prefix = [0u8; 256];
    prefix[..64].copy_from_slice(&mgbfs_core::wire::encode_macro_header(header, 3, 4, 16).unwrap());
    assert_eq!(decode_macro_prefix(&prefix, key, 5, 1, 2, 16, 2, 1024, 4).unwrap(), Some(header));
    assert!(decode_prefix(&prefix, key, 5, 1, 2, 16, 2, 1024).is_err());
}

#[test]
fn received_prefix_is_bound_to_ticket_and_exact_envelope_before_owner_use() {
    let key = TicketKey {
        depth: 9,
        epoch: 99,
        source: 0,
        plane: Plane::Candidate,
        generation: 123,
    };
    let header = FrameHeader {
        kind: FrameKind::Dense,
        run_tag: 7,
        sequence: 99,
        batch: 123,
        depth: 9,
        source: 0,
        destination: 1,
        count: 2,
    };
    let mut prefix = [0u8; 256];
    prefix[..64].copy_from_slice(&header.encode(16).unwrap());
    assert_eq!(
        decode_prefix(&prefix, key, 7, 1, 2, 16, 3, 1024).unwrap(),
        Some(header)
    );
    assert!(decode_prefix(&prefix, key, 7, 1, 2, 16, 3, 1280).is_err());
    assert!(decode_prefix(&prefix, key, 7, 0, 2, 16, 3, 1024).is_err());
    assert!(decode_prefix(
        &prefix,
        TicketKey { epoch: 100, ..key },
        7,
        1,
        2,
        16,
        3,
        1024
    )
    .is_err());
    assert!(decode_prefix(
        &prefix,
        TicketKey {
            generation: 124,
            ..key
        },
        7,
        1,
        2,
        16,
        3,
        1024
    )
    .is_err());
    assert!(decode_prefix(&prefix, key, 8, 1, 2, 16, 3, 1024).is_err());
    assert!(decode_prefix(&prefix, key, 7, 1, 2, 16, 1, 1024).is_err());
    assert!(decode_prefix(&prefix[..64], key, 7, 1, 2, 16, 3, 1024).is_err());
    prefix[70] = 1;
    assert!(decode_prefix(&prefix, key, 7, 1, 2, 16, 3, 1024).is_err());
}

#[test]
fn empty_destination_has_no_prefix_and_never_infers_rows_from_padding() {
    let key = TicketKey {
        depth: 0,
        epoch: 0,
        source: 0,
        plane: Plane::Candidate,
        generation: 0,
    };
    assert!(decode_prefix(&[], key, 7, 1, 2, 16, 3, 0)
        .unwrap()
        .is_none());
    assert!(decode_prefix(&[0; 256], key, 7, 1, 2, 16, 3, 0).is_err());
    assert!(decode_prefix(&[], key, 7, 1, 2, 16, 3, 256).is_err());
    assert!(decode_prefix(&[], key, 7, 1, 2, 17, 3, 0).is_err());
    let mut prefix = [0u8; 256];
    prefix[..64].copy_from_slice(
        &FrameHeader {
            kind: FrameKind::Dense,
            run_tag: 7,
            sequence: 0,
            batch: 0,
            depth: 0,
            source: 0,
            destination: 1,
            count: 0,
        }
        .encode(16)
        .unwrap(),
    );
    assert!(decode_prefix(&prefix, key, 7, 1, 2, 16, 3, 256).is_err());
}
