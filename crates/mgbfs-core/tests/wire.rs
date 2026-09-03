use mgbfs_core::wire::*;
fn header() -> FrameHeader {
    FrameHeader {
        kind: FrameKind::Dense,
        run_tag: 0x0706050403020100,
        sequence: 9,
        batch: 10,
        depth: 11,
        source: 1,
        destination: 0,
        count: 3,
    }
}
fn expected() -> ExpectedFrame {
    ExpectedFrame {
        run_tag: 0x0706050403020100,
        sequence: 9,
        batch: 10,
        depth: 11,
        source: 1,
        destination: 0,
        world: 2,
        kind: FrameKind::Dense,
        max_records: 3,
        max_payload: 768,
        state_stride: 16,
    }
}
const FROZEN: [u8; 64] = [
    0x32, 0x42, 0x47, 0x4d, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 9, 0, 0, 0,
    0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0,
    0, 0, 0, 0, 0,
];
#[test]
fn frozen_frame_is_fieldwise_le_and_checks_session_before_payload() {
    assert_eq!(header().encode(16).unwrap(), FROZEN);
    assert_eq!(FrameHeader::decode(&FROZEN, &expected()).unwrap(), header());
    for offset in [0, 4, 8, 12, 16, 24, 32, 40, 44, 48, 52, 56, 63] {
        let mut bad = FROZEN;
        bad[offset] ^= 0x80;
        assert!(
            FrameHeader::decode(&bad, &expected()).is_err(),
            "offset={offset}"
        );
    }
    assert!(FrameHeader::decode(&FROZEN[..63], &expected()).is_err());
    let mut e = expected();
    e.max_payload = 767;
    assert!(FrameHeader::decode(&FROZEN, &e).is_err());
    e = expected();
    e.world = 1;
    assert!(FrameHeader::decode(&FROZEN, &e).is_err());
}
#[test]
fn layouts_account_each_plane_padding_and_empty_frames() {
    for (kind, want) in [
        (FrameKind::Dense, 1536),
        (FrameKind::HashFirst, 1280),
        (FrameKind::Request, 512),
        (FrameKind::Response, 768),
        (FrameKind::Receipt, 768),
    ] {
        let l = payload_layout(kind, 17, 32).unwrap();
        assert_eq!(l.bytes, want);
        assert_eq!(payload_layout(kind, 0, 32).unwrap().bytes, 0);
        assert!(l
            .planes
            .iter()
            .all(|p| p.offset % 256 == 0 && p.reserved % 256 == 0));
    }
    assert!(payload_layout(FrameKind::Dense, 1, 0).is_err());
    assert!(payload_layout(FrameKind::Dense, 1, 17).is_err());
    assert!(payload_layout(FrameKind::Response, u32::MAX, u64::MAX - 15).is_err());
}
#[test]
fn padding_and_actual_frame_length_are_validated() {
    let l = payload_layout(FrameKind::Dense, 3, 16).unwrap();
    let mut p = vec![0; 768];
    p[0] = 42;
    p[256] = 7;
    p[512] = 1;
    validate_payload(&p, &l).unwrap();
    for i in [48, 255, 268, 511, 560, 767] {
        p[i] = 1;
        assert!(validate_payload(&p, &l).is_err());
        p[i] = 0;
    }
    assert!(validate_payload(&p[..767], &l).is_err());
    let mut bad = l.clone();
    bad.planes[1].offset = 0;
    assert!(validate_payload(&p, &bad).is_err());
}
#[test]
fn origin_uses_absolute_parent_and_rejects_reserved_or_out_of_range_fields() {
    let o = OriginRef {
        source: 1,
        movement: 5,
        parent: 0x1716151413121110,
    };
    let frozen = [
        1, 0, 0, 0, 5, 0, 0, 0, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    ];
    assert_eq!(o.encode(), frozen);
    assert_eq!(OriginRef::decode(&frozen, 2, 6).unwrap(), o);
    let mut bad = frozen;
    bad[6] = 1;
    assert!(OriginRef::decode(&bad, 2, 6).is_err());
    assert!(OriginRef::decode(&frozen, 1, 6).is_err());
    assert!(OriginRef::decode(&frozen, 2, 5).is_err());
}
