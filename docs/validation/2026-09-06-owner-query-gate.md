# Owner allocation query gate

Kaggle distributed-sanitizer v27 completed on two T4 devices at source
`e3fd4870d32aa3aae7117fa915eb44c77e8fc6a3`.
Downloaded and checked `owner-query.log` contains `BOUNDED_OWNER_QUERY_PASS`.
All ten runtime fixtures passed in plain and all four sanitizer modes;
the runtime logs have three zero-error summaries and one zero-hazard/error/
warning racecheck summary. The source-pinned COMPLETE summary contains all
twelve successful profile process smokes. Retained evidence is under
`test_results/distributed-sanitizer-v27/`.

This verifies CUDA linking and use of the owner query by the owner allocator.
It does not validate the later shared/owned ledger integration or admission
vote. Those changes are assigned to v28 with eleven runtime fixtures and
explicit device allocation/reserve output assertions.
