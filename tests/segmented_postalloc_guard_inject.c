#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stddef.h>

/* Test-only: force one CUDA memory query immediately following a real
   segment allocation to report zero free bytes. The CUDA allocation itself
   is real; the injected query is a deterministic reserve-pressure signal. */
static __thread unsigned matched, events, armed, injected;
static const size_t hash_seq[] = {96,60,48,24};
static const size_t bitmap_seq[] = {60,24};

int cudaMalloc(void **p, size_t n) {
    typedef int (*fn_t)(void **,size_t);
    static fn_t real;
    if (!real) real=(fn_t)dlsym(RTLD_NEXT,"cudaMalloc");
    if (!real) return 999;
    int rc=real(p,n);
    const char *mode=getenv("MGBFS_TEST_POSTALLOC_BACKEND");
    if (rc || !mode || injected) return rc;
    const size_t *seq=strcmp(mode,"hash")==0?hash_seq:bitmap_seq;
    unsigned count=strcmp(mode,"hash")==0?4:2;
    if (n==seq[matched]) matched++;
    else matched=(n==seq[0])?1:0;
    if (matched==count) {
        matched=0; events++;
        const char *raw=getenv("MGBFS_TEST_POSTALLOC_EVENT");
        unsigned target=raw?(unsigned)strtoul(raw,0,10):2;
        if (events==target) armed=1;
    }
    return rc;
}

int cudaMemGetInfo(size_t *free_bytes,size_t *total_bytes) {
    typedef int (*fn_t)(size_t *,size_t *);
    static fn_t real;
    if (!real) real=(fn_t)dlsym(RTLD_NEXT,"cudaMemGetInfo");
    if (!real) return 999;
    int rc=real(free_bytes,total_bytes);
    if (!rc && ((armed && !injected) ||
                (injected && getenv("MGBFS_TEST_POSTALLOC_REPEAT")))) {
        if (!injected) {
            armed=0;injected=1;
            fprintf(stderr,"TEST_POSTALLOC_RESERVE_PRESSURE pid=%ld actual_free=%zu\n",(long)getpid(),*free_bytes);
            fflush(stderr);
        }
        *free_bytes=0;
    }
    return rc;
}
