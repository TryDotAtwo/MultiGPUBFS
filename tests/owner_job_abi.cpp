#include "owner_job.h"
#include <cstddef>
#include <type_traits>
static_assert(std::is_standard_layout<MgbfsBucketJob>::value);
static_assert(sizeof(MgbfsOwnerRange) == 16);
static_assert(alignof(MgbfsOwnerRange) == 8);
static_assert(sizeof(MgbfsBucketJob) == 64);
static_assert(alignof(MgbfsBucketJob) == 64);
static_assert(offsetof(MgbfsBucketJob, bucket) == 0);
static_assert(offsetof(MgbfsBucketJob, lane) == 4);
static_assert(offsetof(MgbfsBucketJob, incoming) == 8);
static_assert(offsetof(MgbfsBucketJob, prev) == 24);
static_assert(offsetof(MgbfsBucketJob, curr) == 40);
static_assert(offsetof(MgbfsBucketJob, accepted_count) == 56);
static_assert(offsetof(MgbfsBucketJob, generation) == 60);
int main() { return 0; }
