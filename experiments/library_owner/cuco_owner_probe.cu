// Real indexed owner contract: stable committed keys, KEEP_FIRST and fail-fast.
#include "cuco_owner.cuh"
#include "cuco_rank_batch.cuh"
#include "owner_abi.h"
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <array>
#include <algorithm>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <set>
#include <type_traits>

template<class Ref>
__global__ void probe_dynamic_refs(Ref const* refs, uint32_t const* high_words,
    uint8_t* hits, uint32_t rows) {
  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += blockDim.x * gridDim.x) {
    uint32_t const shard = high_words[row] >> 31;
    hits[row] = refs[shard].contains(mgbfs::key_index(3, row));
  }
}

namespace {
using Key = std::array<uint32_t, 4>;
void require(bool value, char const* message) {
  if (!value) throw std::runtime_error(message);
}
void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
template<class F> void rejects(F function, char const* message) {
  bool rejected = false;
  try { function(); } catch (std::runtime_error const&) { rejected = true; }
  require(rejected, message);
}
struct Input {
  rmm::device_buffer words, sources;
  rmm::cuda_stream_view stream;
  uint32_t capacity, rows{0};
  Input(uint32_t size, rmm::cuda_stream_view s, rmm::device_async_resource_ref resource)
      : words(size * 16, s, resource), sources(size * 4, s, resource), stream(s), capacity(size) {}
  void upload(std::vector<Key> const& keys, std::vector<uint32_t> const& ids) {
    require(keys.size() == ids.size() && keys.size() <= capacity, "FIXTURE_INPUT");
    rows = static_cast<uint32_t>(keys.size());
    std::vector<uint32_t> plane(rows);
    for (unsigned c = 0; c < 4; ++c) {
      for (unsigned row = 0; row < rows; ++row) plane[row] = keys[row][c];
      check(cudaMemcpyAsync(static_cast<uint32_t*>(words.data()) + c * capacity,
          plane.data(), rows * 4, cudaMemcpyHostToDevice, stream.value()));
      stream.synchronize();
    }
    check(cudaMemcpyAsync(sources.data(), ids.data(), rows * 4,
        cudaMemcpyHostToDevice, stream.value()));
    stream.synchronize();
  }
  MgbfsLibraryKeysV1 keys() const {
    MgbfsLibraryKeysV1 view{};
    view.rows = rows;
    for (unsigned c = 0; c < 4; ++c)
      view.words[c] = static_cast<uint32_t const*>(words.data()) + c * capacity;
    return view;
  }
  MgbfsLibraryCandidatesV1 candidates() const {
    return {keys(), static_cast<uint32_t const*>(sources.data())};
  }
};
std::vector<uint32_t> read(MgbfsLibrarySurvivorsV1 result, rmm::cuda_stream_view stream) {
  std::vector<uint32_t> host(result.rows);
  if (result.rows) check(cudaMemcpyAsync(host.data(), result.source_indices,
      result.rows * 4, cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  return host;
}
std::vector<Key> read_keys(MgbfsLibraryKeysV1 keys, rmm::cuda_stream_view stream) {
  std::vector<Key> host(keys.rows);
  std::vector<uint32_t> plane(keys.rows);
  for (unsigned c = 0; c < 4; ++c) {
    if (keys.rows) check(cudaMemcpyAsync(plane.data(), keys.words[c], keys.rows * 4,
        cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();
    for (unsigned i = 0; i < keys.rows; ++i) host[i][c] = plane[i];
  }
  return host;
}
}
int main() {
  rmm::cuda_stream stream;
  constexpr size_t pool_bytes = 64 << 20;
  rmm::mr::cuda_memory_resource upstream;
  rmm::mr::pool_memory_resource pool{&upstream, pool_bytes, pool_bytes};
  rmm::mr::statistics_resource_adaptor stats{&pool};
  rmm::device_async_resource_ref resource{stats};
  {
    // Characterization gate for the rank-batch design: a GPU row picks one
    // persistent cuco ref by hash prefix, with shared candidate planes.
    Input history0(1, stream.view(), resource), history1(1, stream.view(), resource);
    Input candidates(3, stream.view(), resource);
    Key a{1,2,3,0x11}, b{4,5,6,0x80000022}, c{4,5,7,0x80000022};
    history0.upload({a}, {0});history1.upload({b}, {0});
    candidates.upload({a,b,c}, {0,1,2});
    auto const h0=history0.keys(),h1=history1.keys(),v=candidates.keys();
    mgbfs::IndexKeyViews views[2]{};
    for(unsigned word=0;word<4;++word){
      views[0].planes[0][word]=h0.words[word];
      views[1].planes[0][word]=h1.words[word];
      views[0].planes[3][word]=v.words[word];
      views[1].planes[3][word]=v.words[word];
    }
    auto set0=mgbfs::cuco_owner_detail::make_set(2,views[0],resource,stream.view());
    auto set1=mgbfs::cuco_owner_detail::make_set(2,views[1],resource,stream.view());
    mgbfs::cuco_owner_detail::insert_rows<<<1,32,0,stream.value()>>>(
        set0->ref(cuco::insert),0,0,1);
    mgbfs::cuco_owner_detail::insert_rows<<<1,32,0,stream.value()>>>(
        set1->ref(cuco::insert),0,0,1);
    check(cudaGetLastError());
    using Ref=decltype(set0->ref(cuco::contains));
    static_assert(std::is_trivially_copyable_v<Ref>);
    std::array<Ref,2> host_refs{set0->ref(cuco::contains),set1->ref(cuco::contains)};
    rmm::device_buffer refs(sizeof(host_refs),stream.view(),resource);
    rmm::device_buffer hits(3,stream.view(),resource);
    check(cudaMemcpyAsync(refs.data(),host_refs.data(),sizeof(host_refs),
        cudaMemcpyHostToDevice,stream.value()));
    probe_dynamic_refs<<<1,32,0,stream.value()>>>(
        static_cast<Ref const*>(refs.data()),v.words[3],
        static_cast<uint8_t*>(hits.data()),3);
    check(cudaGetLastError());
    std::array<uint8_t,3> actual{};
    check(cudaMemcpyAsync(actual.data(),hits.data(),actual.size(),
        cudaMemcpyDeviceToHost,stream.value()));
    stream.synchronize();
    require(actual==std::array<uint8_t,3>{1,1,0},"DYNAMIC_SHARD_REF_MEMBERSHIP");
  }
  {
    // One GPU transaction spans two persistent shards. The host supplies only
    // fixed capacities and launches; all survivor and reservation counts stay
    // on device until this assertion point.
    constexpr uint32_t cap=8;
    Input previous0(1,stream.view(),resource),previous1(1,stream.view(),resource);
    Input batch(cap,stream.view(),resource);
    Key old0{1,2,3,0x100},x{4,5,6,0x110};
    Key old1{1,2,3,0x40000100},y{4,5,6,0x40000110};
    previous0.upload({old0},{0});previous1.upload({old1},{0});
    batch.upload({old0,x,x,old1,y,y},{0,1,2,3,4,5});
    rmm::device_buffer valid(sizeof(uint32_t),stream.view(),resource);
    uint32_t six=6;
    check(cudaMemcpyAsync(valid.data(),&six,sizeof(six),cudaMemcpyHostToDevice,stream.value()));
    rmm::device_buffer ring(sizeof(MgbfsStateRingControl),stream.view(),resource);
    rmm::device_buffer control(sizeof(MgbfsOwnerControl),stream.view(),resource);
    rmm::device_buffer extent(sizeof(MgbfsStateExtent),stream.view(),resource);
    rmm::device_buffer layer(sizeof(uint32_t),stream.view(),resource);
    rmm::device_buffer source_states(cap*16,stream.view(),resource);
    rmm::device_buffer next_states(16*16,stream.view(),resource);
    check(cudaMemsetAsync(next_states.data(),0,next_states.size(),stream.value()));
    auto upload_states=[&](std::initializer_list<uint8_t> labels){
      std::vector<uint8_t> host(cap*16);
      uint32_t row=0;
      for(uint8_t label:labels){std::fill_n(host.begin()+row*16,16,label);++row;}
      check(cudaMemcpyAsync(source_states.data(),host.data(),host.size(),
          cudaMemcpyHostToDevice,stream.value()));
      stream.synchronize();
    };
    upload_states({11,12,13,14,15,16});
    MgbfsStateRingControl initial_ring{0,0,0,0,16,8,0,0,0};
    uint32_t zero=0;
    check(cudaMemcpyAsync(ring.data(),&initial_ring,sizeof(initial_ring),cudaMemcpyHostToDevice,stream.value()));
    check(cudaMemsetAsync(control.data(),0,sizeof(MgbfsOwnerControl),stream.value()));
    check(cudaMemcpyAsync(layer.data(),&zero,sizeof(zero),cudaMemcpyHostToDevice,stream.value()));
    auto previous=std::array<MgbfsLibraryKeysV1,2>{previous0.keys(),previous1.keys()};
    auto current=std::array<MgbfsLibraryKeysV1,2>{MgbfsLibraryKeysV1{},MgbfsLibraryKeysV1{}};
    auto capacities=std::array<uint32_t,2>{2,2};
    void* rank_owner=nullptr;
    auto prior_resource=rmm::mr::get_current_device_resource_ref();
    rmm::mr::set_current_device_resource_ref(stats);
    require(mgbfs_library_rank_create_cuco_v1(previous.data(),current.data(),
        capacities.data(),2,cap,0,2,stream.value(),&rank_owner)==0&&rank_owner,
        "RANK_ABI_CREATE");
    auto run_batch=[&](uint64_t epoch){
      auto capacity_view=batch.candidates();capacity_view.keys.rows=cap;
      MgbfsLibraryRankDeviceBatchV1 d{};
      require(mgbfs_library_rank_compare_v1(rank_owner,epoch,capacity_view,
          static_cast<uint32_t const*>(valid.data()),
          static_cast<MgbfsOwnerControl*>(control.data()),
          static_cast<MgbfsStateRingControl*>(ring.data()),&d)==0,
          "RANK_ABI_COMPARE");
      require(mgbfs_owner_shard_counts(d.high_words,d.valid_rows,d.selected,d.selected_count,
          cap,0,2,2,d.shard_counts,d.shard_offsets,
          static_cast<MgbfsStateRingControl*>(ring.data()),
          static_cast<MgbfsOwnerControl*>(control.data()),stream.value())==0,
          "RANK_SHARD_COUNTS_ENQUEUE");
      require(mgbfs_state_reserve_rank_batch(
          static_cast<MgbfsStateRingControl*>(ring.data()),
          static_cast<MgbfsOwnerControl*>(control.data()),
          static_cast<MgbfsStateExtent*>(extent.data()),d.shard_counts,
          d.accepted_counts,d.accepted_capacities,2,d.shard_offsets,
          static_cast<uint32_t*>(layer.data()),16,cap,0,stream.value())==0,
          "RANK_RESERVE_ENQUEUE");
      require(mgbfs_library_rank_commit_v1(rank_owner,epoch,
          static_cast<MgbfsOwnerControl*>(control.data()),
          static_cast<MgbfsStateRingControl*>(ring.data()),
          static_cast<MgbfsStateExtent*>(extent.data()))==0,"RANK_ABI_COMMIT");
      require(mgbfs_state_materialize_rank_batch(
          static_cast<uint8_t const*>(source_states.data()),
          static_cast<uint32_t const*>(valid.data()),cap,d.source_indices,
          d.selected_count,cap,16,static_cast<uint8_t*>(next_states.data()),
          static_cast<MgbfsStateRingControl*>(ring.data()),
          static_cast<MgbfsOwnerControl*>(control.data()),
          static_cast<MgbfsStateExtent*>(extent.data()),stream.value())==0,
          "RANK_ABI_MATERIALIZE");
      return d;
    };
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    stream.synchronize(); // fixture uploads, outside the captured owner DAG
    check(cudaStreamBeginCapture(stream.value(),cudaStreamCaptureModeGlobal));
    auto d=run_batch(1);
    check(cudaStreamEndCapture(stream.value(),&graph));
    check(cudaGraphInstantiate(&executable,graph));
    check(cudaGraphLaunch(executable,stream.value()));
    stream.synchronize();
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
    uint32_t got_layer=0;std::array<uint32_t,2> got_counts{};
    MgbfsOwnerControl got_control{};MgbfsStateExtent got_extent{};
    check(cudaMemcpy(&got_layer,layer.data(),sizeof(got_layer),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(got_counts.data(),d.accepted_counts,sizeof(got_counts),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&got_control,control.data(),sizeof(got_control),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&got_extent,extent.data(),sizeof(got_extent),cudaMemcpyDeviceToHost));
    require(!got_control.error&&got_control.stage==2&&got_layer==2&&
        got_extent.count==2&&got_counts==std::array<uint32_t,2>{1,1},
        "RANK_BATCH_DEVICE_COMMIT");
    MgbfsLibraryKeysV1 shard0{},shard1{};
    require(mgbfs_library_rank_export_shard_v1(rank_owner,0,1,&shard0)==0&&
        mgbfs_library_rank_export_shard_v1(rank_owner,1,1,&shard1)==0&&
        read_keys(shard0,stream.view())==std::vector<Key>{x}&&
        read_keys(shard1,stream.view())==std::vector<Key>{y},
        "RANK_BATCH_ACCEPTED_KEYS");
    std::array<uint32_t,2> first_sources{};
    check(cudaMemcpy(first_sources.data(),d.source_indices,sizeof(first_sources),cudaMemcpyDeviceToHost));
    require(first_sources==std::array<uint32_t,2>{1,4},"RANK_BATCH_FIRST_SOURCES");
    std::array<uint8_t,16*16> dense_states{};
    check(cudaMemcpy(dense_states.data(),next_states.data(),dense_states.size(),
        cudaMemcpyDeviceToHost));
    require(dense_states[0]==12&&dense_states[16]==15&&got_extent.ready==1,
        "RANK_ABI_FIRST_DENSE_STATES");
    require(mgbfs_library_rank_seal_v1(rank_owner)!=0,
        "RANK_ABI_SEAL_REJECTS_PENDING_READERS");
    require(mgbfs_library_rank_complete_v1(rank_owner,1)==0,"RANK_ABI_COMPLETE_FIRST");
    Key z{7,8,9,0x120};
    batch.upload({x,z,y},{0,1,2});
    upload_states({21,22,23});
    uint32_t three=3;
    check(cudaMemcpyAsync(valid.data(),&three,sizeof(three),cudaMemcpyHostToDevice,stream.value()));
    auto d2=run_batch(2);
    stream.synchronize();
    check(cudaMemcpy(&got_layer,layer.data(),sizeof(got_layer),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(got_counts.data(),d2.accepted_counts,sizeof(got_counts),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&got_control,control.data(),sizeof(got_control),cudaMemcpyDeviceToHost));
    require(!got_control.error&&got_layer==3&&got_counts==std::array<uint32_t,2>{2,1},
        "RANK_BATCH_STABLE_PERSISTENT_INDICES");
    require(mgbfs_library_rank_export_shard_v1(rank_owner,0,2,&shard0)==0&&
        mgbfs_library_rank_export_shard_v1(rank_owner,1,1,&shard1)==0&&
        read_keys(shard0,stream.view())==std::vector<Key>({x,z})&&
        read_keys(shard1,stream.view())==std::vector<Key>{y},
        "RANK_BATCH_SECOND_KEYS");
    uint32_t second_source=0;
    check(cudaMemcpy(&second_source,d2.source_indices,sizeof(second_source),cudaMemcpyDeviceToHost));
    require(second_source==1,"RANK_BATCH_SECOND_SOURCE");
    check(cudaMemcpy(dense_states.data(),next_states.data(),dense_states.size(),
        cudaMemcpyDeviceToHost));
    require(dense_states[32]==22,"RANK_ABI_SECOND_DENSE_STATE");
    require(mgbfs_library_rank_complete_v1(rank_owner,2)==0,"RANK_ABI_COMPLETE_SECOND");
    Key full{10,11,12,0x130};
    batch.upload({full},{0});
    upload_states({31});
    uint32_t one=1;
    check(cudaMemcpyAsync(valid.data(),&one,sizeof(one),cudaMemcpyHostToDevice,stream.value()));
    auto d3=run_batch(3);
    stream.synchronize();
    MgbfsStateRingControl final_ring{};
    check(cudaMemcpy(&final_ring,ring.data(),sizeof(final_ring),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(got_counts.data(),d3.accepted_counts,sizeof(got_counts),cudaMemcpyDeviceToHost));
    require(final_ring.fatal&&final_ring.tail==3&&
        got_counts==std::array<uint32_t,2>{2,1},"RANK_BATCH_OVERFLOW_ATOMICITY");
    check(cudaMemcpy(dense_states.data(),next_states.data(),dense_states.size(),
        cudaMemcpyDeviceToHost));
    require(dense_states[48]==0,"RANK_ABI_OVERFLOW_NO_DENSE_WRITE");
    require(mgbfs_library_rank_export_shard_v1(rank_owner,0,2,&shard0)==0&&
        read_keys(shard0,stream.view())==std::vector<Key>({x,z}),
        "RANK_BATCH_OVERFLOW_NO_PERSISTENT_WRITE");
    require(mgbfs_library_rank_complete_v1(rank_owner,3)==0,"RANK_ABI_COMPLETE_FATAL");
    require(mgbfs_library_rank_seal_v1(rank_owner)==0,"RANK_ABI_SEAL");
    Key replacement{90,91,92,0x100};
    previous0.upload({replacement},{0});
    require(mgbfs_library_rank_export_shard_v1(rank_owner,0,2,&shard0)==0&&
        read_keys(shard0,stream.view())==std::vector<Key>({x,z}),
        "RANK_ABI_SEALED_KEYS_OWN_STORAGE");
    require(mgbfs_library_rank_destroy_v1(rank_owner)==0,"RANK_ABI_DESTROY");
    rmm::mr::set_current_device_resource_ref(prior_resource);
  }
  {
    constexpr uint32_t incoming = 4096;
    size_t private_bytes;
    {
      mgbfs::CucoOwner a({}, {}, 16, incoming, stream.view(), resource);
      mgbfs::CucoOwner b({}, {}, 16, incoming, stream.view(), resource);
      stream.synchronize();
      private_bytes = stats.get_bytes_counter().value;
    }
    stream.synchronize();
    auto workspace = std::make_shared<mgbfs::CucoWorkspace>(incoming, stream.view(), resource);
    mgbfs::CucoOwner a({}, {}, 16, incoming, stream.view(), resource, workspace);
    stream.synchronize();
    auto const one_shard_bytes = stats.get_bytes_counter().value;
    mgbfs::CucoOwner b({}, {}, 16, incoming, stream.view(), resource, workspace);
    stream.synchronize();
    auto const added_shard_bytes = stats.get_bytes_counter().value - one_shard_bytes;
    std::cout << "SHARED_OWNER_INCREMENT_BYTES " << added_shard_bytes << std::endl;
    // A 16-key persistent shard must not reserve another incoming-sized table.
    // This bound is deliberately loose for cuco extent rounding, but less than
    // even one uint64 slot per incoming candidate (load factor requires >=2).
    require(added_shard_bytes < incoming * sizeof(uint64_t),
            "SHARED_WORKSPACE_MUST_REMOVE_DUPLICATED_TRANSIENT_TABLE");
    require(stats.get_bytes_counter().value + incoming * 16 <= private_bytes,
            "SHARED_WORKSPACE_MUST_REMOVE_DUPLICATED_COLUMNS");
    Input input(incoming, stream.view(), resource);
    Key x{1,2,3,4}, y{1,2,3,5};
    input.upload({x,x}, {70,10});
    auto first = a.compare(1, input.candidates());
    a.commit(1, first.rows);
    // Borrowed result is consumed AFTER commit, before workspace completion.
    require(read(first, stream.view()) == std::vector<uint32_t>{70}, "SHARED_LATE_READER");
    a.complete(1);
    input.upload({x,y,y}, {90,80,20});
    auto second = b.compare(1, input.candidates());
    require(read(second, stream.view()) == std::vector<uint32_t>{90,80}, "SHARED_SHARD_ISOLATION");
    b.commit(1, second.rows);
    stream.synchronize();
    b.complete(1);
    input.upload({x,y}, {4,3});
    auto third = a.compare(2, input.candidates());
    require(read(third, stream.view()) == std::vector<uint32_t>{3}, "SHARED_PERSISTENT_KEYS");
    a.commit(2, third.rows);
    stream.synchronize();
    a.complete(2);
    require(read_keys(a.export_committed(), stream.view()) == std::vector<Key>{x,y}, "SHARED_KEYS_A");
    require(read_keys(b.export_committed(), stream.view()) == std::vector<Key>{x,y}, "SHARED_KEYS_B");
    a.seal();
    b.seal();
  }
  {
    auto workspace = std::make_shared<mgbfs::CucoWorkspace>(16, stream.view(), resource);
    mgbfs::CucoOwner a({}, {}, 8, 16, stream.view(), resource, workspace);
    mgbfs::CucoOwner b({}, {}, 8, 16, stream.view(), resource, workspace);
    Input input(16, stream.view(), resource);
    input.upload({Key{7,8,9,10}}, {42});
    auto result = a.compare(1, input.candidates());
    a.commit(1, result.rows);
    rejects([&] { b.compare(1, input.candidates()); }, "SHARED_COMMIT_IS_NOT_RELEASE");
    require(read(result, stream.view()) == std::vector<uint32_t>{42}, "SHARED_REJECT_KEEPS_READER");
    a.complete(1);
    a.seal();
  }
  {
    Input previous(1, stream.view(), resource), current(1, stream.view(), resource);
    Input input(16, stream.view(), resource);
    Key a{10,20,30,40}, b{11,20,30,40}, c{10,21,30,40};
    Key d{10,20,31,40}, e{10,20,30,41}, f{99,88,77,66};
    previous.upload({a}, {0});
    current.upload({b}, {0});
    {
      mgbfs::CucoOwner owner(previous.keys(), current.keys(), 8, 16, stream.view(), resource);
      stream.synchronize();
      auto const allocated = stats.get_bytes_counter().value;
      auto const peak = stats.get_bytes_counter().peak;
      input.upload({a,c,c,d,b,e}, {500,90,1,80,3,70});
      auto first = owner.compare(1, input.candidates());
      require(read(first, stream.view()) == std::vector<uint32_t>{90,80,70}, "KEEP_FIRST_ROW");
      owner.commit(1, 3);
      stream.synchronize();
      // Recycle the exact same caller slot. A persistent candidate index would
      // now change identity and lose C from membership.
      input.upload({c,f,f}, {4,55,2});
      require(read(owner.compare(2, input.candidates()), stream.view()) ==
              std::vector<uint32_t>{55}, "COMMITTED_KEYS_SURVIVE_SLOT_REUSE");
      owner.commit(2, 1);
      stream.synchronize();
      input.upload({a,b,c,d,e,f}, {0,1,2,3,4,5});
      require(owner.compare(3, input.candidates()).rows == 0, "ALL_HISTORY_WINDOWS");
      owner.commit(3, 0);
      stream.synchronize();
      require(read_keys(owner.export_committed(), stream.view()) ==
              std::vector<Key>{c,d,e,f}, "APPEND_ORDER_KEYS");
      require(stats.get_bytes_counter().value == allocated &&
              stats.get_bytes_counter().peak == peak, "NO_STEADY_STATE_ALLOCATION");
      owner.seal();
      previous.upload({f}, {0});
      current.upload({f}, {0});
      require(read_keys(owner.export_committed(), stream.view()) ==
              std::vector<Key>{c,d,e,f}, "SEALED_EXPORT_OWNS_KEYS");
    }
    {
      mgbfs::CucoOwner owner({}, {}, 1, 16, stream.view(), resource);
      input.upload({a,b}, {1,2});
      rejects([&] { owner.compare(1, input.candidates()); }, "ACCEPTED_CAPACITY");
      rejects([&] { owner.export_committed(); }, "CAPACITY_POISONS");
      stream.synchronize();
    }
    {
      mgbfs::CucoOwner owner({}, {}, 8, 16, stream.view(), resource);
      input.upload({a,b}, {1,2});
      owner.compare(1, input.candidates());
      rejects([&] { owner.commit(1, 1); }, "REAL_COMMIT_CREDITS");
      rejects([&] { owner.compare(2, input.candidates()); }, "CREDIT_POISONS");
      stream.synchronize();
    }
    {
      mgbfs::CucoOwner owner({}, {}, 8, 1, stream.view(), resource);
      input.upload({a,b}, {1,2});
      rejects([&] { owner.compare(1, input.candidates()); }, "INPUT_CAPACITY");
      stream.synchronize();
    }
    {
      mgbfs::CucoOwner owner({}, {}, 8, 16, stream.view(), resource);
      require(owner.compare(9, {}).rows == 0, "EMPTY_INPUT");
      owner.commit(9, 0);
      rejects([&] { owner.compare(9, {}); }, "EPOCH_REPLAY");
      stream.synchronize();
    }
    {
      // Non-tile-aligned, multi-block batches with an independent full-key CPU
      // oracle. This catches arbitrary insert-winner provenance, broken stable
      // compaction, and keys lost when the transient planes are overwritten.
      Input large(4097, stream.view(), resource);
      mgbfs::CucoOwner owner({}, {}, 6000, 4097, stream.view(), resource);
      std::set<Key> seen;
      std::vector<Key> accepted;
      for (uint32_t batch = 0; batch < 4; ++batch) {
        std::vector<Key> keys;
        std::vector<uint32_t> ids, expected;
        for (uint32_t row = 0; row < 4097 - batch * 73; ++row) {
          uint32_t const value = (row * 37 + batch * 613) % 5003;
          // Intentional repeats plus variation in every hash component.
          uint32_t const identity = value / 2;
          Key key{identity % 17, identity / 17, identity ^ 0xaabbccdd,
                  identity * 2654435761u};
          uint32_t const source = 90000 - row * 3;
          keys.push_back(key);
          ids.push_back(source);
          if (seen.insert(key).second) {
            expected.push_back(source);
            accepted.push_back(key);
          }
        }
        large.upload(keys, ids);
        auto result = owner.compare(batch + 1, large.candidates());
        require(read(result, stream.view()) == expected, "MULTIBLOCK_CPU_SURVIVORS");
        owner.commit(batch + 1, result.rows);
        stream.synchronize();
        require(read_keys(owner.export_committed(), stream.view()) == accepted,
                "MULTIBLOCK_CPU_ACCEPTED_KEYS");
      }
      owner.seal();
    }
  }
  stream.synchronize();
  require(stats.get_bytes_counter().value == 0, "OWNER_POOL_LEAK");
  require(pool.pool_size() == pool_bytes, "OWNER_POOL_GROWTH");
  std::cout << "{\"status\":\"PASS\",\"scope\":\"cuco_indexed_owner_contract\"}\n";
}
