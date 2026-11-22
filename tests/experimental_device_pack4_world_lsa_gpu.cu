// Exact layered permutation BFS protocol experiment for up to eight GPU processes using NCCL LSA.
// Tiny test fixture enumerates its finite domain on CPU for an independent oracle.
// GPU retains only packed frontier rows, not an id-to-state domain table. It uses
// the existing packed action and exact bitmap rank helpers, with per-rank bitmaps.
// This is an independent tiny-graph acceptance test, not the production backend.
// One cooperative GPU block per rank has separate producer, consumer and
// coordinator warps. This bounded research prototype is not the production BFS.
// All graph data, durable visited, frontiers, output and mailbox are preallocated.
// CPU handles admission, launch and final validation only; no layer/portion host calls.
#include "../cuda/experimental_device_ring.cuh"
#include "../cuda/bitmap_pack4_action.cuh"
#include "../cuda/permutation_bitmap.cuh"
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <queue>
#include <map>
#include <string>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace mgbfs_device_ring;
using U64=unsigned long long;
using SysAtomic=cuda::atomic_ref<U64,cuda::thread_scope_system>;
#ifndef MGBFS_EXPERIMENTAL_WORLD
#define MGBFS_EXPERIMENTAL_WORLD 2
#endif
constexpr unsigned WORLD=MGBFS_EXPERIMENTAL_WORLD, PRODUCERS=4, MAX_SLOTS=7, MAX_N=1024, MAX_RECORDS=4096;
static_assert(WORLD>=1 && WORLD<=8,"fixture supports 1 to 8 ranks");
constexpr unsigned ADVANCE=1, COMPLETE=2, INCOMPLETE=3, ERROR=4;
#define CUDA_OK(expr) do { auto e=(expr); if(e!=cudaSuccess) { \
  std::fprintf(stderr,"%s: %s\n",#expr,cudaGetErrorString(e)); return 2; } \
} while(0)
#define NCCL_OK(expr) do { auto e=(expr); if(e!=ncclSuccess) { \
  std::fprintf(stderr,"%s: %s (%d)\n",#expr,ncclGetErrorString(e),int(e)); return 3; } \
} while(0)

struct alignas(64) Mailbox {
  Slot out[PRODUCERS][WORLD][MAX_SLOTS]; // SPSC for every producer/destination
  alignas(8) U64 final_tail[PRODUCERS][WORLD];
  alignas(8) U64 closed_epoch[PRODUCERS];
  alignas(8) U64 done_count,done_epoch,failure,decision_type,decision_epoch;
};
struct Record {unsigned layer;uint64_t word;};
struct PackedWord {
  uint64_t word;MgbfsPack4Dictionary dictionary;
  __device__ int64_t value(uint32_t,uint32_t e) const {
    return dictionary.symbols[(word>>(4*e))&15u];
  }
};
struct Report {
  U64 sent,received,blocked,duplicates,duplicates_full,loops,records;
  unsigned status,depth,rank;
};
__device__ __forceinline__ U64 ld(U64& x) {
  return SysAtomic(x).load(cuda::memory_order_acquire);
}
__device__ __forceinline__ void st(U64& x,U64 v) {
  SysAtomic(x).store(v,cuda::memory_order_release);
}

__global__ void init_box(Mailbox* m,unsigned capacity,U64 job_epoch) {
  if(blockIdx.x||threadIdx.x)return;
  for(unsigned p=0;p<PRODUCERS;p++) {
    for(unsigned d=0;d<WORLD;d++) {
      for(unsigned i=0;i<capacity;i++)init_slot(m->out[p][d][i],job_epoch,i);
      m->final_tail[p][d]=0;
    }
    m->closed_epoch[p]=0;
  }
  m->done_count=m->done_epoch=m->failure=m->decision_type=m->decision_epoch=0;
}
using BlockAtomic=cuda::atomic_ref<U64,cuda::thread_scope_block>;
__device__ __forceinline__ U64 block_load(U64& v) {
  return BlockAtomic(v).load(cuda::memory_order_acquire);
}
__device__ __forceinline__ void block_add(U64& v,U64 n=1) {
  BlockAtomic(v).fetch_add(n,cuda::memory_order_acq_rel);
}
__device__ __forceinline__ U64 block_reserve(U64& v) {
  return BlockAtomic(v).fetch_add(1,cuda::memory_order_acq_rel);
}
__device__ __forceinline__ void fail(Mailbox& m,U64 reason) {
  SysAtomic(m.failure).fetch_or(reason,cuda::memory_order_acq_rel);
}
__global__ void exact_layers(ncclWindow_t win,ncclDevComm dev,unsigned rank,
    unsigned n,const uint32_t* permutations,BitmapDeviceDomain domain,
    MgbfsPack4Dictionary dictionary,unsigned generators,unsigned root,uint64_t root_word,
    unsigned queue_cap,unsigned row_cap,U64 job_epoch,
    unsigned delayed_reads,uint64_t* bank0,uint64_t* bank1,uint64_t* visited,
    Record* records,Report* report) {
  if(blockIdx.x)return;
  __shared__ Mailbox* boxes[WORLD];
  __shared__ U64 current_count,next_count,record_count,producers_done,consumers_done;
  __shared__ U64 total_sent,total_received,total_blocked,total_duplicates,total_duplicates_full,total_steps;
  __shared__ unsigned depth,choice;
  const unsigned tid=threadIdx.x;
  if(tid==64) {
    for(unsigned r=0;r<WORLD;r++)boxes[r]=static_cast<Mailbox*>(ncclGetLsaPointer(win,0,r));
    current_count=next_count=record_count=producers_done=consumers_done=0;
    total_sent=total_received=total_blocked=total_duplicates=total_duplicates_full=total_steps=0;
    depth=choice=0;
    bool missing=false;for(unsigned r=0;r<WORLD;r++)missing|=!boxes[r];
    if(dev.lsaRank!=int(rank)||dev.lsaSize!=int(WORLD)||missing) {
      report->status=ERROR;choice=ERROR;
    } else if(root%WORLD==rank) {
      bank0[0]=root_word;current_count=1;
      uint64_t id=root/WORLD;visited[id>>6]|=uint64_t(1)<<(id&63);
      records[0]={0,root_word};record_count=1;
    }
  }
  __syncthreads();
  // Read only immutable bootstrap data here: choice is subsequently written
  // by the coordinator and can change while another warp is still starting.
  bool missing=false;for(unsigned r=0;r<WORLD;r++)missing|=!boxes[r];
  if(dev.lsaRank!=int(rank)||dev.lsaSize!=int(WORLD)||missing)return;
  Mailbox& local=*boxes[rank];Mailbox& coordinator=*boxes[0];
  U64 send_ticket[WORLD]={};
  U64 receive_ticket=0;
  for(;;) {
    const U64 epoch=U64(depth)+1;
    uint64_t* current=(depth&1)?bank1:bank0;
    uint64_t* next=(depth&1)?bank0:bank1;
    if(tid<PRODUCERS) {
      // Every producer owns distinct SPSC rings and advances a parent/generator
      // cursor only after the target ring has published that candidate.
      U64 loops=0;bool cancelled=false;
      for(U64 parent=tid;parent<current_count&&!cancelled;parent+=PRODUCERS) {
        for(unsigned edge=0;edge<generators&&!cancelled;edge++) {
          uint64_t packed=current[parent],child_id=0;
          BitmapPack4Action action{dictionary.elements,generators,1,&packed,permutations,dictionary};
          uint32_t bad=0;
          if(!action.valid(edge,&bad)||!bitmap_rank(domain,action,edge,&child_id)||child_id>=n) {
            fail(local,2);cancelled=true;break;
          }
          uint64_t child_word=0;
          for(unsigned e=0;e<dictionary.elements;e++) {
            unsigned src=permutations[uint64_t(edge)*dictionary.elements+e];
            child_word|=((packed>>(4*src))&15ull)<<(4*e);
          }
          unsigned target=unsigned(child_id%WORLD);Payload payload{child_word,depth+1,rank};
          for(;;) {
            if(ld(coordinator.decision_epoch)==epoch||++loops>250000000ull) {
              if(loops>250000000ull)fail(local,2);
              cancelled=true;break;
            }
            Result result=try_write(local.out[tid][target],queue_cap,job_epoch,send_ticket[target],payload);
            if(result==Result::ok){send_ticket[target]++;block_add(total_sent);break;}
            if(result!=Result::not_ready){fail(local,2);cancelled=true;break;}
            block_add(total_blocked);__nanosleep(32);
          }
        }
      }
      if(!cancelled) {
        for(unsigned dst=0;dst<WORLD;dst++)local.final_tail[tid][dst]=send_ticket[dst];
        st(local.closed_epoch[tid],epoch);
      }
      block_add(total_steps,loops);block_add(producers_done);
    } else if(tid>=32&&tid<32+WORLD*PRODUCERS) {
      // A separate resident consumer warp owns each incoming SPSC ring.
      const unsigned channel=tid-32,src=channel/PRODUCERS,lane=channel%PRODUCERS;
      U64 loops=0;
      for(;;) {
        if(ld(coordinator.decision_epoch)==epoch||++loops>250000000ull) {
          if(loops>250000000ull)fail(local,2);
          break;
        }
        if(rank==1 && loops<=delayed_reads) {__nanosleep(32);continue;}
        Payload payload{};
        Result result=try_read(boxes[src]->out[lane][rank],queue_cap,job_epoch,receive_ticket,payload);
        if(result==Result::ok) {
          uint64_t id=0;PackedWord candidate{payload.value,dictionary};
          if(payload.kind!=depth+1||payload.source!=src||
             !mgbfs_pack4_valid_word(dictionary,payload.value)||
             !bitmap_rank(domain,candidate,0,&id)||id>=n||id%WORLD!=rank) {
            fail(local,2);break;
          }
          uint64_t local_id=id/WORLD,mask=uint64_t(1)<<(local_id&63);
          auto* word=reinterpret_cast<unsigned long long*>(&visited[local_id>>6]);
          if(atomicOr(word,static_cast<unsigned long long>(mask))&mask) {
            block_add(total_duplicates);
            if(block_load(next_count)>=row_cap)block_add(total_duplicates_full);
          } else {
            U64 row=block_reserve(next_count);
            if(row>=row_cap){fail(local,1);break;}
            U64 record=block_reserve(record_count);
            if(record>=MAX_RECORDS){fail(local,2);break;}
            next[row]=payload.value;records[record]={depth+1,payload.value};
          }
          if(try_release(boxes[src]->out[lane][rank],queue_cap,job_epoch,receive_ticket)!=Result::ok)
            {fail(local,2);break;}
          receive_ticket++;block_add(total_received);
        } else if(result!=Result::not_ready){fail(local,2);break;}
        else if(ld(boxes[src]->closed_epoch[lane])==epoch &&
                receive_ticket==boxes[src]->final_tail[lane][rank])break;
        else __nanosleep(32);
      }
      // Release sequence makes all committed rows visible to the coordinator.
      block_add(total_steps,loops);block_add(consumers_done);
    } else if(tid==64) {
      bool published=false;U64 loops=0;
      for(;;) {
        if(rank==0 && ld(coordinator.decision_epoch)!=epoch) {
          U64 reasons=0;for(unsigned r=0;r<WORLD;r++)reasons|=ld(boxes[r]->failure);
          if(reasons) {
            coordinator.decision_type=(reasons&2u)?ERROR:INCOMPLETE;
            st(coordinator.decision_epoch,epoch);
          }
        }
        if(!published&&ld(local.failure)==0&&block_load(producers_done)==PRODUCERS&&
           block_load(consumers_done)==WORLD*PRODUCERS) {
          local.done_count=block_load(next_count);st(local.done_epoch,epoch);published=true;
        }
        if(rank==0 && ld(coordinator.decision_epoch)!=epoch) {
          bool all=true;U64 sum=0,reasons=0;
          for(unsigned r=0;r<WORLD;r++) {
            reasons|=ld(boxes[r]->failure);
            if(ld(boxes[r]->done_epoch)!=epoch)all=false;
            else sum+=boxes[r]->done_count;
          }
          if(reasons||all) {
            coordinator.decision_type=reasons?((reasons&2u)?ERROR:INCOMPLETE):(sum?ADVANCE:COMPLETE);
            st(coordinator.decision_epoch,epoch);
          }
        }
        if(ld(coordinator.decision_epoch)==epoch) {
          choice=unsigned(coordinator.decision_type);break;
        }
        if(++loops>250000000ull){fail(local,2);choice=ERROR;break;}
        __nanosleep(32);
      }
      block_add(total_steps,loops);
    }
    // On ADVANCE all senders and receivers reached their terminal state.
    // On error they leave without publishing the partial next layer.
    __syncthreads();
    if(choice!=ADVANCE)break;
    if(tid==64) {
      current_count=next_count;next_count=0;producers_done=consumers_done=0;
      depth++;
    }
    __syncthreads();
  }
  if(tid==64) {
    report->sent=total_sent;report->received=total_received;report->blocked=total_blocked;
    report->duplicates=total_duplicates;report->duplicates_full=total_duplicates_full;
    report->loops=total_steps;report->records=record_count;
    report->status=choice;report->depth=depth;report->rank=rank;
  }
}
struct Graph {
  std::string name;unsigned n,root;
  std::vector<std::vector<unsigned>> e;
  std::vector<uint64_t> words;
  std::vector<uint32_t> permutations,counts;
  std::vector<int64_t> symbols;
  MgbfsPack4Dictionary dictionary{};
};
static Graph make_permutation_graph(const char* name,const std::vector<int64_t>& initial,
                                    const std::vector<std::vector<uint32_t>>& generators) {
  Graph g{};g.name=name;
  auto sorted=initial;std::sort(sorted.begin(),sorted.end());
  for(int64_t x:sorted) {
    if(g.symbols.empty()||g.symbols.back()!=x){g.symbols.push_back(x);g.counts.push_back(1);}
    else g.counts.back()++;
  }
  g.dictionary.elements=unsigned(initial.size());g.dictionary.distinct=unsigned(g.symbols.size());
  for(unsigned i=0;i<g.symbols.size();i++)g.dictionary.symbols[i]=g.symbols[i];
  std::map<uint64_t,unsigned> index;
  do {
    uint64_t word=0;if(!mgbfs_pack4_encode(g.dictionary,sorted.data(),&word))std::abort();
    index[word]=unsigned(g.words.size());g.words.push_back(word);
  } while(std::next_permutation(sorted.begin(),sorted.end()));
  g.n=unsigned(g.words.size());
  uint64_t root=0;if(!mgbfs_pack4_encode(g.dictionary,initial.data(),&root))std::abort();
  g.root=index.at(root);g.e.resize(g.n);
  for(auto& gen:generators)for(auto i:gen)g.permutations.push_back(i);
  for(unsigned i=0;i<g.n;i++)for(const auto& gen:generators) {
    uint64_t child=0;
    for(unsigned j=0;j<gen.size();j++)child|=((g.words[i]>>(4*gen[j]))&15ull)<<(4*j);
    g.e[i].push_back(index.at(child));
  }
  return g;
}
static std::vector<std::vector<uint32_t>> adjacent(unsigned n) {
  std::vector<std::vector<uint32_t>> gens;
  for(unsigned i=0;i+1<n;i++) {
    std::vector<uint32_t> p(n);for(unsigned j=0;j<n;j++)p[j]=j;
    std::swap(p[i],p[i+1]);gens.push_back(p);
  }
  return gens;
}
static Graph graph(unsigned which) {
  if(which==0)return make_permutation_graph("s5_arbitrary_root",{4,0,3,1,2},adjacent(5));
  if(which==1)return make_permutation_graph("repeated_001122",{1,0,2,0,1,2},adjacent(6));
  if(which==2)return make_permutation_graph("signed_directed_rotation",{-7,25,0,13},{{1,2,3,0}});
  if(which==3)return make_permutation_graph("s6_adjacent",{0,1,2,3,4,5},adjacent(6));
  return make_permutation_graph("duplicate_when_full",{0,1,2},{{1,0,2},{1,0,2},{0,1,2}});
}
// Independent reference uses a standard FIFO BFS and does not use the GPU protocol.
static std::vector<int> oracle(const Graph& g) {
  std::vector<int> distance(g.n,-1);std::queue<unsigned> q;
  distance[g.root]=0;q.push(g.root);
  while(!q.empty()) {
    unsigned v=q.front();q.pop();
    for(unsigned w:g.e[v])if(distance[w]<0){distance[w]=distance[v]+1;q.push(w);}
  }
  return distance;
}
static std::string encode(const ncclUniqueId& id) {
  const char* h="0123456789abcdef";std::string s;
  for(unsigned char c:id.internal){s.push_back(h[c>>4]);s.push_back(h[c&15]);}return s;
}
static bool decode(const char* s,ncclUniqueId& id) {
  if(std::strlen(s)!=2*sizeof(id.internal))return false;
  auto d=[](char c)->int {if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;return -1;};
  for(size_t i=0;i<sizeof(id.internal);i++){int a=d(s[2*i]),b=d(s[2*i+1]);if(a<0||b<0)return false;id.internal[i]=char(a*16+b);}
  return true;
}
static int rank_main(int rank,const ncclUniqueId& id) {
  int n=0,ver=0,coop=0;
  CUDA_OK(cudaGetDeviceCount(&n));NCCL_OK(ncclGetVersion(&ver));if(n!=int(WORLD)||ver<22900)return 77;
  const bool swapped=std::getenv("MGBFS_DEVICE_TEST_SWAP")!=nullptr;
  const int device=swapped?int(WORLD)-1-rank:rank;
  CUDA_OK(cudaSetDevice(device));
  CUDA_OK(cudaDeviceGetAttribute(&coop,cudaDevAttrCooperativeLaunch,device));
  if(!coop)return 77;
  for(unsigned peer_rank=0;peer_rank<WORLD;peer_rank++) {
    if(peer_rank==unsigned(rank))continue;
    int peer_device=swapped?int(WORLD)-1-int(peer_rank):int(peer_rank),peer=0,atom=0;
    CUDA_OK(cudaDeviceCanAccessPeer(&peer,device,peer_device));
    CUDA_OK(cudaDeviceGetP2PAttribute(&atom,cudaDevP2PAttrNativeAtomicSupported,device,peer_device));
    std::printf("META rank=%d device=%d peer=%d swapped=%d p2p=%d atomics=%d coop=%d\n",rank,device,peer_device,int(swapped),peer,atom,coop);
    if(!peer||!atom)return 77;
    auto enabled=cudaDeviceEnablePeerAccess(peer_device,0);
    if(enabled!=cudaSuccess&&enabled!=cudaErrorPeerAccessAlreadyEnabled)return 2;
    if(enabled==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();
  }
  ncclComm_t comm{};ncclWindow_t win{};ncclDevComm dev{};
  Mailbox* box=nullptr;cudaStream_t stream{};int* barrier=nullptr;
  uint64_t *bank0=nullptr,*bank1=nullptr,*visited=nullptr;
  uint32_t *permutations=nullptr,*counts=nullptr;int64_t* symbols=nullptr;
  Record* records=nullptr;Report* report=nullptr;
  NCCL_OK(ncclCommInitRank(&comm,WORLD,id,rank));
  ncclCommProperties_t props=NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_OK(ncclCommQueryProperties(comm,&props));
  if(!props.deviceApiSupport||props.nLsaTeams!=1)return 77;
  NCCL_OK(ncclMemAlloc(reinterpret_cast<void**>(&box),sizeof(Mailbox)));
  CUDA_OK(cudaMalloc(&barrier,sizeof(int)));
  CUDA_OK(cudaMalloc(&bank0,MAX_N*sizeof(uint64_t)));CUDA_OK(cudaMalloc(&bank1,MAX_N*sizeof(uint64_t)));
  CUDA_OK(cudaMalloc(&permutations,256*sizeof(uint32_t)));
  CUDA_OK(cudaMalloc(&counts,16*sizeof(uint32_t)));CUDA_OK(cudaMalloc(&symbols,16*sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&visited,((MAX_N+63)/64)*sizeof(uint64_t)));
  CUDA_OK(cudaMalloc(&records,MAX_RECORDS*sizeof(Record)));
  CUDA_OK(cudaMalloc(&report,sizeof(Report)));CUDA_OK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  NCCL_OK(ncclCommWindowRegister(comm,box,sizeof(Mailbox),&win,NCCL_WIN_COLL_SYMMETRIC));
  ncclDevCommRequirements req=NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;req.lsaBarrierCount=1;
  NCCL_OK(ncclDevCommCreate(comm,&req,&dev));if(dev.lsaRank!=rank||dev.lsaSize!=int(WORLD))return 77;
  unsigned case_index=0;bool any_failed=false;
  for(unsigned graph_index=0;graph_index<5;graph_index++) {
    Graph g=graph(graph_index);auto distance=oracle(g);
    unsigned max_depth=0;for(int x:distance)if(x>=0)max_depth=std::max(max_depth,unsigned(x));
    unsigned needed=1;
    for(unsigned d=0;d<=max_depth;d++)for(unsigned r=0;r<WORLD;r++) {
      unsigned count=0;for(unsigned v=0;v<g.n;v++)if(distance[v]==int(d)&&v%WORLD==r)count++;
      needed=std::max(needed,count);
    }
    for(unsigned qcap: {1u,7u})for(unsigned variant=0;variant<3;variant++) {
      if(variant==2&&needed==1)continue;
      unsigned cap=variant==0?MAX_N:variant==1?needed:needed-1;
      int first_bad=-1;
      for(unsigned d=0;d<=max_depth;d++)for(unsigned r=0;r<WORLD;r++) {
        unsigned count=0;for(unsigned v=0;v<g.n;v++)if(distance[v]==int(d)&&v%WORLD==r)count++;
        if(count>cap&&(first_bad<0||int(d)<first_bad))first_bad=int(d);
      }
      unsigned expected_status=first_bad<0?COMPLETE:INCOMPLETE;
      unsigned expected_depth=first_bad<0?max_depth:unsigned(first_bad-1);
      U64 job_epoch=501+(case_index++);
      CUDA_OK(cudaMemcpyAsync(permutations,g.permutations.data(),g.permutations.size()*sizeof(uint32_t),cudaMemcpyHostToDevice,stream));
      CUDA_OK(cudaMemcpyAsync(counts,g.counts.data(),g.counts.size()*sizeof(uint32_t),cudaMemcpyHostToDevice,stream));
      CUDA_OK(cudaMemcpyAsync(symbols,g.symbols.data(),g.symbols.size()*sizeof(int64_t),cudaMemcpyHostToDevice,stream));
      CUDA_OK(cudaMemsetAsync(visited,0,((MAX_N+63)/64)*sizeof(uint64_t),stream));CUDA_OK(cudaMemsetAsync(report,0,sizeof(Report),stream));
      init_box<<<1,1,0,stream>>>(box,qcap,job_epoch);CUDA_OK(cudaGetLastError());
      CUDA_OK(cudaMemsetAsync(barrier,0,sizeof(int),stream));
      NCCL_OK(ncclAllReduce(barrier,barrier,1,ncclInt,ncclSum,comm,stream));
      CUDA_OK(cudaStreamSynchronize(stream));
      unsigned urank=unsigned(rank),un=g.n,root=g.root,delay=(graph_index==1&&rank==1)?10000:0;
      BitmapDeviceDomain domain{symbols,counts,g.dictionary.distinct,g.dictionary.elements,g.n};
      MgbfsPack4Dictionary dictionary=g.dictionary;
      unsigned generators=unsigned(g.permutations.size()/dictionary.elements);
      uint64_t root_word=g.words[root];
      void* args[]={&win,&dev,&urank,&un,&permutations,&domain,&dictionary,&generators,
                    &root,&root_word,&qcap,&cap,&job_epoch,&delay,&bank0,&bank1,&visited,&records,&report};
      CUDA_OK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(exact_layers),dim3(1),dim3(96),args,0,stream));
      CUDA_OK(cudaStreamSynchronize(stream));
      Report out{};CUDA_OK(cudaMemcpy(&out,report,sizeof(out),cudaMemcpyDeviceToHost));
      std::vector<Record> actual(out.records<=MAX_RECORDS?out.records:0);
      if(!actual.empty())CUDA_OK(cudaMemcpy(actual.data(),records,actual.size()*sizeof(Record),cudaMemcpyDeviceToHost));
      std::vector<std::pair<unsigned,uint64_t>> got,want;
      for(auto r:actual)if(r.layer<=out.depth)got.emplace_back(r.layer,r.word);
      for(unsigned v=0;v<g.n;v++)if(v%WORLD==unsigned(rank)&&distance[v]>=0&&unsigned(distance[v])<=expected_depth)
        want.emplace_back(unsigned(distance[v]),g.words[v]);
      std::sort(got.begin(),got.end());std::sort(want.begin(),want.end());
      // This special duplicate-at-full assertion targets two-rank ownership only.
      bool exercised_full_duplicate=WORLD!=2||graph_index!=4||rank!=0||cap!=1||out.duplicates_full>0;
      bool ok=out.status==expected_status&&out.depth==expected_depth&&got==want&&out.records<=MAX_RECORDS&&exercised_full_duplicate;
      std::printf("%s graph=%s rank=%d q=%u cap=%u status=%u depth=%u complete_records=%zu partial_records=%llu sent=%llu recv=%llu blocked=%llu duplicates=%llu duplicates_full=%llu loops=%llu\n",
        ok?"PASS":"FAIL",g.name.c_str(),rank,qcap,cap,out.status,out.depth,got.size(),
        out.records-got.size(),out.sent,out.received,out.blocked,out.duplicates,out.duplicates_full,out.loops);
      if(!ok)std::fprintf(stderr,"EXPECTED graph=%s rank=%d status=%u depth=%u count=%zu got=%zu\n",
        g.name.c_str(),rank,expected_status,expected_depth,want.size(),got.size());
      int local_ok=ok?0:1;CUDA_OK(cudaMemcpyAsync(barrier,&local_ok,sizeof(int),cudaMemcpyHostToDevice,stream));
      NCCL_OK(ncclAllReduce(barrier,barrier,1,ncclInt,ncclSum,comm,stream));
      CUDA_OK(cudaMemcpyAsync(&local_ok,barrier,sizeof(int),cudaMemcpyDeviceToHost,stream));
      CUDA_OK(cudaStreamSynchronize(stream));
      if(local_ok){any_failed=true;break;}
    }
    if(any_failed)break;
  }
  NCCL_OK(ncclDevCommDestroy(comm,&dev));NCCL_OK(ncclCommWindowDeregister(comm,win));
  CUDA_OK(cudaStreamDestroy(stream));
  CUDA_OK(cudaFree(bank0));CUDA_OK(cudaFree(bank1));CUDA_OK(cudaFree(visited));
  CUDA_OK(cudaFree(permutations));CUDA_OK(cudaFree(counts));CUDA_OK(cudaFree(symbols));
  CUDA_OK(cudaFree(records));CUDA_OK(cudaFree(report));CUDA_OK(cudaFree(barrier));
  NCCL_OK(ncclMemFree(box));NCCL_OK(ncclCommDestroy(comm));
  return any_failed?4:0;
}
int main(int argc,char** argv) {
  setvbuf(stdout,nullptr,_IONBF,0);
  if(argc==4&&!std::strcmp(argv[1],"--rank")) {
    int rank=std::atoi(argv[2]);ncclUniqueId id{};
    if(rank<0||rank>=int(WORLD)||!decode(argv[3],id))return 6;
    return rank_main(rank,id);
  }
  if(argc!=1)return 6;
  ncclUniqueId id{};NCCL_OK(ncclGetUniqueId(&id));
  std::string value=encode(id);std::vector<pid_t> pids(WORLD);
  for(unsigned i=0;i<WORLD;i++) {
    pids[i]=fork();
    if(!pids[i]){auto index=std::to_string(i);execl(argv[0],argv[0],"--rank",index.c_str(),value.c_str(),nullptr);_exit(127);}
    if(pids[i]<0)return 7;
  }
  bool ok=true;
  for(unsigned i=0;i<WORLD;i++) {
    int s=0;if(waitpid(pids[i],&s,0)<0)ok=false;
    std::printf("CHILD rank=%d exit=%d\n",i,WIFEXITED(s)?WEXITSTATUS(s):-1);
    if(!WIFEXITED(s)||WEXITSTATUS(s))ok=false;
  }
  if(!ok)return 8;
  std::puts("PASS experimental multi-warp packed bitmap NCCL LSA BFS vs CPU layer multisets; not production");
  return 0;
}
