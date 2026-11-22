// Exact layered permutation BFS protocol experiment for two GPU processes using NCCL LSA.
// Tiny test fixture enumerates its finite domain on CPU and transfers id->packed-word
// as a lookup; GPU child rank uses the actual packed action and bitmap rank helpers.
// This test-only lookup is not an implementation proposal for production-sized graphs.
// This is an independent tiny-graph acceptance test, not the production backend.
// One cooperative GPU thread per rank alternates progress of send/receive.
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
constexpr unsigned WORLD=2, MAX_SLOTS=7, MAX_N=1024, MAX_RECORDS=4096;
constexpr unsigned ADVANCE=1, COMPLETE=2, INCOMPLETE=3, ERROR=4;
#define CUDA_OK(expr) do { auto e=(expr); if(e!=cudaSuccess) { \
  std::fprintf(stderr,"%s: %s\n",#expr,cudaGetErrorString(e)); return 2; } \
} while(0)
#define NCCL_OK(expr) do { auto e=(expr); if(e!=ncclSuccess) { \
  std::fprintf(stderr,"%s: %s (%d)\n",#expr,ncclGetErrorString(e),int(e)); return 3; } \
} while(0)

struct alignas(64) Mailbox {
  Slot out[WORLD][MAX_SLOTS]; // one producer and consumer per (source,destination)
  alignas(8) U64 final_tail[WORLD];
  alignas(8) U64 closed_epoch;
  alignas(8) U64 done_count;
  alignas(8) U64 done_epoch;
  alignas(8) U64 failure; // 1 capacity, 2 protocol
  alignas(8) U64 decision_type; // rank zero only
  alignas(8) U64 decision_epoch;
};
struct Record {unsigned layer,vertex;};
struct Report {
  U64 sent,received,blocked,duplicates,loops,records;
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
  for(unsigned d=0;d<WORLD;d++) {
    for(unsigned i=0;i<capacity;i++)init_slot(m->out[d][i],job_epoch,i);
    m->final_tail[d]=0;
  }
  m->closed_epoch=m->done_count=m->done_epoch=m->failure=0;
  m->decision_type=m->decision_epoch=0;
}
__global__ void exact_layers(ncclWindow_t win,ncclDevComm dev,unsigned rank,
    unsigned n,const unsigned* offsets,const unsigned* edges,
    const uint64_t* words,const uint32_t* permutations,BitmapDeviceDomain domain,
    MgbfsPack4Dictionary dictionary,unsigned generators,unsigned root,unsigned queue_cap,unsigned row_cap,U64 job_epoch,
    unsigned delayed_reads,unsigned* bank0,unsigned* bank1,unsigned char* visited,
    Record* records,Report* report) {
  if(blockIdx.x||threadIdx.x)return;
  Mailbox* boxes[WORLD];
  for(unsigned i=0;i<WORLD;i++)boxes[i]=static_cast<Mailbox*>(ncclGetLsaPointer(win,0,i));
  if(dev.lsaRank!=int(rank)||dev.lsaSize!=int(WORLD)||!boxes[0]||!boxes[1]) {
    report->status=ERROR;return;
  }
  Mailbox& local=*boxes[rank]; Mailbox& coordinator=*boxes[0];
  unsigned* current=bank0;unsigned* next=bank1;
  unsigned current_count=0,next_count=0,depth=0,parent=0,edge=0;
  U64 sent[WORLD]={0,0},received[WORLD]={0,0};
  U64 total_sent=0,total_received=0,blocked=0,duplicates=0,steps=0,nrecords=0;
  bool closed=false,done=false;
  unsigned status=ERROR;
  if(root%WORLD==rank) {
    current[current_count++]=root;visited[root]=1;records[nrecords++]={0,root};
  }
  // Iteration cap is only a fault detector for this bounded experimental test.
  while(steps++<250000000ull) {
    const U64 epoch=U64(depth)+1;
    if(rank==0 && ld(coordinator.decision_epoch)!=epoch) {
      U64 failure=0;
      for(unsigned r=0;r<WORLD;r++)failure|=ld(boxes[r]->failure);
      if(failure) {
        coordinator.decision_type=(failure&2u)?ERROR:INCOMPLETE;
        st(coordinator.decision_epoch,epoch);
      }
    }
    if(ld(coordinator.decision_epoch)==epoch) {
      unsigned choice=unsigned(coordinator.decision_type);
      if(choice==ADVANCE) {
        unsigned* temp=current;current=next;next=temp;
        current_count=next_count;next_count=0;depth++;parent=0;edge=0;
        closed=false;done=false;continue;
      }
      status=choice;break;
    }
    if(!done && ld(local.failure)==0) {
      // Drain first: no blocked producer can starve its consumer.
      if(steps>delayed_reads)for(unsigned src=0;src<WORLD;src++) {
        Payload payload{};
        Result result=try_read(boxes[src]->out[rank],queue_cap,job_epoch,received[src],payload);
        if(result==Result::ok) {
          const unsigned v=unsigned(payload.value);
          if(payload.value>=n||payload.kind!=depth+1||payload.source!=src||v%WORLD!=rank) {
            st(local.failure,2);break;
          }
          if(visited[v]) duplicates++;
          else {
            if(next_count>=row_cap) {st(local.failure,1);break;}
            if(nrecords>=MAX_RECORDS) {st(local.failure,2);break;}
            // Materialize a durable frontier row and output before publishing visited.
            next[next_count++]=v;
            records[nrecords++]={depth+1,v};
            visited[v]=1;
          }
          if(try_release(boxes[src]->out[rank],queue_cap,job_epoch,received[src])!=Result::ok)
            {st(local.failure,2);break;}
          received[src]++;total_received++;
        } else if(result!=Result::not_ready) {st(local.failure,2);break;}
      }
      if(ld(local.failure)==0&&!closed) {
        // A child cursor is advanced only when a slot has actually been published.
        while(parent<current_count && edge>=generators){parent++;edge=0;}
        if(parent<current_count) {
          uint64_t packed=words[current[parent]], child_id=0;
          BitmapPack4Action action{dictionary.elements,generators,1,&packed,permutations,dictionary};
          uint32_t bad=0;
          if(!action.valid(edge,&bad)||!bitmap_rank(domain,action,edge,&child_id)||child_id>=n) {
            st(local.failure,2);continue;
          }
          unsigned v=unsigned(child_id);
          if(v>=n)st(local.failure,2);
          else {
            unsigned target=v%WORLD;Payload p{v,depth+1,rank};
            Result result=try_write(local.out[target],queue_cap,job_epoch,sent[target],p);
            if(result==Result::ok){sent[target]++;total_sent++;edge++;}
            else if(result==Result::not_ready)blocked++;
            else st(local.failure,2);
          }
        } else {
          for(unsigned dst=0;dst<WORLD;dst++)local.final_tail[dst]=sent[dst];
          st(local.closed_epoch,epoch);closed=true;
        }
      }
      if(closed && ld(local.failure)==0) {
        bool drained=true;
        for(unsigned src=0;src<WORLD;src++) {
          if(ld(boxes[src]->closed_epoch)!=epoch ||
             received[src]!=boxes[src]->final_tail[rank])drained=false;
        }
        if(drained) {
          local.done_count=next_count;
          st(local.done_epoch,epoch);done=true;
        }
      }
    }
    if(rank==0 && ld(coordinator.decision_epoch)!=epoch) {
      U64 failure=0;
      for(unsigned r=0;r<WORLD;r++)failure|=ld(boxes[r]->failure);
      if(failure) {
        coordinator.decision_type=(failure&2u)?ERROR:INCOMPLETE;
        st(coordinator.decision_epoch,epoch);
      } else {
        bool all=true;U64 sum=0;
        for(unsigned r=0;r<WORLD;r++) {
          if(ld(boxes[r]->done_epoch)!=epoch)all=false;
          else sum+=boxes[r]->done_count;
        }
        if(all) {
          coordinator.decision_type=sum?ADVANCE:COMPLETE;
          st(coordinator.decision_epoch,epoch);
        }
      }
    }
    __nanosleep(32);
  }
  report->sent=total_sent;report->received=total_received;report->blocked=blocked;
  report->duplicates=duplicates;report->loops=steps;report->records=nrecords;
  report->status=status;report->depth=depth;report->rank=rank;
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
  return make_permutation_graph("s6_adjacent",{0,1,2,3,4,5},adjacent(6));
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
  int n=0,ver=0,peer=0,atom=0,coop=0;
  CUDA_OK(cudaGetDeviceCount(&n));NCCL_OK(ncclGetVersion(&ver));if(n!=2||ver<22900)return 77;
  CUDA_OK(cudaSetDevice(rank));
  CUDA_OK(cudaDeviceCanAccessPeer(&peer,rank,1-rank));
  CUDA_OK(cudaDeviceGetP2PAttribute(&atom,cudaDevP2PAttrNativeAtomicSupported,rank,1-rank));
  CUDA_OK(cudaDeviceGetAttribute(&coop,cudaDevAttrCooperativeLaunch,rank));
  if(!peer||!atom||!coop)return 77;
  auto enabled=cudaDeviceEnablePeerAccess(1-rank,0);
  if(enabled!=cudaSuccess&&enabled!=cudaErrorPeerAccessAlreadyEnabled)return 2;
  if(enabled==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();
  ncclComm_t comm{};ncclWindow_t win{};ncclDevComm dev{};
  Mailbox* box=nullptr;cudaStream_t stream{};int* barrier=nullptr;
  unsigned *offsets=nullptr,*edges=nullptr,*bank0=nullptr,*bank1=nullptr;
  uint64_t* words=nullptr;uint32_t *permutations=nullptr,*counts=nullptr;int64_t* symbols=nullptr;
  unsigned char* visited=nullptr;Record* records=nullptr;Report* report=nullptr;
  NCCL_OK(ncclCommInitRank(&comm,2,id,rank));
  ncclCommProperties_t props=NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_OK(ncclCommQueryProperties(comm,&props));
  if(!props.deviceApiSupport||props.nLsaTeams!=1)return 77;
  NCCL_OK(ncclMemAlloc(reinterpret_cast<void**>(&box),sizeof(Mailbox)));
  CUDA_OK(cudaMalloc(&barrier,sizeof(int)));CUDA_OK(cudaMalloc(&offsets,(MAX_N+1)*sizeof(unsigned)));
  CUDA_OK(cudaMalloc(&edges,MAX_N*MAX_N*sizeof(unsigned)));
  CUDA_OK(cudaMalloc(&bank0,MAX_N*sizeof(unsigned)));CUDA_OK(cudaMalloc(&bank1,MAX_N*sizeof(unsigned)));
  CUDA_OK(cudaMalloc(&words,MAX_N*sizeof(uint64_t)));CUDA_OK(cudaMalloc(&permutations,256*sizeof(uint32_t)));
  CUDA_OK(cudaMalloc(&counts,16*sizeof(uint32_t)));CUDA_OK(cudaMalloc(&symbols,16*sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&visited,MAX_N));CUDA_OK(cudaMalloc(&records,MAX_RECORDS*sizeof(Record)));
  CUDA_OK(cudaMalloc(&report,sizeof(Report)));CUDA_OK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  NCCL_OK(ncclCommWindowRegister(comm,box,sizeof(Mailbox),&win,NCCL_WIN_COLL_SYMMETRIC));
  ncclDevCommRequirements req=NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;req.lsaBarrierCount=1;
  NCCL_OK(ncclDevCommCreate(comm,&req,&dev));if(dev.lsaRank!=rank||dev.lsaSize!=2)return 77;
  unsigned case_index=0;bool any_failed=false;
  for(unsigned graph_index=0;graph_index<4;graph_index++) {
    Graph g=graph(graph_index);auto distance=oracle(g);
    std::vector<unsigned> off(g.n+1),ee;
    for(unsigned i=0;i<g.n;i++){off[i]=unsigned(ee.size());ee.insert(ee.end(),g.e[i].begin(),g.e[i].end());}
    off[g.n]=unsigned(ee.size());
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
      CUDA_OK(cudaMemcpyAsync(words,g.words.data(),g.words.size()*sizeof(uint64_t),cudaMemcpyHostToDevice,stream));
      CUDA_OK(cudaMemcpyAsync(permutations,g.permutations.data(),g.permutations.size()*sizeof(uint32_t),cudaMemcpyHostToDevice,stream));
      CUDA_OK(cudaMemcpyAsync(counts,g.counts.data(),g.counts.size()*sizeof(uint32_t),cudaMemcpyHostToDevice,stream));
      CUDA_OK(cudaMemcpyAsync(symbols,g.symbols.data(),g.symbols.size()*sizeof(int64_t),cudaMemcpyHostToDevice,stream));
      CUDA_OK(cudaMemcpyAsync(offsets,off.data(),off.size()*sizeof(unsigned),cudaMemcpyHostToDevice,stream));
      CUDA_OK(cudaMemcpyAsync(edges,ee.data(),ee.size()*sizeof(unsigned),cudaMemcpyHostToDevice,stream));
      CUDA_OK(cudaMemsetAsync(visited,0,MAX_N,stream));CUDA_OK(cudaMemsetAsync(report,0,sizeof(Report),stream));
      init_box<<<1,1,0,stream>>>(box,qcap,job_epoch);CUDA_OK(cudaGetLastError());
      CUDA_OK(cudaMemsetAsync(barrier,0,sizeof(int),stream));
      NCCL_OK(ncclAllReduce(barrier,barrier,1,ncclInt,ncclSum,comm,stream));
      CUDA_OK(cudaStreamSynchronize(stream));
      unsigned urank=unsigned(rank),un=g.n,root=g.root,delay=(graph_index==1&&rank==1)?10000:0;
      BitmapDeviceDomain domain{symbols,counts,g.dictionary.distinct,g.dictionary.elements,g.n};
      MgbfsPack4Dictionary dictionary=g.dictionary;
      unsigned generators=unsigned(g.permutations.size()/dictionary.elements);
      void* args[]={&win,&dev,&urank,&un,&offsets,&edges,&words,&permutations,&domain,&dictionary,&generators,
                    &root,&qcap,&cap,&job_epoch,&delay,&bank0,&bank1,&visited,&records,&report};
      CUDA_OK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(exact_layers),dim3(1),dim3(1),args,0,stream));
      CUDA_OK(cudaStreamSynchronize(stream));
      Report out{};CUDA_OK(cudaMemcpy(&out,report,sizeof(out),cudaMemcpyDeviceToHost));
      std::vector<Record> actual(out.records<=MAX_RECORDS?out.records:0);
      if(!actual.empty())CUDA_OK(cudaMemcpy(actual.data(),records,actual.size()*sizeof(Record),cudaMemcpyDeviceToHost));
      std::vector<std::pair<unsigned,unsigned>> got,want;
      for(auto r:actual)if(r.layer<=out.depth)got.emplace_back(r.layer,r.vertex);
      for(unsigned v=0;v<g.n;v++)if(v%WORLD==unsigned(rank)&&distance[v]>=0&&unsigned(distance[v])<=expected_depth)
        want.emplace_back(unsigned(distance[v]),v);
      std::sort(got.begin(),got.end());std::sort(want.begin(),want.end());
      bool ok=out.status==expected_status&&out.depth==expected_depth&&got==want&&out.records<=MAX_RECORDS;
      std::printf("%s graph=%s rank=%d q=%u cap=%u status=%u depth=%u complete_records=%zu partial_records=%llu sent=%llu recv=%llu blocked=%llu duplicates=%llu loops=%llu\n",
        ok?"PASS":"FAIL",g.name.c_str(),rank,qcap,cap,out.status,out.depth,got.size(),
        out.records-got.size(),out.sent,out.received,out.blocked,out.duplicates,out.loops);
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
  CUDA_OK(cudaStreamDestroy(stream));CUDA_OK(cudaFree(offsets));CUDA_OK(cudaFree(edges));
  CUDA_OK(cudaFree(bank0));CUDA_OK(cudaFree(bank1));CUDA_OK(cudaFree(visited));
  CUDA_OK(cudaFree(words));CUDA_OK(cudaFree(permutations));CUDA_OK(cudaFree(counts));CUDA_OK(cudaFree(symbols));
  CUDA_OK(cudaFree(records));CUDA_OK(cudaFree(report));CUDA_OK(cudaFree(barrier));
  NCCL_OK(ncclMemFree(box));NCCL_OK(ncclCommDestroy(comm));
  return any_failed?4:0;
}
int main(int argc,char** argv) {
  setvbuf(stdout,nullptr,_IONBF,0);
  if(argc==4&&!std::strcmp(argv[1],"--rank")) {
    int rank=std::atoi(argv[2]);ncclUniqueId id{};
    if(rank<0||rank>1||!decode(argv[3],id))return 6;
    return rank_main(rank,id);
  }
  if(argc!=1)return 6;
  ncclUniqueId id{};NCCL_OK(ncclGetUniqueId(&id));
  std::string value=encode(id);pid_t pids[2]{};
  for(int i=0;i<2;i++) {
    pids[i]=fork();
    if(!pids[i]){execl(argv[0],argv[0],"--rank",i?"1":"0",value.c_str(),nullptr);_exit(127);}
    if(pids[i]<0)return 7;
  }
  bool ok=true;
  for(int i=0;i<2;i++) {
    int s=0;if(waitpid(pids[i],&s,0)<0)ok=false;
    std::printf("CHILD rank=%d exit=%d\n",i,WIFEXITED(s)?WEXITSTATUS(s):-1);
    if(!WIFEXITED(s)||WEXITSTATUS(s))ok=false;
  }
  if(!ok)return 8;
  std::puts("PASS experimental NCCL LSA permutation BFS production rank/action vs independent CPU multiset oracle; not production");
  return 0;
}
