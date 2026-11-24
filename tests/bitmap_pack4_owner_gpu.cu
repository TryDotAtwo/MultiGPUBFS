#include "../cuda/bitmap_pack4_owner.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CK(x) do { cudaError_t ec=(x); if(ec!=cudaSuccess) {std::cerr<<"CUDA "<<__LINE__<<" "<<cudaGetErrorString(ec)<<"\n";std::exit(1);}}while(0)
#define REQUIRE(x) do { if(!(x)){std::cerr<<"FAIL "<<__LINE__<<" "#x"\n";std::exit(1);} }while(0)

static uint64_t mix(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
template<class T> T* alloc(size_t n){T* p=nullptr;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T> void upload(T* p,const std::vector<T>& x){CK(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));}

static void run(int gpu,int wire,bool repeated) {
 const bool byte_payload=wire!=0, packed=wire==1;
 CK(cudaSetDevice(gpu));
 constexpr uint32_t world=2,shards=2,q=16,n=4;
 std::vector<int64_t> root=byte_payload?
  (repeated?std::vector<int64_t>{2,2,7,25}:std::vector<int64_t>{2,7,11,25}):
  (repeated?std::vector<int64_t>{-50,-50,12,900}:std::vector<int64_t>{-50,-1,12,900});
 std::vector<std::vector<int64_t>> states;auto cursor=root;
 do{states.push_back(cursor);}while(std::next_permutation(cursor.begin(),cursor.end()));
 MgbfsPack4Dictionary dict{};dict.elements=n;
 for(auto value:root)if(!dict.distinct||dict.symbols[dict.distinct-1]!=value)dict.symbols[dict.distinct++]=value;
 std::array<std::vector<GenericRouteRecord>,2> meta={std::vector<GenericRouteRecord>(world*shards*q),std::vector<GenericRouteRecord>(world*shards*q)};
 std::array<std::vector<int64_t>,2> payload={std::vector<int64_t>(world*shards*n*q),std::vector<int64_t>(world*shards*n*q)};
 std::array<std::vector<uint32_t>,2> counts={std::vector<uint32_t>(world*shards),std::vector<uint32_t>(world*shards)};
 for(uint64_t id=0;id<states.size();id+=2){
  const uint32_t sh=(uint64_t(uint32_t(mix(id)))*shards)>>32;
  for(uint32_t src=0;src<world;src++){
   uint32_t queue=src*shards+sh,row=counts[src][queue]++;
   REQUIRE(row<q);
   uint64_t lo=0;for(uint32_t e=0;e<n;e++)if(byte_payload)lo|=uint64_t(states[id][e])<<(8*e);
   meta[src][queue*q+row]=GenericRouteRecord{id,lo,0,0,0,0};
   for(uint32_t e=0;e<n;e++)payload[src][queue*n*q+e*q+row]=states[id][e];
  }
 }
 for(uint32_t sh=0;sh<shards;sh++)REQUIRE(counts[0][sh]>0 && counts[1][shards+sh]>0);
 GenericRouteRecord *lm=alloc<GenericRouteRecord>(meta[0].size()),*rm=alloc<GenericRouteRecord>(meta[1].size());
 void *lp=nullptr,*rp=nullptr;
 if(byte_payload&&!packed){lp=alloc<uint8_t>(payload[0].size());rp=alloc<uint8_t>(payload[1].size());}
 else {lp=alloc<int64_t>(payload[0].size());rp=alloc<int64_t>(payload[1].size());}
 uint32_t *lc=alloc<uint32_t>(counts[0].size()),*rc=alloc<uint32_t>(counts[1].size());
 uint64_t* visited=alloc<uint64_t>(1);CK(cudaMemset(visited,0,8));
 upload(lm,meta[0]);upload(rm,meta[1]);
 if(byte_payload&&!packed){
  for(uint32_t src=0;src<2;src++){
   std::vector<uint8_t> bytes(payload[src].begin(),payload[src].end());
   upload(static_cast<uint8_t*>(src?rp:lp),bytes);
  }
 }else {upload(static_cast<int64_t*>(lp),payload[0]);upload(static_cast<int64_t*>(rp),payload[1]);}
 upload(lc,counts[0]);upload(rc,counts[1]);

 std::vector<std::vector<int64_t>> results;
 uint32_t total_spills=0;
 for(uint32_t sh=0;sh<shards;sh++){
  const uint32_t visible=std::max(counts[0][sh],counts[1][shards+sh]);
  uint32_t selected=world*visible;
  uint32_t *selection=alloc<uint32_t>(world*q),*spill=alloc<uint32_t>(world*q),*spill_count=alloc<uint32_t>(1),*error=alloc<uint32_t>(1);
  uint64_t *accepted=alloc<uint64_t>(1),*rows=alloc<uint64_t>(2);
  bool replay=false;
  for(uint32_t pass=0;selected;pass++){
   REQUIRE(pass<40);CK(cudaMemset(spill_count,0,4));CK(cudaMemset(error,0,4));CK(cudaMemset(accepted,0,8));CK(cudaMemset(rows,0xff,16));
   MgbfsPack4OwnerArgs a{};
   a.dictionary=dict;a.elements=n;a.state_bytes=byte_payload?1:8;a.packed=packed;
   a.local=lp;a.remote=rp;a.local_meta=lm;a.remote_meta=rm;a.local_counts=lc;a.remote_counts=rc;
   a.rank=0;a.world=world;a.owner=0;a.total=states.size();a.owner_len=states.size()/2;a.bits=visited;
   a.shard=sh;a.shards=shards;a.q=q;a.visible=visible;a.selection=replay?selection:nullptr;a.selected=selected;
   a.capacity=1;a.banks=2;a.bank=0;a.rows=rows;a.accepted=accepted;a.spill=spill;a.spill_count=spill_count;a.error=error;
   CK((cudaError_t)mgbfs_pack4_owner_accept(a,nullptr));CK(cudaDeviceSynchronize());
   uint32_t err=0,num_spill=0;uint64_t num=0,word=0;
   CK(cudaMemcpy(&err,error,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&num_spill,spill_count,4,cudaMemcpyDeviceToHost));
   CK(cudaMemcpy(&num,accepted,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&word,rows,8,cudaMemcpyDeviceToHost));
   REQUIRE(err==0&&num<=1&&num_spill<=selected);total_spills+=num_spill;
   if(num){
    REQUIRE(mgbfs_pack4_valid_word(dict,word));
    std::vector<int64_t> out(n);REQUIRE(mgbfs_pack4_decode(dict,word,out.data()));results.push_back(out);
   }
   selected=num_spill;replay=true;
   if(selected)CK(cudaMemcpy(selection,spill,selected*4,cudaMemcpyDeviceToDevice));
  }
  // A full segment accepting only already-visited states must not spill.
  CK(cudaMemset(spill_count,0,4));CK(cudaMemset(error,0,4));uint64_t one=1;CK(cudaMemcpy(accepted,&one,8,cudaMemcpyHostToDevice));
  MgbfsPack4OwnerArgs a{};a.dictionary=dict;a.elements=n;a.state_bytes=byte_payload?1:8;a.packed=packed;
  a.local=lp;a.remote=rp;a.local_meta=lm;a.remote_meta=rm;a.local_counts=lc;a.remote_counts=rc;
  a.rank=0;a.world=world;a.owner=0;a.total=states.size();a.owner_len=states.size()/2;a.bits=visited;
  a.shard=sh;a.shards=shards;a.q=q;a.visible=visible;a.selected=world*visible;
  a.capacity=1;a.banks=2;a.rows=rows;a.accepted=accepted;a.spill=spill;a.spill_count=spill_count;a.error=error;
  CK((cudaError_t)mgbfs_pack4_owner_accept(a,nullptr));CK(cudaDeviceSynchronize());
  uint32_t err=0,num_spill=0;CK(cudaMemcpy(&err,error,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&num_spill,spill_count,4,cudaMemcpyDeviceToHost));
  REQUIRE(err==0 && num_spill==0);
  CK(cudaFree(selection));CK(cudaFree(spill));CK(cudaFree(spill_count));CK(cudaFree(error));CK(cudaFree(accepted));CK(cudaFree(rows));
 }
 std::vector<std::vector<int64_t>> expected;for(uint32_t i=0;i<states.size();i+=2)expected.push_back(states[i]);
 std::sort(results.begin(),results.end());std::sort(expected.begin(),expected.end());REQUIRE(results==expected);
 uint64_t mask=0;CK(cudaMemcpy(&mask,visited,8,cudaMemcpyDeviceToHost));REQUIRE(mask==((1ull<<(states.size()/2))-1));REQUIRE(total_spills>0);
 CK(cudaFree(lm));CK(cudaFree(rm));CK(cudaFree(lp));CK(cudaFree(rp));CK(cudaFree(lc));CK(cudaFree(rc));CK(cudaFree(visited));
 std::cout<<"OWNER_PASS gpu="<<gpu<<" wire="<<(packed?"packed-u8":(byte_payload?"soa-u8":"wide-i64"))<<" repeated="<<repeated<<" unique="<<results.size()<<" shards=2 spills="<<total_spills<<"\n";
}

static void run_invalid(int gpu) {
 CK(cudaSetDevice(gpu));MgbfsPack4Dictionary d{};d.elements=4;d.distinct=4;
 d.symbols[0]=-50;d.symbols[1]=-1;d.symbols[2]=12;d.symbols[3]=900;
 std::vector<GenericRouteRecord> m(2);m[0].hash=0;
 std::vector<int64_t> payload={-50,-1,12,901,0,0,0,0};
 // SoA for queue 0: one element per row, then empty queue 1.
 std::vector<uint32_t> c={1,0},empty={0,0};
 auto *dm=alloc<GenericRouteRecord>(2);auto *dp=alloc<int64_t>(8);
 auto *dc=alloc<uint32_t>(2),*de=alloc<uint32_t>(2),*spill=alloc<uint32_t>(2),*spills=alloc<uint32_t>(1),*err=alloc<uint32_t>(1),*selection=alloc<uint32_t>(1);
 auto *bits=alloc<uint64_t>(1),*rows=alloc<uint64_t>(2),*accepted=alloc<uint64_t>(1);
 upload(dm,m);upload(dp,payload);upload(dc,c);upload(de,empty);
 MgbfsPack4OwnerArgs a{};a.dictionary=d;a.elements=4;a.state_bytes=8;
 a.local=dp;a.remote=dp;a.local_meta=dm;a.remote_meta=dm;a.local_counts=dc;a.remote_counts=de;
 a.rank=0;a.world=2;a.owner=0;a.total=24;a.owner_len=12;a.bits=bits;a.shards=1;a.q=1;a.visible=1;a.selected=2;
 a.capacity=1;a.banks=2;a.rows=rows;a.accepted=accepted;a.spill=spill;a.spill_count=spills;a.error=err;
 auto reset=[&](){CK(cudaMemset(bits,0,8));CK(cudaMemset(rows,0xff,16));CK(cudaMemset(accepted,0,8));CK(cudaMemset(spills,0,4));CK(cudaMemset(err,0,4));};
 auto check=[&](){
  uint32_t e=0,sc=0;uint64_t b=0,cnt=0,row=0;
  CK(cudaMemcpy(&e,err,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&sc,spills,4,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&b,bits,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&cnt,accepted,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&row,rows,8,cudaMemcpyDeviceToHost));
  REQUIRE(e!=0&&sc==0&&b==0&&cnt==0&&row==UINT64_MAX);
 };
 reset();CK((cudaError_t)mgbfs_pack4_owner_accept(a,nullptr));CK(cudaDeviceSynchronize());check(); // unknown symbol
 payload[3]=900;upload(dp,payload);m[0].hash=1;upload(dm,m);
 reset();CK((cudaError_t)mgbfs_pack4_owner_accept(a,nullptr));CK(cudaDeviceSynchronize());check(); // wrong owner
 m[0].hash=0;upload(dm,m);uint32_t outside=2;CK(cudaMemcpy(selection,&outside,4,cudaMemcpyHostToDevice));
 a.selection=selection;a.selected=1;reset();CK((cudaError_t)mgbfs_pack4_owner_accept(a,nullptr));CK(cudaDeviceSynchronize());check(); // invalid origin
 a.selection=nullptr;a.selected=2;a.dictionary.symbols[1]=a.dictionary.symbols[0];
 reset();REQUIRE(mgbfs_pack4_owner_accept(a,nullptr)==int(cudaErrorInvalidValue));CK(cudaDeviceSynchronize());
 uint64_t b=1;CK(cudaMemcpy(&b,bits,8,cudaMemcpyDeviceToHost));REQUIRE(b==0);
 CK(cudaFree(dm));CK(cudaFree(dp));CK(cudaFree(dc));CK(cudaFree(de));CK(cudaFree(spill));CK(cudaFree(spills));CK(cudaFree(err));CK(cudaFree(selection));CK(cudaFree(bits));CK(cudaFree(rows));CK(cudaFree(accepted));
 std::cout<<"OWNER_INVALID_PASS gpu="<<gpu<<" cases=4\n";
}



static void run_cap(int gpu){
 CK(cudaSetDevice(gpu));constexpr uint32_t n=4,q=2;
 MgbfsPack4Dictionary d{};d.elements=n;d.distinct=n;d.symbols[0]=2;d.symbols[1]=7;d.symbols[2]=11;d.symbols[3]=25;
 uint64_t packed=uint64_t(2)|(uint64_t(7)<<8)|(uint64_t(11)<<16)|(uint64_t(25)<<24);
 std::vector<GenericRouteRecord> host_meta={GenericRouteRecord{0,packed,0,0,0,0},GenericRouteRecord{0,packed,0,0,0,0}};
 std::vector<uint32_t> host_counts={2};auto *meta=alloc<GenericRouteRecord>(2);auto *counts=alloc<uint32_t>(1);
 upload(meta,host_meta);upload(counts,host_counts);
 auto *bits=alloc<uint64_t>(1),*rows=alloc<uint64_t>(2),*accepted=alloc<uint64_t>(1),*global=alloc<uint64_t>(1);
 auto *spill=alloc<uint32_t>(2),*selected_in=alloc<uint32_t>(2),*spill_count=alloc<uint32_t>(1),*error=alloc<uint32_t>(1);
 CK(cudaMemset(bits,0,8));CK(cudaMemset(rows,0xff,16));CK(cudaMemset(accepted,0,8));uint64_t occupied=1;CK(cudaMemcpy(global,&occupied,8,cudaMemcpyHostToDevice));
 MgbfsPack4OwnerArgs a{};a.dictionary=d;a.elements=n;a.state_bytes=1;a.packed=1;
 a.local_meta=meta;a.remote_meta=meta;a.local_counts=counts;a.remote_counts=counts;a.world=1;a.owner=0;a.total=24;a.owner_len=24;a.bits=bits;
 a.shards=1;a.q=q;a.visible=2;a.selected=2;a.capacity=1;a.banks=2;a.rows=rows;a.accepted=accepted;
 a.spill=spill;a.spill_count=spill_count;a.error=error;a.global_count=global;a.global_limit=1;
 CK(cudaMemset(spill_count,0,4));CK(cudaMemset(error,0,4));CK((cudaError_t)mgbfs_pack4_owner_accept(a,nullptr));CK(cudaDeviceSynchronize());
 uint32_t err=0,spilled=0;uint64_t mask=0,num=0,g=0;
 CK(cudaMemcpy(&err,error,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&spilled,spill_count,4,cudaMemcpyDeviceToHost));
 CK(cudaMemcpy(&mask,bits,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&num,accepted,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&g,global,8,cudaMemcpyDeviceToHost));
 REQUIRE((err&16u)&&spilled>0&&!mask&&!num&&g==1);
 CK(cudaMemcpy(selected_in,spill,spilled*4,cudaMemcpyDeviceToDevice));CK(cudaMemset(global,0,8));CK(cudaMemset(spill_count,0,4));CK(cudaMemset(error,0,4));
 a.selection=selected_in;a.selected=spilled;CK((cudaError_t)mgbfs_pack4_owner_accept(a,nullptr));CK(cudaDeviceSynchronize());
 CK(cudaMemcpy(&err,error,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&spilled,spill_count,4,cudaMemcpyDeviceToHost));
 CK(cudaMemcpy(&mask,bits,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&num,accepted,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&g,global,8,cudaMemcpyDeviceToHost));
 REQUIRE(!err&&!spilled&&mask==1&&num==1&&g==1);
 // An already visited duplicate at a full rank cap consumes no quota or spill.
 a.selection=nullptr;a.selected=2;CK(cudaMemset(error,0,4));CK(cudaMemset(spill_count,0,4));
 CK((cudaError_t)mgbfs_pack4_owner_accept(a,nullptr));CK(cudaDeviceSynchronize());
 CK(cudaMemcpy(&err,error,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&spilled,spill_count,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&g,global,8,cudaMemcpyDeviceToHost));
 REQUIRE(!err&&!spilled&&g==1);
 CK(cudaFree(meta));CK(cudaFree(counts));CK(cudaFree(bits));CK(cudaFree(rows));CK(cudaFree(accepted));CK(cudaFree(global));CK(cudaFree(spill));CK(cudaFree(selected_in));CK(cudaFree(spill_count));CK(cudaFree(error));
 std::cout<<"OWNER_CAP_REPLAY_PASS gpu="<<gpu<<"\n";
}

static void run_packet(int gpu,bool inline_states,bool direct=false) {
 CK(cudaSetDevice(gpu));
 constexpr uint32_t n=4,world=3,owner=1,shards=2,shard=1,q=5,visible=2;
 std::vector<int64_t> root=inline_states?std::vector<int64_t>{2,7,11,15}:std::vector<int64_t>{-50,-1,12,900};
 std::vector<std::vector<int64_t>> states;
 auto perm=root;
 do {states.push_back(perm);} while(std::next_permutation(perm.begin(),perm.end()));
 MgbfsPack4Dictionary dictionary{};
 dictionary.elements=n;dictionary.distinct=n;
 std::copy(root.begin(),root.end(),dictionary.symbols);
 std::vector<std::vector<int64_t>> expected;
 uint64_t expected_bits=0;
 std::vector<GenericRouteRecord> input(q);
 std::vector<uint64_t> wire(2*q);
 std::vector<int64_t> payload(n*q);
 for(uint64_t id=owner;id<states.size()&&expected.size()<visible;id+=world) {
  if((uint64_t(uint32_t(mix(id)))*shards)>>32!=shard)continue;
  const size_t row=expected.size();
  uint64_t bytes=0;
  for(uint32_t e=0;e<n;e++) {
   if(inline_states)bytes|=uint64_t(states[id][e])<<(8*e);
   else payload[e*q+row]=states[id][e];
  }
  input[row]=GenericRouteRecord{id,bytes,0,0,0,0};
  if(direct) {
   wire[2*row]=id;
   for(uint32_t e=0;e<n;e++)wire[2*row+1]|=uint64_t(states[id][e])<<(4*e);
  }
  expected.push_back(states[id]);
  expected_bits|=1ull<<(id/world);
 }
 REQUIRE(expected.size()==visible);
 auto *meta=alloc<GenericRouteRecord>(q);
 auto *data=alloc<int64_t>(n*q);
 auto *counts=alloc<uint32_t>(1),*spill=alloc<uint32_t>(q),*count=alloc<uint32_t>(1),*error=alloc<uint32_t>(1);
 auto *bits=alloc<uint64_t>(1),*rows=alloc<uint64_t>(2),*accepted=alloc<uint64_t>(1);
 if(direct)upload(reinterpret_cast<uint64_t*>(meta),wire);else upload(meta,input);
 upload(data,payload);upload(counts,std::vector<uint32_t>{visible});
 CK(cudaMemset(bits,0,8));
 MgbfsPack4OwnerArgs a{};
 a.dictionary=dictionary;a.elements=n;a.state_bytes=inline_states?1:8;a.packed=inline_states;
 a.local=inline_states?nullptr:data;a.local_meta=meta;a.local_counts=counts;
 a.rank=owner;a.world=world;a.owner=owner;a.total=states.size();a.owner_len=states.size()/world;a.bits=bits;
 a.shard=shard;a.shards=shards;a.q=q;a.visible=visible;a.selected=visible;
 a.capacity=1;a.banks=2;a.bank=1;a.rows=rows;a.accepted=accepted;a.spill=spill;a.spill_count=count;a.error=error;
 auto *selection=alloc<uint32_t>(q);
 std::vector<std::vector<int64_t>> accepted_states;
 auto accept=direct?mgbfs_pack4_owner_accept_wire_v1:mgbfs_pack4_owner_accept_packet_v1;
 for(size_t pass=0;pass<visible;pass++) {
  CK(cudaMemset(accepted,0,8));CK(cudaMemset(count,0,4));CK(cudaMemset(error,0,4));
  CK((cudaError_t)accept(a,nullptr));CK(cudaDeviceSynchronize());
  uint32_t err=0,spill_count=0;
  uint64_t accepted_count=0,word=0;
  CK(cudaMemcpy(&err,error,4,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&spill_count,count,4,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&accepted_count,accepted,8,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&word,rows+1,8,cudaMemcpyDeviceToHost));
  REQUIRE(err==0&&accepted_count==1&&spill_count==visible-pass-1);
  std::vector<int64_t> state(n);
  REQUIRE(mgbfs_pack4_decode(dictionary,word,state.data()));
  accepted_states.push_back(state);
  if(spill_count) {
   uint32_t origin=q;
   CK(cudaMemcpy(&origin,spill,4,cudaMemcpyDeviceToHost));REQUIRE(origin<visible);
   CK(cudaMemcpy(selection,spill,4,cudaMemcpyDeviceToDevice));
   a.selection=selection;a.selected=spill_count;
  }
 }
 uint64_t mask=0;
 CK(cudaMemcpy(&mask,bits,8,cudaMemcpyDeviceToHost));
 std::sort(expected.begin(),expected.end());std::sort(accepted_states.begin(),accepted_states.end());
 REQUIRE(mask==expected_bits&&accepted_states==expected);
 a.selection=nullptr;a.selected=world*visible;
 REQUIRE(accept(a,nullptr)==int(cudaErrorInvalidValue));
 CK(cudaFree(meta));CK(cudaFree(data));CK(cudaFree(counts));CK(cudaFree(spill));CK(cudaFree(count));CK(cudaFree(error));
 CK(cudaFree(bits));CK(cudaFree(rows));CK(cudaFree(accepted));CK(cudaFree(selection));
}


#ifdef MGBFS_PACK4_OWNER_TEST_BARRIER
static void run_concurrent(int gpu) {
 CK(cudaSetDevice(gpu));
 constexpr uint32_t n=4,shards=2,q=1;
 std::vector<int64_t> root={2,7,11,25};std::vector<std::vector<int64_t>> states;auto cur=root;
 do{states.push_back(cur);}while(std::next_permutation(cur.begin(),cur.end()));
 MgbfsPack4Dictionary d{};d.elements=n;d.distinct=n;for(uint32_t i=0;i<n;i++)d.symbols[i]=root[i];
 std::array<uint64_t,2> ids={UINT64_MAX,UINT64_MAX};
 for(uint64_t id=0;id<states.size();id++){uint32_t sh=(uint64_t(uint32_t(mix(id)))*shards)>>32;if(ids[sh]==UINT64_MAX)ids[sh]=id;}
 REQUIRE(ids[0]!=UINT64_MAX&&ids[1]!=UINT64_MAX);
 std::vector<GenericRouteRecord> meta(2);std::vector<uint32_t> counts={1,1};
 for(uint32_t sh=0;sh<2;sh++){
  uint64_t bytes=0;for(uint32_t e=0;e<n;e++)bytes|=uint64_t(states[ids[sh]][e])<<(8*e);
  meta[sh]=GenericRouteRecord{ids[sh],bytes,0,0,0,0};
 }
 auto *dm=alloc<GenericRouteRecord>(2);auto *dc=alloc<uint32_t>(2),*gate=alloc<uint32_t>(1),*selection=alloc<uint32_t>(1);
 auto *bits=alloc<uint64_t>(1),*global=alloc<uint64_t>(1);upload(dm,meta);upload(dc,counts);
 std::array<cudaStream_t,2> streams{};std::array<uint64_t*,2> rows{},accepted{};
 std::array<uint32_t*,2> spill{},spills{},err{};std::array<MgbfsPack4OwnerArgs,2> args{};
 for(uint32_t sh=0;sh<2;sh++){
  CK(cudaStreamCreateWithFlags(&streams[sh],cudaStreamNonBlocking));
  rows[sh]=alloc<uint64_t>(2);accepted[sh]=alloc<uint64_t>(1);spill[sh]=alloc<uint32_t>(1);
  spills[sh]=alloc<uint32_t>(1);err[sh]=alloc<uint32_t>(1);
  auto& a=args[sh];a.dictionary=d;a.elements=n;a.state_bytes=1;a.packed=1;
  a.local_meta=dm;a.remote_meta=dm;a.local_counts=dc;a.remote_counts=dc;
  a.world=1;a.owner=0;a.total=states.size();a.owner_len=states.size();a.bits=bits;
  a.shard=sh;a.shards=shards;a.q=q;a.visible=1;a.selected=1;
  a.capacity=1;a.banks=2;a.rows=rows[sh];a.accepted=accepted[sh];a.spill=spill[sh];
  a.spill_count=spills[sh];a.error=err[sh];a.test_gate=gate;a.global_count=global;
 }
 for(uint64_t limit:std::array<uint64_t,2>{2,1}){
  CK(cudaMemset(bits,0,8));CK(cudaMemset(gate,0,4));CK(cudaMemset(global,0,8));
  for(uint32_t sh=0;sh<2;sh++){
   CK(cudaMemset(rows[sh],0xff,16));CK(cudaMemset(accepted[sh],0,8));CK(cudaMemset(spills[sh],0,4));CK(cudaMemset(err[sh],0,4));
   args[sh].global_limit=limit;args[sh].selection=nullptr;args[sh].selected=1;args[sh].test_gate=gate;
  }
  CK(cudaDeviceSynchronize());
  for(uint32_t sh=0;sh<2;sh++)CK((cudaError_t)mgbfs_pack4_owner_accept(args[sh],streams[sh]));
  CK(cudaDeviceSynchronize());uint32_t arrived=0;uint64_t mask=0,g=0;
  CK(cudaMemcpy(&arrived,gate,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&mask,bits,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&g,global,8,cudaMemcpyDeviceToHost));
  REQUIRE(arrived==2&&g==limit);
  int deferred=-1;
  for(uint32_t sh=0;sh<2;sh++){
   uint32_t error=0,num_spill=0;uint64_t count=0;
   CK(cudaMemcpy(&error,err[sh],4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&num_spill,spills[sh],4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&count,accepted[sh],8,cudaMemcpyDeviceToHost));
   if(count==0){REQUIRE(limit==1&&error==16&&num_spill==1&&deferred<0);deferred=sh;REQUIRE(!(mask&(1ull<<ids[sh])));}
   else {REQUIRE(count==1&&!error&&!num_spill&&(mask&(1ull<<ids[sh])));}
  }
  if(limit==1){
   REQUIRE(deferred>=0);uint32_t sh=uint32_t(deferred);
   CK(cudaMemcpy(selection,spill[sh],4,cudaMemcpyDeviceToDevice));CK(cudaMemset(err[sh],0,4));CK(cudaMemset(spills[sh],0,4));CK(cudaDeviceSynchronize());
   args[sh].selection=selection;args[sh].global_limit=2;args[sh].test_gate=nullptr;
   CK((cudaError_t)mgbfs_pack4_owner_accept(args[sh],streams[sh]));CK(cudaStreamSynchronize(streams[sh]));
   uint32_t error=0,num_spill=0;CK(cudaMemcpy(&error,err[sh],4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&num_spill,spills[sh],4,cudaMemcpyDeviceToHost));
   CK(cudaMemcpy(&mask,bits,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&g,global,8,cudaMemcpyDeviceToHost));
   REQUIRE(!error&&!num_spill&&g==2);
  }
  REQUIRE(mask==((1ull<<ids[0])|(1ull<<ids[1])));
  for(uint32_t sh=0;sh<2;sh++){
   uint64_t count=0,word=0;std::vector<int64_t> decoded(n);
   CK(cudaMemcpy(&count,accepted[sh],8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&word,rows[sh],8,cudaMemcpyDeviceToHost));
   REQUIRE(count==1&&mgbfs_pack4_decode(d,word,decoded.data())&&decoded==states[ids[sh]]);
  }
 }
 for(uint32_t sh=0;sh<2;sh++){CK(cudaStreamDestroy(streams[sh]));CK(cudaFree(rows[sh]));CK(cudaFree(accepted[sh]));CK(cudaFree(spill[sh]));CK(cudaFree(spills[sh]));CK(cudaFree(err[sh]));}
 CK(cudaFree(dm));CK(cudaFree(dc));CK(cudaFree(bits));CK(cudaFree(gate));CK(cudaFree(global));CK(cudaFree(selection));
 std::cout<<"OWNER_CONCURRENT_CAP_BARRIER_PASS gpu="<<gpu<<" shards=2 bitmap_words=1 cap_retry_replayed=1\n";
}

#endif

int main(){int devices=0;CK(cudaGetDeviceCount(&devices));REQUIRE(devices>=2);for(int d=0;d<2;d++){for(int wire=0;wire<3;wire++)for(bool repeated:{false,true})run(d,wire,repeated);run_invalid(d);run_cap(d);run_packet(d,true);run_packet(d,false);run_packet(d,true,true);
#ifdef MGBFS_PACK4_OWNER_TEST_BARRIER
 run_concurrent(d);
#endif
 }
 std::cout<<"PACK4_OWNER_PASS gpu=2 exact_multisets=12\n";}
