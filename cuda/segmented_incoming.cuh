#pragma once
#include "generic_incoming_action.cuh"

// Iterate only the globally agreed live prefix of each source queue while
// preserving physical q-stride indices in pending references and spill lists.
template<class State,bool Packed>
struct MgbfsBoundedIncomingAction : IncomingAllAction<State,Packed,false> {
 uint32_t visible_per_source;
 __device__ uint32_t child(uint32_t i) const {
  return (i/visible_per_source)*this->stride + i%visible_per_source;
 }
};
