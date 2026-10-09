#include <cassert>
#include "../experiments/nccl_probe_teardown.h"
int main() {
  bool destroyed=false;
  int finalized=0, polled=0, destroys=0;
  auto finalize=[&]{++finalized;return 7;};
  auto progress=[&](int code){++polled;return destroyed?5:(code==7?0:code);};
  auto destroy=[&]{++destroys;destroyed=true;return 0;};
  assert(nccl_probe_teardown(finalize,progress,destroy)==0);
  assert(finalized==1 && polled==1 && destroys==1);
  destroys=0;
  assert(nccl_probe_teardown([]{return 9;},[](int x){return x;},destroy)==9);
  assert(destroys==0);
}
