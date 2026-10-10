"""Startup-only hardware-derived exact-admission candidate geometry."""
def shard_candidates(inventory,plan):
 if not inventory or any(int(i.get('sm_count',0))<1 for i in inventory):return [1,4,16]
 sm=min(int(i['sm_count']) for i in inventory);l2=max(1,min(int(i.get('l2_bytes',1)) for i in inventory))
 working=int(plan['capacity'])*(int(plan['elements'])*int(plan['state_bytes'])+16)
 target=max(1,min(sm,(working+l2-1)//l2));target=1<<(target-1).bit_length()
 target=min(4096,target);values=[1,max(1,target//2),target,min(4096,target*2)]
 return list(dict.fromkeys(values))

def startup_policy(inventory,plan):
 shards=shard_candidates(inventory,plan)
 sm=min((int(i.get('sm_count',1)) for i in inventory),default=1)
 return {'shards':shards,'size_threshold_per_rank':max(1024,sm*512),'probe_capacity_per_rank':min(int(plan['capacity']),max(32768,sm*16384)),'scope':'real selected-device SM,L2,VRAM and exact admitted state bytes; configuration candidates require GPU measurement'}
