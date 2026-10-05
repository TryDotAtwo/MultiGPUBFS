"""Independent bounded full-word L/R/X oracle for a complete small tail."""
import hashlib
import json
import math
from pathlib import Path


def verify(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    n, r = manifest['graph']['n'], manifest['graph']['r']
    if math.factorial(n)//math.factorial(r) > 100_000 or manifest['status']!='COMPLETE':
        raise ValueError('bounded complete oracle only')
    frontier = {tuple(manifest['graph']['start'])}
    visited, expected = set(frontier), []
    while frontier:
        expected.append(frontier)
        children = set()
        for state in frontier:
            children.update((state[1:]+state[:1], state[-1:]+state[:-1],
                             (state[1],state[0],*state[2:])))
        frontier = children - visited
        visited.update(frontier)
    actual = [set() for _ in expected]
    width = manifest['packing']['bytes_per_state']
    bits = manifest['packing'].get('bits_per_symbol',4)
    for entry in manifest['files']:
        if not entry['full_layer'] and not entry.get('layer_complete',False):
            raise ValueError('partial layer in complete archive')
        data = (root/entry['path']).read_bytes()
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
            raise ValueError('checksum/size')
        file_rows=0
        for offset in range(0,len(data),width):
            value = int.from_bytes(data[offset:offset+width],'little')
            if value >> (bits*n):
                raise ValueError('nonzero padding')
            state = tuple((value>>(bits*i))&((1<<bits)-1) for i in range(n))
            if state in actual[entry['depth']]:
                raise ValueError('duplicate state')
            actual[entry['depth']].add(state)
            file_rows+=1
        if file_rows!=entry['states']:
            raise ValueError('file row count')
    if actual!=expected or [layer['states'] for layer in manifest['layers']]!=list(map(len,expected)):
        raise ValueError('full layer word sets differ')
    return dict(status='VERIFIED_FULL_STATE_LAYERS',states=len(visited),layers=len(expected))


if __name__=='__main__':
    import sys
    print(json.dumps(verify(sys.argv[1]),indent=2))
