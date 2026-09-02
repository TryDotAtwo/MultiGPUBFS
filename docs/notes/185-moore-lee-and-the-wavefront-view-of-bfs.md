# Moore, Lee, and the wavefront view of BFS

## Question

Did early work present BFS primarily as a queue discipline, or as a spreading
shortest-path wave?

## What the primary Lee paper establishes

C. Y. Lee's 1961 paper starts from path-connection problems in logical
drawings, wiring diagrams, and route finding. It models a space as cells with
finite neighborhoods and separates the computation into:

1. a **search** that assigns an optimal "cell mass" and a chain coordinate;
2. a **trace** that follows those coordinates backward from target to source.

For the minimal-distance case, Lee explicitly says that the illustrations use
one of Moore's algorithms and that this is a specialization of Lee's more
general Algorithm A. The general algorithm orders lexicographic vectors of
monotone path properties, so it should not be collapsed into ordinary
unit-edge FIFO BFS. Lee also notes that when a path property grows one unit at
a time, the general machinery can be simplified.

The paper's own physical analogy is a wave expanding from a source. On a square
cell grid with obstacles, the first wave arrival assigns the minimum-distance
mass; the later trace selects one path back through decreasing masses. This is
the same conceptual separation used in modern BFS:

```text
wave/search     -> distance labels and reached layers
backtrace       -> one chosen shortest witness
```

The wave may contain many cells and many possible predecessors while the trace
retains only one route. Thus shortest-distance computation and path selection
were already visibly different output stages.

## Smallest modern translation

On a four-neighbor unit grid, start cell `s` receives label zero. Every complete
wave round labels previously unlabeled neighbors with the next integer:

```text
label 0 cells = F_0
label 1 cells = F_1
label 2 cells = F_2
...
```

An obstacle removes a cell or transition from the declared graph. When target
`t` first receives label `d`, the expanding wave has proved a length-`d` route
and has closed every smaller label. A trace may then repeatedly choose a
neighbor labeled `d-1,d-2,...,0`. The queue is one mechanism for scheduling the
wave, not the historical idea that makes the proof intuitive.

## Historical boundary

E. F. Moore's *The Shortest Path Through a Maze* is bibliographically attested
as pages 285--292 of the 1959 Harvard proceedings, and Lee cites it directly.
The full Moore text was not obtained in this pass, so this note does **not**
attribute detailed pseudocode or terminology to Moore from secondary
retellings. What is directly supported here is:

- Moore published the named shortest-maze work in 1959;
- Lee treated a Moore minimal-distance algorithm as prior definitive work;
- Lee's own 1961 search/trace formulation and wave analogy are visible in the
  primary paper.

## New intuition

BFS is easier to recognize historically and mathematically as **labeling by
first wave arrival, followed optionally by a trace**, rather than as "the
algorithm that owns a FIFO queue." This also explains why distance layers are
canonical while a returned parent path depends on a selection rule.

## Follow-up source search

A second targeted search found no inspectable full scan of Moore's eight-page
text. Google Books exposes a catalog record for a 1959 Bell Telephone System
monograph, while the searchable proceedings records expose metadata only.

Lawler's 1976 textbook gives a useful secondary description: it calls Moore's
procedure essentially Dijkstra-like for unit edge costs and says Moore avoided
storing numeric distance values by retaining two bits per grid node for the
shortest-path tree. This is a lead for checking the original pages, **not**
primary evidence used elsewhere in this study. Until the Moore text is
inspected, the exact marking scheme, storage claim, and pseudocode remain
secondary-source reports.

## Third source search: a stronger reconstruction, still not the paper

A third targeted search again failed to produce an inspectable primary scan.
Google Books identifies an eight-page Bell Telephone System monograph, WorldCat
identifies the printed 1959 item, and a SciSpace result labels the paper open
access, but its page returned an access error during inspection. Search-result
metadata is not a substitute for reading the eight pages.

Alexander Schrijver's historical chapter supplies a more detailed secondary
reconstruction. It reports that Moore presented four procedures, A--D. In its
account, Algorithm A starts by labeling the source `0` and, for
`k=0,1,...`, labels every still-unlabeled neighbor of a vertex labeled `k` with
`k+1`, stopping when the target is labeled. This is metric-layer propagation in
almost the exact set language of modern BFS. Schrijver also separates Moore's
later Algorithm D as a shortest-route method for general edge lengths.

This narrows two historical claims without closing the primary-source gap:

1. the association of Moore's Algorithm A with the modern unit-edge BFS wave is
   supported by a specialist historical reconstruction, not merely by generic
   textbook attribution;
2. “Moore's algorithm” is ambiguous: the paper reportedly contains several
   algorithms, including a weighted one, so the whole paper must not be
   collapsed into FIFO BFS.

The evidence hierarchy is now:

```text
Lee 1961 primary text: directly inspected wave/search/trace formulation
Moore 1959 bibliographic identity: confirmed
Moore A--D details: convergent secondary reports
Moore 1959 exact wording, figures, and full procedures: still unverified.
```

Obtaining the original through a library or lawful scan remains the only clean
way to promote the final line. No executable experiment can resolve a missing
historical source.

## Sources

- C. Y. Lee, [*An Algorithm for Path Connections and Its Applications*](https://janders.eecg.utoronto.ca/1387_2015/readings/lee.pdf),
  *IRE Transactions on Electronic Computers* 10(3), 346--365, 1961. Primary
  paper; especially the introduction, Sections II--III, the minimal-distance
  illustrations, and the wave discussion.
- E. F. Moore, *The Shortest Path Through a Maze*, in *Proceedings of the
  International Symposium on the Theory of Switching, Part II*, Harvard
  University Press, 285--292, 1959. Bibliographic metadata confirmed through
  [CiNii](https://cir.nii.ac.jp/crid/1570854175170619520) and Lee's reference
  list; full text not inspected in this pass.
- E. L. Lawler, [*Combinatorial Optimization: Networks and Matroids*](https://www.nzdr.ru/data/media/biblio/kolxoz/M/MA/MAc/Lawler%20E.L.%20Combinatorial%20optimization..%20networks%20and%20matroids%20%281976%29%28384s%29.pdf),
  1976, comments to the shortest-path section. Secondary description retained
  only as a lead to claims that require checking against Moore's original.
- Alexander Schrijver, [*On the History of Combinatorial Optimization (Till
  1960)*](https://homepages.cwi.nl/~lex/files/histco.pdf), historical chapter,
  section “The Bellman-Ford method: Moore.” Secondary reconstruction of
  Algorithms A--D and Algorithm A's layer-labeling rule; not a replacement for
  Moore's paper.
- [Google Books record](https://books.google.com/books/about/The_Shortest_Path_Through_a_Maze.html?id=IVZBHAAACAAJ)
  and [WorldCat record](https://search.worldcat.org/title/shortest-path-through-a-maze/oclc/83616630)
  for the eight-page Bell Telephone System monograph; metadata only.
