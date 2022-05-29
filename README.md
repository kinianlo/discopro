# discopro
A package to support running QNLP experiments involving anaphora. This package is meant to be used with [discopy](https://github.com/oxford-quantum-group/discopy).

### Installation
#### Method 1
```
pip install git+https://github.com/kinianlo/discopro.git
```

#### Method 2
Clone the repository with:
```
git clone https://github.com/kinianlo/discopro.git
```
Go into the repository directory:
```
cd discopro
```
Install the package with `pip`:
```
pip install . --upgrade
```
## Demo
### Connecting pronoun to referent
Consider a sentence that contains a anaphora, such as, `Bob likes Alice. He was happy.`. 
```
from discopy import Word, Ty, Id, Cup
from discopy.grammar import draw

# Define grammartical (pregroup) types
N = Ty('n') # noun 
S = Ty('s') # sentence

# Define the words 
bob = Word('Bob', N)
likes = Word('likes', N.r @ S @ N.l)
alice = Word('Alice', N)
he = Word('He', N)
was = Word('was', N.r @ S @ S.l @ N)
happy = Word('happy', N.r @ S)

# Define a discopy diagram 
diagram = bob @ likes @ alice @ he @ was @ happy 
diagram >>= Cup(N, N.r) @ Id(S) @ Cup(N.l, N) @ Cup(N, N.r) @ Id(S) @ diagram.cups(S.l @ N, N.r @ S)
draw(diagram)
```
![image](https://user-images.githubusercontent.com/3414912/156348380-a2ef4682-0d48-47a5-8de8-b31681d33e78.png)

One of the main usage of `discopro` is to connect the pronoun (He) and the referent (John) with a `Cup(N, N.r)`:
```
# Connect the pronoun `He` to the referent `Bob`
from discopro.anaphora import connect_anaphora
ref_box_idx, pro_box_idx = 0, 3
diagram_connected = connect_anaphora(diagram, pro_box_idx, ref_box_idx)
draw(diagram_connected)
```
![image](https://user-images.githubusercontent.com/3414912/156349015-47911290-4c28-485b-b4bb-9674d5a8fc37.png)
Note that some of the swaps introduced are unnecessary. One can use the `min_swaps` option to the `connect_anaphora` function to prevent unnecessary swaps.
```
diagram_connected = connect_anaphora(diagram, pro_box_idx, ref_box_idx, min_swaps=True)
draw(diagram_connected)
```

## Remove cups
Another usage of `discopro` is to remove cups by flipping the word boxes. Sometimes swaps are also removed as a side effect.
```
from discopro.rewriting import contract
diagram_connected = contract(diagram_connected)
diagram_connected.draw()
```
![image](https://user-images.githubusercontent.com/3414912/156356286-afc50cd2-07d8-47e9-9c41-23c52d5dabc9.png)

Note that three cups were removed.

Swaps are preventing the algorithm from removing more cups. One can connect the anaphora with the referent on the top of the diagram without introducing any swaps in the first place.
```
from discopro.anaphora import connect_anaphora_on_top
diagram_connected = connect_anaphora_on_top(diagram, 3, 0)
diagram_connected.draw()
```
However, doing so would render the diagram not being in the pregroup form. Now try to contract the connected diagram:
```
contract(diagram_connected).draw()
```
Notice that the word box for `Bob` also got flipped resulting in 2 more cups being removed.
