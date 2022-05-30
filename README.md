# discopro
A package to support running QNLP experiments involving anaphora. This package is meant to be used with [discopy](https://github.com/oxford-quantum-group/discopy). An interactive demo can be found [here](http://discox.herokuapp.com/).

### Installation
#### Method 1
```
pip install git+https://github.com/kinianlo/discopro.git --upgrade
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
![9eed51d2-d86c-42b2-aa82-63ade877ab2d](https://user-images.githubusercontent.com/3414912/170871052-72b528dd-bfb8-4466-8f62-74edbb74ef4a.png)



One of the main usage of `discopro` is to connect the pronoun (He) and the referent (John) with a `Cup(N, N.r)`:
```
# Connect the pronoun `He` to the referent `Bob`
from discopro.anaphora import connect_anaphora
ref_box_idx, pro_box_idx = 0, 3
diagram_connected = connect_anaphora(diagram, pro_box_idx, ref_box_idx)
draw(diagram_connected)
```
![b278be70-5822-4469-81bf-bb4f5a232c6b](https://user-images.githubusercontent.com/3414912/170871087-25f03829-6e3e-4d86-a180-93be6e4102fe.png)


Note that some of the swaps introduced are unnecessary. One can use the `min_swaps` option in the `connect_anaphora` function to prevent unnecessary swaps.
```
diagram_connected = connect_anaphora(diagram, pro_box_idx, ref_box_idx, min_swaps=True)
draw(diagram_connected)
```
![7bba73ec-4c15-40dd-a647-54dbed0932b7](https://user-images.githubusercontent.com/3414912/170871105-965fbdc5-c554-400b-bfd4-3b9425cc499f.png)

## Remove cups
Another usage of `discopro` is to remove cups by flipping the word boxes. Sometimes swaps are also removed as a side effect.
```
from discopro.rewriting import contract
diagram_connected = contract(diagram_connected)
diagram_connected.draw()
```
![ba94e138-ad87-402f-bd72-b48803412c1d](https://user-images.githubusercontent.com/3414912/170871120-1806d5a9-f1f3-482d-be7d-b1f3caa763c2.png)


Note that three cups were removed.

Swaps are preventing the algorithm from removing more cups. One can connect the anaphora with the referent on the top of the diagram without introducing any swaps in the first place.
```
from discopro.anaphora import connect_anaphora_on_top
diagram_connected = connect_anaphora_on_top(diagram, 3, 0)
diagram_connected.draw()
```
![b1f1b3ee-2186-47b9-b92c-16e1b486ab91](https://user-images.githubusercontent.com/3414912/170871163-ef974f86-0a3b-4777-b20c-451d83270f66.png)

Note that doing so would render the diagram not being in the pregroup form. Now try to contract the connected diagram:
```
contract(diagram_connected).draw()
```
![6fb70a87-7aba-435f-9403-5143649aa869](https://user-images.githubusercontent.com/3414912/170871175-5d635128-aa65-4d9d-8fee-41752a1f9972.png)

Notice that the word box for `Bob` also got flipped resulting in 2 more cups being removed.
