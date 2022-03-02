# discopro
A package to support running QNLP experiments involving anaphora. This package is meant to be used as a plugin to [discopy](https://github.com/oxford-quantum-group/discopy).

### Installation
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
Consider a sentence that contains a anaphora, such as, `John sleeps. He snores`. 
```
from discopy import Word, Ty, Id, Cup

# Define grammartical (pregroup) types
N = Ty('n') # noun 
S = Ty('s') # sentence

# Define the words 
john = Word('John', N)
sleeps = Word('sleeps', N.r @ S)
he = Word('He', N)
snores = Word('snores', N.r @ S)

# Define a discopy diagram 
diagram = john @ sleeps @ he @ snores >> Cup(N, N.r) @ Id(S) @ Cup(N, N.r) @ Id(S)
diagram.draw()
```
![image](https://user-images.githubusercontent.com/3414912/156348380-a2ef4682-0d48-47a5-8de8-b31681d33e78.png)

One of the main usage of `discopro` is to connect the pronoun (He) and the referent (John) with a `Cup(N, N.r)`:
```
from discopro.anaphora import connect_anaphora
ref_box_idx, pro_box_idx = 0, 2
diagram = connect_anaphora(diagram, pro_box_idx, ref_box_idx)
diagram.draw()
```
![image](https://user-images.githubusercontent.com/3414912/156349015-47911290-4c28-485b-b4bb-9674d5a8fc37.png)

## Remove cups
Another usage of `discopro` is to remove cups (and sometimes to remove swaps as a side effect). Consider another sentnce: `The fish ate the worm. It was tasty`:
```
from discopy import Word, Ty, Id, Cup

# Define grammartical (pregroup) types
N = Ty('n') # noun 
S = Ty('s') # sentence

# Define the words
the = Word('the', N @ N.l)
fish = Word('fish', N)
ate = Word('ate', N.r @ S @ N.l)
worm = Word('worm', N)
it = Word('it', N)
was = Word('was', N.r @ S @ S.l @ N)
tasty = Word('tasty', N.r @ S)

# Define the discopy diagram
diagram = the @ fish @ ate @ the @ worm @ it @ was @ tasty
diagram >>= Id(N) @ Cup(N.l, N) @ Id(N.r @ S) @ Cup(N.l, N) @ Cup(N.l, N) @ Cup(N, N.r) @ Id(S @ S.l) @ Cup(N, N.r) @ Id(S)
diagram >>= Cup(N, N.r) @ Id(S @ S) @ Cup(S.l, S)
diagram.draw()
```
![image](https://user-images.githubusercontent.com/3414912/156356208-1d34ef2b-548c-49e7-9886-4bfb2b95e8e0.png)

Use `rewriting.contract` to remove cups:
```
from discopro.rewriting import contract
diagram = contract(diagram)
diagram.draw()
```
![image](https://user-images.githubusercontent.com/3414912/156356286-afc50cd2-07d8-47e9-9c41-23c52d5dabc9.png)
