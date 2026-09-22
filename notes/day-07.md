Yesterday’s 2.28 NLL was measured on the same names the model trained on. That number can look good because the model memorised. Today we hide 20% of the names before training and only use them to score. If hidden-name loss is much worse than training loss, the model memorised (called overfitting). If both are under ~2.45, extra context is actually helping.

What we changed vs yesterday:
We no longer score the model on the same names it trained on.

How the split works:
We shuffle the 32,033 names (with no seed, I'm trading off reproduceability for experimentation).
80% training names, 10% validation names, 10% test names.
We build (three previous letters → next letter) examples separately
for each list so a name cannot appear on both sides.

Learning rate:
I swept rates from 0.001 to 1.
Loss fell fastest without exploding around 0.111.
That is the rate I trained with.

Results (hidden layer width = 100, embedding dimensions = 2, learning rate = 0.111, training runs = 500000):
Training loss (all training rows, end of run): 2.1879
Validation loss (all validation rows): 2.2034
Test loss (computed once, after I stopped changing things): 2.2159
Twenty sample names:
abraycen.
aulen.
hamoder.
yamira.
keline.
anie.
khyel.
brebanna.
e.
mykan.
sasna.
mecc.
keylyn.
jun.
die.
juge.
locessenta.
domperji.
emmarlie.
ahifoberley.

Results (hidden layer width = 300, embedding dimensions = 10, learning rate = 0.111, training runs = 100000):
Training loss (all training rows, end of run): 2.1121
Validation loss (all validation rows): 2.1856
Test loss (computed once, after I stopped changing things): 2.1745
Twenty sample names:
dau.
sta.
charia.
marci.
meah.
jai.
alah.
jian.
wiricston.
faal.
valco.
zaia.
kenzinake.
karaysh.
maranvi.
adzelisen.
jayvenchida.
ison.
melijah.
gatemiah.

Argmax names (need to give at least the first letter in the context window otherwise we get the same name):
.ana.
ana.
bran.
carianna.
daylan.
elin.
farianna.
garrie.
harianna.
isa.
jaylan.
kaylan.
lan.
marianna.
naya.
olin.
parianna.
quin.
raylan.
samarianna.
tai.
uri.
van.
warianna.
xarianna.
yanna.
zarianna.

What the gap means:
If validation is close to training, the model is not just memorising.
If validation is much higher, it memorised training names.

What I still find shaky, in full sentences:
- Why does a single training run give us confidence in the learning rate decay with a single loss (maybe the loss for that particular run/descent was low but over the long term the learning rate might not be optimum?)