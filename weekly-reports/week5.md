### Viikkoraportti 5

**Tällä viikolla käytetty aika**: *≈ 8–10 tuntia* (vertaisarviointi, palautteen läpikäynti, perhoskerros ja MNIST takaisin repositorioon, dokumentaatio)

-----

### 1. Mitä tein tällä viikolla?

- Tein ensimmäisen vertaisarvioinnin kurssin ohjeiden mukaisesti.
- Kävin [issue #1](https://github.com/valtterihaikarainen/algolab/issues/1) -palautteen läpi. Tärkein huomio: määrittelydokumentti kuvasi perhos-MLP:n, mutta viikon 4 julkisessa tilassa näkyi lähinnä transformer.
- Lisäsin repositorioon **perhoskerroksen** (`ButterflyLinear`), MNIST-laturin ja `train_mnist`-demon sekä testit, jotka kuuluivat alkuperäiseen ytimeen.
- Päivitin määrittelyyn lyhyen **nykyisen laajuuden** ja toteutusdokumentin (mm. versionhallinta tältä syksyltä).
- CMake: Debug-build ei enää pakota `-O2`; testit voi ohittaa (`-DBUILD_TESTING=OFF`); GoogleTest `v1.18.0`.

-----

### 2. Miten ohjelma on edistynyt?

- Ydintoiminta on nyt linjassa speksin kanssa: tensoriydin, tiheä kerros, perhoskerros, MNIST, transformer-demo ja manuaalinen backprop.
- Testit kattavat butterfly-ekvivalenssin, MNIST-laturin (pieni IDX-fixture) sekä aiemmat tensori-/transformer-testit.

-----

### 3. Mitä opin tällä viikolla / tänään?

- Vertaisarviointi menee pieleen, jos speksi ja `main` kertovat eri tarinaa; speksi pitää päivittää kun laajuus laajenee, ei vain kun työ alkaa.
- Pienet CMake-liput (optimointi vs Debug, FetchContent vain testeille) vaikuttavat siihen, voiko arvioija kääntää kirjaston ilman verkkoa.

-----

### 4. Mikä on jäänyt epäselväksi tai ollut haastavaa?

- Attention-silmukoiden `Tensor::operator()`-kustannus ja ylimääräiset flatten-kopiot ovat edelleen tunnettuja pullonkauloja; korjaan niitä tarvittaessa viikolla 6.
- `lcov` GCC 15:llä vaatii `--ignore-errors inconsistent` README:ssa.

-----

### 5. Mitä teen seuraavaksi?

- Vastaan issue #1:een GitHubissa ja viimeistelen dokumentit lopullista palautusta varten.
- Lisään README:een lcov-ohjeen, joka toimii arvioijan ympäristössä.

-----
