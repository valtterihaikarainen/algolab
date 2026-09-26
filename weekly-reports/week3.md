
### Viikkoraportti 3

**Tällä viikolla käytetty aika**: *≈ 10–14 tuntia* (tiheä kerros, CLI, testit, sum-korjaus, testausdokumentti)

-----

### 1. Mitä tein tällä viikolla?

- Toteutin **tiheän lineaarikerroksen** (`DenseLinear`): eteenpäin @f$y = xW + b@f$ (bias leviää eräkoon yli) ja **manuaalisen backpropin** lähtöön @f$\partial L/\partial x@f$, @f$\partial L/\partial W@f$, @f$\partial L/\partial b@f$ käyttäen olemassa olevia `matmul`, `transpose2d` ja `sum` -operaatioita.
- Korjasin **`sum`-funktion ulostulomuodon**: kun `keepdim` on false, reduoitu dimensio poistetaan oikein (aiemmin muoto säilyi virheellisesti).
- Lisäsin **ajettavan ohjelman** (`vadugrad_cli`, ulostulo `build/vadugrad`), joka tulostaa kiinteän 2→3-demoeteenpäin- ja gradienttiarvot; `--help` näyttää lyhyen käyttöohjeen.
- Laajensin **yksikkötestejä**: `fill`/kopio, `reshape`, virheelliset indeksit, tyhjä `compute_strides`-muoto; uudet testit tihelle kerrokselle (käsin lasketut arvot, erä-aggregointi gradienteissa).
- Aloitin **testausdokumentin** ([docs/testing.md](../docs/testing.md)): testien ajaminen, gcov-kattavuuden kerääminen ja taulukko esimerkkikattavuudesta `tensor.cpp` / `dense_linear.cpp`.

-----

### 2. Miten ohjelma on edistynyt?

- Projektilla on nyt **käyttöliittymä** kurssin mielessä: komentoriviohjelma, jolla työn voi ajaa ja tulkita ilman pelkkää testibinaaria.
- **Ydintoiminnan** kannalta ensimmäinen opetettava kerros on käytettävissä sekä eteen- että taaksepäin; seuraavat loogiset askeleet ovat aktivointi (esim. ReLU), häviö (softmax + cross-entropy) ja yksinkertainen optimointi.

-----

### 3. Mitä opin tällä viikolla / tänään?

- Miten **batchen aggregointi** bias-gradientissa toteutuu `sum(..., axis=0)` -reduktiolla 2D-gradientille.
- **Markdown-viikkoraportin** listat kannattaa aloittaa rivin alusta (ilman sisennystä ennen `-` tai `*`), jotta renderöijät eivät yhdistä kappaleita yhdeksi riviksi.

-----

### 4. Mikä on jäänyt epäselväksi tai on ollut haastavaa?

- **gcov/lcov**-raportin siivoaminen niin, että vain oma lähdekoodi näkyy selkeästi HTMLissä (polkusuodatukset vaihtelevat ympäristöittäin).

-----

### 5. Mitä teen seuraavaksi?

- ReLU (tai tanh) ja **softmax + cross-entropy** eteen- ja taaksepäin.
- Yksinkertainen **SGD-päivitys** ja MNIST-erien luku; tarkkuuden mittaus testijoukossa.

-----
