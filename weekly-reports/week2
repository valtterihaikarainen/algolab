
### Viikkoraportti 2

**Tällä viikolla käytetty aika**: *≈ 8–12 tuntia* (CMaken ja GoogleTestin asennus, `Tensor`-luokan toteutus, testit, dokumentaatio, testikattavuustyökalut)

-----

### 1\. Mitä tein tällä viikolla?

  * Määritin **CMake**-koontijärjestelmän C++17:llä, käännöslipuilla ja **GoogleTestillä** (`FetchContentin` kautta), jotta projekti voidaan konfiguroida ja testata toistettavasti.
  * Toteutin **numeerisen ytimen** alun: **`Tensor`**-luokan, jossa on yhtenäinen **rivienkertainen (row-major)** tallennus, **muoto (shape)** ja **askeleet (strides)**, **syväkopiointi- (deep copy)** ja **siirtosemantiikka (move)**, **indeksointi** (`at`, `offset`, `operator()`), **`fill`** ja **`reshape`** (säilyttäen elementtien määrän).
  * Lisäsin vapaan funktion **`compute_strides`** rivienkertaisten askelten (strides) laskentaan.
  * Kirjoitin **yksikkötestejä** askelten laskennalle ja perus **kopiointi/sijoitus**-toiminnalle (savutestejä ehkäisemään regressioita rajapinnan kasvaessa).
  * Lisäsin otsikkotiedostoon **Doxygen-tyyliset kommentit** (sekä lyhyen tiedostokommentin toteutukseen), jotta julkinen toiminta, parametrit ja virheet on dokumentoitu kurssin vaatimusten mukaisesti.
  * Dokumentoin kääntämisen, testien ajamisen ja **gcov/lcov**-kattavuuden keräämisen repositoryn **README**-tiedostoon, jotta testikattavuutta voidaan seurata samaan tapaan kuin kurssin Pythonin `coverage`-työnkulussa ([Yksikkötestaus — kattavuus](https://algolabra-hy.github.io/unittest-en#has-enough-testing-been-done-test-coverage)).

-----

### 2\. Miten ohjelma on edistynyt?

  * Projekti ei ole enää vain "paperilla": nyt on olemassa **käännettävä kirjasto** (`vadugrad`) ja **läpi menevä testipatteristo**.
  * **`Tensor`**-tyyppi tulee sisältämään aktivaatiot, painot ja gradientit tulevia **tiheitä (dense)** ja **perhoskerroksia (butterfly)** sekä [määrittelydokumentissa](docs/specification-doc.md) kuvattua **opetussilmukkaa** varten.
  * **Seuraavat** vaiheet ovat testien laajentaminen (`reshape`:n ja `fill`:n oikeellisuus, indeksointirajat, datan yhtäsuuruus kopioinnin jälkeen) ja **kerroskoodin (layer)** aloittaminen (esim. tiheän lineaarisen kerroksen myötä- ja vastasuuntaiset laskennat eli forward/backward-vaiheet) tämän tallennusrakenteen päälle.

-----

### 3\. Mitä opin tällä viikolla / tänään?

  * Käytännön yksityiskohtia **RAII**:sta ja **viiden säännöstä (rule of five)** C++:ssa, kun omistetaan raakapuskureita moniulotteista taulukkoa varten.
  * Miten **rivienkertaiset askeleet (row-major strides)** johdetaan ja miten niitä käytetään **tasaisessa indeksoinnissa (flat indexing)**.
  * **GoogleTestin** integroinnin CMakeen ja **gcov**-yhteensopivien käännösten valmistelemisen **rivi-/haarakattavuusraportteja (line/branch coverage)** varten.

-----

### 4\. Mikä on jäänyt epäselväksi tai on ollut haastavaa?

  * **Siistien ja tehokkaiden rajapintojen (API)** suunnittelu **eräajo-operaatioille (batched, mini-batches)** minimaalisen `Tensor`-luokan päälle ilman suurta ulkoista tensorikirjastoa.
  * **Perhosvaiheen (butterfly)** indeksoinnin yhdistäminen tähän tallennusrakenteeseen rakenteellisen kerroksen myötä- ja vastasuuntaisia vaiheita toteutettaessa – tämä tulee vaatimaan huolellisia kaavioita ja testejä.

-----

### 5\. Mitä teen seuraavaksi?

  * Lisään **vahvempia yksikkötestejä** `Tensor`-luokalle (arvot kopioinnin jälkeen, `reshape`-invariantit, rajojen ulkopuoliset indeksit).
  * Aloitan **tiheän lineaarikerroksen (dense linear layer)** ja sen **vastasuuntaisen vaiheen (backward pass)** toteuttamisen, ja testaan niitä pienillä, käsin lasketuilla matriiseilla.
  * Lisään mahdollisesti ajoissa **pienen CLI-aloituspisteen**, jotta projekti pysyy "ajettavana" [viikon 3 virstanpylvään](https://algolabra-hy.github.io/schedule-en) lähestyessä.

-----