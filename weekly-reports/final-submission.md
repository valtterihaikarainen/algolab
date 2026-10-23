### Viikkoraportti 7

**Tällä viikolla käytetty aika**: *≈ 80 tuntia* (projektin rajaus, koe, testit, dokumentaatio)

-----

### 1. Mitä tein tällä viikolla?

- Rajasin projektin uudelleen määrittelyn ytimeen: **voiko perhoskerros korvata tiheän kerroksen pienessä neuroverkossa?** Poistin transformerin pääkoodista, koska se ei kuulunut alkuperäiseen suunnitelmaan ja hämärsi vertaispalautteessakin huomattua kokonaiskuvaa.
- Toteutin kokeen verkon `Mlp` (784 → 512 → ReLU → piilokerros → ReLU → 10), jossa piilokerros on `none`, `dense` tai `butterfly`, sekä koko verkon läpi menevän manuaalisen backpropagaation.
- Laajensin `train_mnist`-ohjelmaa: komentoriviasetukset, siemen, CSV-loki per epookki, testijoukon arviointi ja ajanotto. Lisäsin skriptit MNISTin lataukseen (`scripts/download_mnist.sh`) ja tulosten piirtämiseen (`scripts/plot_results.py`).
- Ajoin kokeen: 3 siementä × 3 varianttia × 5 epookkia, ensin Linuxilla (GCC) ja uudelleen macOS:llä (Clang). Tulokset ovat kansioissa `results/` ja `results/macos/`.
- Kirjoitin kaikki dokumentit uudelleen vastaamaan lopullista tilaa: määrittely, toteutus, testaus, käyttöohje ja README.

-----

### 2. Miten ohjelma on edistynyt?

- Löysin ja korjasin kolme merkittävää virhettä:
  - **Alustus:** kaikki painot alustettiin keskihajonnalla 0.02, jolloin signaali kutistui jokaisessa perhosvaiheessa noin kertoimella 0.03. Yhdeksän vaiheen jälkeen ulostuloon ei päätynyt mitään, ja perhosverkko jäi arvaustasolle (10 %). Korjaus: He-alustus ja perhoslohkoille keskihajonta 1/√2, joka säilyttää varianssin.
  - **Nopeus:** `Tensor::operator()` varasi jokaisella alkion luvulla uuden `std::vector<int>`:n. Allokaatiovapaat indeksointifunktiot ja välimuistiystävällinen `matmul` nopeuttivat koulutusta noin 120-kertaisesti. Ennen korjausta yksi epookki olisi kestänyt noin 32 minuuttia.
  - **Adam:** kaikilla parametreilla oli yhteinen askellaskuri, mikä vääristi bias-korjausta. Nyt jokaisella parametrilla on oma tilansa.
- Testejä on nyt 56 (viikolla 6: 37), ja lähdekoodin rivikattavuus on 94 % GCC:llä ja 95 % Clangilla (viikolla 6: 85 %). Uutta:
  - koko verkon päästä päähän -gradienttitarkistus kaikille kolmelle variantille
  - Walsh–Hadamard-testi leveydellä 512
  - gradienttitarkistukset myös perhoskerroksen syötteelle ja biasille
  - integraatiotesti synteettisellä datalla
- Tarkistin testien herkkyyden istuttamalla tyypillisen virheen (perhoslohkon transpoosi unohtuu backwardissa). Gradienttitestit huomasivat sen, vanhat testit eivät olisi huomanneet.
- Hypoteesi piti: perhosvariantti jäi tiheästä vain 0,36 (Linux) ja 0,17 (macOS) prosenttiyksikköä 27 kertaa pienemmällä parametrimäärällä.

-----

### 3. Mitä opin tällä viikolla / tänään?

- Asymptottiset hyödyt eivät yksin ratkaise. Perhoskerros tekee 28 kertaa vähemmän laskutoimituksia, mutta oli GCC:llä vain noin 3 kertaa ja Clangilla vain noin 1,15 kertaa nopeampi kuin tiheä kerros. Muistin käsittelytapa, rajatarkistukset ja välitulosten kopiointi syövät edun leveydellä 512, ja vaikutus riippuu paljon kääntäjästä.
- Alustus ratkaisee, toimiiko syvä tai monivaiheinen rakenne lainkaan. Varianssia säilyttävä alustus on välttämätön, kun vaiheita on monta.
- Gradienttitarkistus on paras työkalu käsin kirjoitetun backpropin testaamiseen. Väärän transpoosin sisältävä verkko oppii silti, vain huonommin, joten pelkät muototestit tai "loss laskee" -testit eivät riitä.
- Sama siemen ei takaa samoja tuloksia eri alustoilla: `std::shuffle` on toteutuskohtainen, joten libstdc++ ja libc++ sekoittavat datan eri järjestykseen.

-----

### 4. Mikä on jäänyt epäselväksi tai ollut haastavaa?

- MNIST on liian helppo tähän vertailuun: `none`-perustaso on yhtä tarkka kuin `dense`, joten koe näyttää vain, ettei perhoskerros juuri haittaa. Se ei näytä, riittääkö perhoskerros silloin, kun lisäkapasiteettia oikeasti tarvitaan. Vaikeampi aineisto (esim. Fashion-MNIST) tai kapeampi ensimmäinen kerros tekisi erosta mitattavan.
- Kolme siementä ei riitä erottamaan 0,1–0,4 prosenttiyksikön eroja. Erot ovat siemenkohtaisen hajonnan suuruisia.
- Perhoskerroksen toteutus on hidas operaatiota kohden. Optimointi uudelleenkäytettävillä buffereilla jäi tietoisesti tekemättä.

-----

### 5. Mitä teen seuraavaksi?

- Palautan lopullisen version ja varmistan, että GitHub-repositorio vastaa dokumentteja.
- Valmistelen demotilaisuuden esityksen.
- Jos jatkaisin projektia, seuraavat askeleet olisivat:
  - Fashion-MNIST-vertailu
  - alustasta riippumaton sekoitus (oma Fisher–Yates xorshift-generaattorilla)
  - perhoskerroksen nopeuttaminen
