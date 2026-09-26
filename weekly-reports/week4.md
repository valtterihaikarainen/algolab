
### Viikkoraportti 4

**Tällä viikolla käytetty aika**: *≈ 14–18 tuntia* (transformer-ydin, koulutuksen apufunktiot, integraatiotestit, dokumentointi vertaisarviointia varten)

-----

### 1. Mitä tein tällä viikolla?

- Toteutin projektin rungon MLP-pohjaisesta ratkaisusta kohti **transformer-arkkitehtuuria**:
  - `MultiHeadAttention` (self + cross, kausaalimaski, manuaalinen backward)
  - `LayerNorm`, `FeedForward` (GELU), `Embedding`, `TransformerBlock`
  - `DecoderOnlyTransformer` (useita lohkoja, final layer norm, LM head)
- Lisäsin **ergonomia-/apuoperaatiot** tensorikäsittelyyn:
  - `flatten_bt`, `unflatten_bt`, `split_heads`, `merge_heads`, `add3d`
  - vakaa `softmax`/`log_softmax` ja `cross_entropy` + gradientti
- Lisäsin **optimointitukea**:
  - `Adam`-optimoija
  - mallitasolle `initialize_parameters(...)` ja `apply_gradients(...)`
- Lisäsin **ajettavan koulutusdemon** `train_lm` (synteettinen next-token -tehtävä), jolla häviön pieneneminen näkyy.
- Laajensin testikantaa:
  - moduulitestit NN-operaatioille ja LayerNormille
  - decoder-mallin forward/backward-muototesti
  - integraatiotesti: pieni erä ylisovitetaan ja häviö pienenee

-----

### 2. Miten ohjelma on edistynyt?

- Ohjelman ydintoiminta on nyt **lähes valmis vertaisarviointiin**: projektissa on toimiva decoder-only-transformer, manuaalinen backward ja optimointisilmukan minimituki.
- Koodi on nyt jaettu selkeämmin osa-alueisiin (`core`, `nn`, `optim`), mikä helpottaa lukemista ja palautteen antamista.
- Testit kattavat sekä yksittäisiä laskentablokkeja että end-to-end -tasoa.

-----

### 3. Mitä opin tällä viikolla / tänään?

- Miten nopeasti shape-/maskivirheet kertautuvat attention-ketjussa, ja miksi selkeät apufunktiot (`split_heads` jne.) parantavat luotettavuutta.
- Miten tärkeää on pitää backward-polku **moduulirajojen mukaisena** (blockkohtaiset gradientit oikeassa järjestyksessä optimointia varten).
- Miten integraatiotesti (häviö laskee pienellä datalla) täydentää yksikkötestejä paremmin kuin pelkät shape-assertit.

-----

### 4. Mikä on jäänyt epäselväksi tai ollut haastavaa?

- Suorituskyky ei vielä ole priorisoitu; toteutus on tarkoituksella selkeyttä painottava (paljon eksplisiittisiä silmukoita).
- Kaikkia tuotantotason ominaisuuksia ei vielä ole (esim. dropout, checkpointing, oikea dataloader tekstikorpukselle).
- Kattavuusraportin prosenttiluvut tulee päivittää uudelleen jokaisen suuremman refaktorin jälkeen.

-----

### 5. Mitä teen seuraavaksi?

- Viimeistelen dokumentaation (toteutus + testaus + käyttöohje) lopulliseen palautusmuotoon.
- Lisään tarvittaessa pienen tekstiaineistoon perustuvan harjoitusajon (`train_text_lm`) synteettisen demon rinnalle.
- Teen suorituskykyvertailun ja kokoan toteutusdokumenttiin puutteet/parannusehdotukset priorisoituna listana.

-----
