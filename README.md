# Maksakuvien semanttinen segmentointi (Liver Medical Image Semantic Segmentation)

Semanttinen segmentointi on kuvankäsittelytekniikka, jossa kuva jaetaan merkityksellisiin osiin siten, että jokaiselle pikselille määritellään oma luokka. Tämä mahdollistaa esimerkiksi maksan erottamisen muista elimistä ja kudoksista lääketieteellisissä kuvissa. Manuaalinen segmentointi on kuitenkin hidasta ja altis inhimillisille virheille, joten automaattiset segmentointimenetelmät tarjoavat mahdollisuuden vähentää ihmisten tekemiä virheitä ja nopeuttaa prosessia.

Tässä projektissa keskitytään kuvien segmentointiin, jossa tavoitteena on jakaa kuvat tarkasti eri alueisiin ja objekteihin pikselitasolla. Kuvasegmentointi on erityisen tärkeää lääketieteellisessä kuvantamisessa, jossa kudosten, elinten ja mahdollisten sairauksien tunnistaminen on olennaista sekä diagnoosin että hoidon suunnittelun kannalta. Tarkka segmentointi mahdollistaa kuvista saadun tiedon hyödyntämisen esimerkiksi automaattisessa analyysissä, jossa tietokoneavusteiset työkalut voivat tukea lääkäreiden työtä.

Syväoppiminen ja erityisesti U-Net-arkkitehtuuri ovat osoittautuneet erittäin tehokkaiksi lääketieteellisten kuvien segmentoinnissa. U-Net on konvoluutio-neuroverkko, joka on suunniteltu nimenomaan lääketieteellisten kuvien analysointiin. Sen rakenne yhdistää alas- ja ylöspäin suuntautuvat osat, mikä mahdollistaa sekä laajojen että yksityiskohtaisten piirteiden hyödyntämisen tarkassa segmentoinnissa. U-Net pystyy erottamaan kuvan alueet tarkasti pikselitasolla ja soveltuu erityisen hyvin tilanteisiin, joissa harjoitusdataa on saatavilla rajallisesti. 

Projektin tavoitteena on rakentaa U-Net-arkkitehtuuriin perustuva neuroverkkomalli, joka kykenee saavuttamaan korkean tarkkuuden segmentointitulokset ja toimimaan luotettavasti myös pienellä datamäärällä.

# Toteutus
Mallimme on toteutettu Python-ohjelmointikielellä hyödyntäen PyTorch-koneoppimiskirjastoa. Mallin toteutus on jaettu kolmeen erilliseen tiedostoon:
1.	Main.py - käynnistää koulutusprosessin ja luo raportin.
2.	Model.py - sisältää mallin arkkitehtuurin ja logiikan mallin kouluttamiseksi erilaisilla hyperparametriyhdistelmillä.
3.	data_handler.py - vastaa datan esikäsittelystä.
 
Malli perustuu U-Net-arkkitehtuuriin, ja sen rakenne koostuu seuraavista osista:
1.	Konvoluutiolohkot: conv_block- funktio luo konvoluutiolohkoja, joissa on kaksi peräkkäistä konvoluutiokerrosta kernelikoolla 3x3 ja reunuksella 1. Jokaisen konvoluutiokerroksen jälkeen seuraa erän normalisointi (Batch Normalization) ja ReLU-aktivaatio. Ylikouluttamisen estämiseksi dropout-kerros on lisätty ensimmäisen aktivointikerroksen jälkeen. 
2.	Alasajovaihe (kontraktio): Kerrokset conv1–conv4 muodostavat alasajovaiheen. Jokainen kerros sisältää konvoluutiolohkon, jota seuraa maksimipoolauskerros pool1–pool4, jossa kernelikoko on 2x2 ja askellus 2. Tämä vaihe pienentää kuvan spatiaalista resoluutiota asteittain samalla, kun piirteiden (kanavien) määrä kasvaa. Tämä mahdollistaa sen, että verkko oppii korkeamman tason piirteitä. 
3.	Pullonkaulakerros (bottleneck): conv5 toimii pullonkaulana alasajo- ja ylösajovaiheiden välillä. Se koostuu konvoluutiolohkosta, joka sisältää kaksi konvoluutiokerrosta, erän normalisoinin (Batch Normalization), ReLU-aktivoinnin ja dropout-kerroksen.
4.	Ylösajovaihe (ekspansio): Transponoidut konvoluutiokerrokset up6–up9 ja konvoluutiolohkot conv6–conv9 muodostavat ylösajovaiheen. Tässä vaiheessa kuvan spatiaalinen resoluutio palautetaan asteittain alkuperäiseen kokoonsa. Lisäksi ylösajovaiheessa hyödynnetään skip connection -yhteyksiä, joissa alasajovaiheen ominaisuuskartat yhdistetään ylösnäytettyihin ominaisuuskarttoihin. Tämä mahdollistaa hienojakoisempien piirteiden säilyttämisen ja parantaa segmentointituloksia.

5.	Lähtökerros: Lopuksi kerros conv10 on 1x1 konvoluutio, joka muuntaa ominaisuuskartan haluttuun lähtökanavien määrään, tässä tapauksessa yhteen.

# Mallin ajaminen

Tässä ohjeessa käydään läpi tarvittavat vaiheet mallin ajamiseksi.

## 1. Tarvittavien kirjastojen asennus

Asenna tarvittavat kirjastot:

```bash
pip install torch torchvision pillow matplotlib
```
## 2. Datasetin jakaminen
Jos dataa ei ole vielä jaettu koulutus- ja testijoukkoihin, voit suorittaa datasetin jakamisen data_handler.py -tiedostossa.

```
if __name__ == "__main__":
    base_dir = 'Liver_Medical_Image_Datasets'
    split_dataset(base_dir)
```
## 3. Mallin koulutus

Kun data on jaettu, voit aloittaa mallin koulutuksen ajamalla `main.py`-tiedoston seuraavasti:

```bash
python main.py
```

Hyperparametrit voidaan määritellä `param_space`-lohkoon, josta malli valitsee satunnaisesti kokeiltavat hyperparametriyhdistelmät.
```
param_space = {
    'batch_size': [4, 8, 16],
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'optimizer_type': ['Adam', 'SGD'],
    'num_epochs': [10, 15, 20, 25]
}
```
Erilaisten hyperparametriyhdistelmien avulla luotavien mallien lukumäärä voidaan määrittää seuraavalla parametrilla:
`num_trials = 10`

`param_space` ja `num_trials = 10` löytyvät `main.py`- tiedoston alusta. 