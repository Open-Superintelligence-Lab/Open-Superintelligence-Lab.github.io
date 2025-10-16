---
hero:
  title: "Sitni Rekurzivni Model"
  subtitle: "Nova AI arhitektura rekurzivnog rasuđivanja"
  tags:
    - "⏱️ Tehnički Dubinski Prikaz"
    - "📄 Istraživački Članak"
---

## Nova AI Arhitektura Rasuđivanja

Kako model od 7M pobedi modele od 1T u Sudoku, Lavirintima, ARC-AGI

**Sitni Model Rasuđivanja (TRM)** koristi 2-slojni transformer (7M parametara) koji ponovo koristi iste slojeve stotine puta da razmišlja o problemima rekurzivno.

Pobedi 100x veće modele u Sudoku-Extreme, Lavirintima, ARC-AGI i još mnogo toga.

U ovom tutorijalu ćemo naučiti kako TRM funkcioniše i uraditi sopstvene eksperimente.

---

## TRM Arhitektura - Pregled

![Arhitektura Sitnog Rekurzivnog Modela](/content/tiny-recursive-model/images/tiny-recursive-model-architecture.png)
*Slika: Arhitektura Sitnog Rekurzivnog Modela koja prikazuje glavni blok obrade (4x transformer slojeva), kombinaciju ulaza pitanja (x), odgovora (y) i rasuđivanja (z), obradu izlaza za računanje gubitka, i mehanizam rekurzivnog ažuriranja koji iterativno poboljšava rasuđivanje i predviđanje kroz maksimalno 16 koraka.*

Dijagram iznad ilustruje kompletnu TRM arhitekturu. Model obrađuje tri ključne komponente:
- **Ulaz (x)**: Pitanje ili problem za rešavanje (npr. raspored lavirinta)
- **Predviđanje (y)**: Trenutni pokušaj odgovora modela
- **Latentno (z)**: Interno stanje rasuđivanja modela

Ove komponente se kombinuju i obrađuju kroz stek od 4 transformer sloja, sa izlazom koji se koristi za računanje cross-entropy gubitka. Ključna inovacija je mehanizam rekurzivnog ažuriranja na dnu, koji iterativno poboljšava i rasuđivanje (z) i predviđanje (y) kroz više koraka da postepeno poboljša rešenje.

---

## Kako TRM Funkcioniše

### Korak 1: Postavljanje

Hajde da treniramo TRM da reši lavirint.

**1. Predstavljanje Lavirinta kao Mreže:**
Prvo, prikazujemo lavirint kao mrežu brojeva. Svaka ćelija u mreži dobija broj.

-   `0` = Prazan put
-   `1` = Zid
-   `2` = Početna tačka
-   `3` = Krajnja tačka

Za konkretni primer, hajde da pratimo mali 3x3 lavirint.

-   **`x_input`** (Nerešen lavirint)
    ```
    [[2, 0, 1],
     [1, 0, 1],
     [1, 0, 3]]
    ```
-   **`y_true`** (Ispravno rešenje, sa `4` koje predstavlja putanju)
    ```
    [[2, 4, 1],
     [1, 4, 1],
     [1, 4, 3]]
    ```

**2. Tokenizacija:**
Termin **token** jednostavno znači jednu jedinicu naših podataka. U ovom slučaju, jedan broj u mreži (`0`, `1`, `2`, `3` ili `4`) je token. Da bismo olakšali mrežu za obradu, "odmotavamo" mrežu u dugu 1D listu.

Za naš 3x3 primer, mreža se odmotava u listu od 9 tokena.

**3. Ugrađivanje: Davanje Značenja Brojevima:**
Da bi model razumeo šta brojevi poput `4` i `1` znače, dodelićemo veliki **vektorski embedding** svakom. Vektorski embedding je dugačak vektor (niz brojeva) koji model može menjati da čuva informacije o zidu, praznom putu, itd.

Ovi vektori će predstavljati značenje "zida" ili "krajnje tačke".

Preporučujem da podsetite sebe šta su vektorski embeddingovi (u LLM-ovima, reči, tokena, itd) pretraživanjem na YouTube-u ili razgovorom sa AI chatbotom.

-   **Sloj Ugrađivanja** je kao rečnik.
-   Sadrži vektorske embeddingove za svaki od naših brojeva.
-   `1`: `[0.3, -1.2, 0.7, 0.0, 1.5, -0.4, 0.9, 2.3]`  ←  Primer vektorskog embeddinga za "zid"
-   **Izlaz:** Duga lista brojeva nazvana **vektor**. Ovaj vektor predstavlja *značenje* "zida" na način koji mreža može razumeti. Sama mreža bira (nauči) brojeve unutar ovog vektora tokom treninga tako da ga može "razumeti".

Nakon ovog koraka, naš ulazni lavirint više nije lista jednostavnih brojeva. To je lista vektora. Za naš 3x3 lavirint, ako koristimo vektor veličine 8 za svaki token, naš ulaz postaje:

-   `x`: Matrica `9x8` vektora koja predstavlja lavirint.

Ova bogata reprezentacija je ono što unosimo u glavni model.

---

### Korak 2: Osnovna Arhitektura: TRM Mozak

"Mozak" TRM-a je mali 2-slojni transformer nazvan `net`. On obrađuje informacije da bi proizveo izlaz. Da bi "razmišljao", TRM koristi dve promenljive, obe istog oblika kao `x`:

-   `y`: Trenutna **najbolja pretpostavka** modela za rešenje. Može biti pogrešna
```
[[2, 4, 1],
  [1, 4, 1],
  [1, 0, 3]]
```
-   `z`: **Latentna misao**. `z` govori šta treba promeniti u `y` da bi se pretvorilo u ispravno rešenje. `z` se propušta kroz transformer više puta da bi model poboljšao šta treba promeniti u `y`, tako model rasuđuje ili razmišlja. Zatim se promena primenjuje na `y`.

Za naš 3x3 primer, `z` i `y` počinju kao `9x8` matrice nula.

---

### Korak 3: Proces Učenja, Iznutra ka Spolja

TRM uči kroz seriju ugneždenih petlji. Počnimo od jezgra i gradimo naš put ka spolja.

#### Najdublja Petlja: `latent_recursion` (Osnovna Misao)

Ovde mali `net` (2-slojni Transformer) obavlja sav svoj posao. Proces je podeljen u dve faze koje se ponavljaju formirajući ciklus razmišljanja i poboljšanja.

**Faza A: Rasuđivanje (Ažuriranje Skraćenice `z`)**
Model "razmišlja" poboljšavajući svoj interni token planiranja `z`, u petlji od 6 koraka. Cilj je izgraditi sve bolji plan za promenu `y`.

1.  **Proces:** U svakom od 6 koraka, `net` prima tri ulaza:
    -   Sam lavirint (`x`).
    -   Trenutnu najbolju pretpostavku modela za rešenje (`y`) - ovo može biti sve nule na početku.
    -   Skraćenicu iz prethodnog koraka (`z`).
2.  **Kako radi:**
    -   **Kombinovanje Ulaza:** Tri ulaza se sabiraju element po element (`x + y + z`). Ovo kreira jedan niz bogatih vektora, gde svaki vektor (koji predstavlja ćeliju u lavirintu) sadrži kombinovane informacije o rasporedu lavirinta (`x`), trenutnoj pretpostavci (`y`), i procesu trenutne misli (`z`).
    -   **Razmišljanje sa Pažnjom:** Ovaj kombinovani niz se ubacuje u 2-slojni Transformer. Mehanizam samo-pažnje Transformera omogućava mu da pogleda sve ćelije odjednom i identifikuje odnose. Na primer, može videti kako "početna" ćelija ima odnos prema potencijalnoj ćeliji putanje, informisan ulaznim podacima `x` i rasuđivanjem `z`.
    -   **Generisanje Sledeće Misli:** Dva transformer sloja obrađuju ovu informaciju i ispisuju novi niz vektora identičnog oblika. Ovaj izlaz *je* novo `z`. Ne postoji zaseban "izlazni glava" da ga generiše; transformacija koju obavljaju dva sloja *je* čin kreiranja sledeće, poboljšane misli. Iako je ulaz bio zbir koji sadrži `x` i `y`, mreža uči da proizvede izlaz koji služi kao korisno novo `z` za sledeći korak.

    Ovaj proces se ponavlja 6 puta, što znači da se informacija propušta kroz ista dva sloja šest uzastopnih puta, postajući progresivno sofisticirana sa svakim prolaskom.
3.  **Primer Toka:** Nakon nekoliko prolaza kroz transformer, `z` može kodirati nisko-nivovske karakteristike poput lokacija zidova. Do šestog prolaza, može predstavljati visoko-nivo plan za ažuriranje odgovora (`y`).
   
   - Interesantno, ista 2 transformer sloja se koriste za detekciju nisko nivoskih karakteristika, pravljenje visoko nivoskog plana i kasnije za ažuriranje same `y`. Ova 2 sloja imaju višestruke namene, što je moć neuronskih mreža, može naučiti da obavi višestruke, manje povezane ili nepovezane transformacije koje zavise samo od ulaznih podataka.

**Faza B: Poboljšanje Odgovora (Ažuriranje Pretpostavke `y`)**
Nakon petlje rasuđivanja od 6 koraka, koristeći najnoviju latentnu misao `z`, model ažurira svoj odgovor `y`.

-   **Kako radi:** Kombinuje svoj prethodni odgovor (`y`) sa svojom finalnom, poboljšanom mišlju (`z`) sabrajući ih (`y + z`) i propušta rezultat kroz isti `net` još jednom. Izlaz je novo, poboljšano `y`.
    -   **Ključno, `x` nije uključeno u ovom koraku.** Ovo je namerna odluka dizajna koja govori jednom `net` koji zadatak treba obaviti.
    -   `x` je prisutno u rasuđivanju (`x + y + z`).
    -   `x` nedostaje u poboljšanju odgovora (`y + z`).

Razlog zašto sam rekao "poboljšanje odgovora" je zato što se ova 6+1 petlja dešava više puta, svaki put "razmišljajući" 6 prolaza i ažurirajući `y` jednom.

#### Srednja Petlja: `deep_recursion` (Potpun Proces Razmišljanja)

Sada kada razumemo kako rasuđivanje + petlja poboljšanja y radi, hajde da vidimo potpun proces razmišljanja od početka gde se cela ova petlja ponavlja 3 puta da bi se dobilo najbolje `y`.

Prethodno opisana unutrašnja petlja (6+1 koraka rasuđivanja i poboljšanja `y`) se izvršava `T` puta (npr., `T=3`). Stanje (`y` i `z`) se **prenosi** između ovih izvršavanja; ne resetuje se na nulu.

-   **Runda 1 (Zagrevanje):** Počinje sa praznim (sve nule) `y` i `z` (zapamtite, ovo je apsolutni početak procesa, tako da nema `y` i `z` za prenošenje). Izvršava potpunu unutrašnju petlju (6 rasuđivanja + 1 poboljšanje `y` koraka) da proizvede pametnije `y_1` i `z_1`. Ovo se radi u "bez-gradijent" modu za brzinu i uštede memorije - neuronska mreža ne uči ovde.
-   **Runda 2 (Zagrevanje):** Uzima `y_1` i `z_1` kao početnu tačku i izvršava unutrašnju petlju ponovo da proizvede još bolje `y_2` i `z_2`. Još uvek nema gradijenata i učenja.
-   **Runda 3 (Zaista):** Počinje sa dobro razmotrenim `y_2` i `z_2`, izvršava unutrašnju petlju još jednom, i ovaj put svi proračuni se prate tako da model može naučiti sa propagacijom unazad.

Ovaj proces zagrevanja modelove "misli" pre finalnog, učivog koraka je ključna optimizacija.

#### Najspoljnija Petlja: Još više petlji!

Model dobija višestruke "šanse" (do 16) da reši isti lavirint, i nakon svake šanse, poboljšava svoje `net` težine. Stanje (`y` i `z`) **se prenosi** sa jedne iteracije srednje petlje na sledeću, kao što je prikazano u pseudokodu rada. Dozvoljava modelu da dobije višestruke "šanse" (do 16) da reši isti lavirint, poboljšavajući se sa svakom.

Ovo je samo ponavljanje srednje petlje do 16 puta. Model može odlučiti da se zaustavi ranije od 16 ako oseća da je dobio ispravan odgovor.

Zašto nam treba ova petlja:

Nakon svake iteracije srednje petlje, ova spoljašnja petlja ažurira težine jednom (zapamtite da Runda 3 u srednjoj petlji radi propagaciju unazad).

Zatim u sledećoj iteraciji ponavlja srednju petlju sa ažuriranim težinama, dopuštajući modelu da postepeno poboljša svoje rešenje sa svakim pokušajem.

#### Znanje kada prestati da razmišlja (Q glava)

Spoljašnja petlja može raditi do 16 puta, ali ne mora. Bilo bi gubljenje vremena nastaviti razmišljati o lavirintu koji je već rešen.

Dakle, model ima mali sporedni mozak nazvan "Q glava". Nakon svakog potpunog procesa razmišljanja (svake srednje petlje), ova Q glava izbacuje rezultat. Ovaj rezultat je u osnovi modelovo poverenje: "Koliko sam siguran da sam ovo dobio tačno?"

Ako je rezultat poverenja dovoljno visok, spoljašnja petlja se samo zaustavlja (`break`), i model prelazi na sledeći lavirint.

Uči da dobije ovaj rezultat poverenja tačno jer je to deo treninga. Nagrađuje se ako je siguran *i* tačan, i kažnjava se ako je siguran ali pogrešan. Rad ovo naziva Adaptivno Vreme Računanja (ACT).

---

```python
# Inicijalizacija
y, z = zeros_like(x), zeros_like(x)

# Petlja dubinske supervizije (do 16 puta)
for supervision_step in range(16):
    
    # Dubinska rekurzija: zagrevanje (2 puta, bez gradijenata)
    with torch.no_grad():
        for _ in range(2):
            # Latentna rekurzija
            for _ in range(6):
                z = net(x + y + z)
            y = net(y + z)
    
    # Dubinska rekurzija: finalni (1 put, SA gradijentima)
    for _ in range(6):
        z = net(x + y + z)
    y = net(y + z)
    
    # Učenje
    y_pred = output_head(y)
    loss = cross_entropy(y_pred, y_true)
    loss.backward()
    optimizer.step()
    
    # Da li bi trebalo da se zaustavimo?
    q = Q_head(y)
    if q > 0:
        break
```

---

### Korak 4: Studije Ablacije - Šta TRM Čini Funkcionalnim?

![Kompletan Studija Ablacije](/content/tiny-recursive-model/images/complete_ablation_study.png)
*Slika: Poređenje gubitka treninga preko četiri TRM konfiguracije tokom 10 epoha na rešavanju lavirinta (30x30, teško). Osnovna linija (plava puna) koristi TRM standardni dizajn: 2-slojna mreža, H=3 (srednja petlja), L=6 (unutrašnja petlja), sa EMA. Ablacije testiraju: uklanjanje EMA (crvena isprekidana), smanjenje dubine rekurzije (zelena tačka-crta), i korišćenje veće 4-slojne mreže (magenta tačkasta).*

Da bismo razumeli šta čini TRM efikasnim, sistematski testiramo varijacije uklanjanjem ili menjanjem ključnih komponenti. Ove **studije ablacije** otkrivaju koje odluke dizajna su esencijalne.

#### Eksperimentalno Postavljanje

Testiramo četiri konfiguracije na zadatku rešavanja lavirinta (30x30 teški lavirinti, 1000 trening primera):

| Konfiguracija | Slojevi | H_ciklusi | L_ciklusi | EMA | Efektivna Dubina* |
|---------------|---------|-----------|-----------|-----|-------------------|
| **Osnovna TRM** | 2 | 3 | 6 | Da | 42 |
| **Bez EMA** | 2 | 3 | 6 | Ne | 42 |
| **Manje Rekurzije** | 2 | 2 | 2 | Da | 12 |
| **Veći Mozak** | 4 | 3 | 3 | Da | 48 |

*Efektivna dubina = T × (n+1) × slojevi

#### Rezultati

**Napomena:** Ovo su 10-epoha eksperimenti—veoma mala količina treninga u poređenju sa 50,000+ epoha izvršavanjima rada. Duže treniranje može značajno promeniti relativnu performansu ovih konfiguracija, naročito za generalizaciju (kao što vidimo sa "Veći Mozak" rezultatima ispod).

| Konfiguracija | Početni Gubitak | Finalni Gubitak | Min Gubitak | Poboljšanje |
|---------------|-----------------|-----------------|-------------|-------------|
| Osnovna | 1.789 | 1.062 | 1.045 | 40.6% |
| Bez EMA | 1.789 | 1.042 | 1.041 | 41.7% |
| Manje Rekurzije | **2.100** | 1.100 | 1.042 | 47.6% |
| Veći Mozak (4-slojni) | 1.789 | **1.007** | **1.007** | **43.7%** |

#### Ključni Nalazi

**1. Paradoks "Većeg Mozga": Kratkoročna vs. Dugoročna Performansa**

4-slojna mreža je postigla **najbolji finalni gubitak** (1.007) u našim 10-epoha eksperimentima, nadmašujući 2-slojnu osnovnu liniju za ~5%. Ovo izgleda da protivreči tvrdnji rada da "manje je više".

**Zašto Razlika?**
- **Kratkoročno** (10 epoha): Više kapaciteta = brže učenje. 4-slojna mreža može brzo memorisati obrasce.
- **Dugoročno** (50k+ epoha): Više kapaciteta = prekomerno prilagođavanje. 2-slojna mreža je *primorana* da nauči ponovne strategije rasuđivanja umesto memorisanja specifičnih rešenja.
  
Osnovna teza rada: **Male mreže primorane da razmišljaju rekurzivno generalizuju bolje od velikih mreža**, čak i ako se treniraju sporije inicijalno. 2-slojna arhitektura je izabrana specifično da spreči memorisanje i prisili oslanjanje na rekurziju.

**2. Dubina Rekurzije je Fundamentalna**

Konfiguracija "Manje Rekurzije" (H=2, L=2) pokazuje ozbiljno degradiranu performansu:
- Počela na **17% višem početnom gubitku** (2.100 vs 1.789) pre bilo kakvog treninga
- Postigla najgori finalni gubitak (1.100) uprkos poboljšanju od 47.6%

**Šta Rad Kaže:** Smanjenje rekurzije sa T=3, n=6 na T=2, n=2 smanjuje Sudoku tačnost sa 87.4% na 73.7% — pad od 14%.

**Zašto Ovo Ima Značaj:** Visok početni gubitak otkriva da plitka rekurzija sakati modelovu reprezentacionu moć *po dizajnu*. Čak i sa perfektnim treningom, nema dovoljno rekurzivnih "koraka razmišljanja" da reši kompleksne probleme. **Ne možete kompenzovati nedovoljnu dubinu rekurzije sa boljim treningom.**

**3. EMA Ima Minimalan Kratkoročan Uticaj**

Uklanjanje EMA jedva je uticalo na 10-epoha performansu (finalni gubitak 1.042 vs 1.062 za osnovnu liniju, samo ~2% razlika).

**Šta Rad Kaže:** Na Sudoku-Extreme, uklanjanje EMA smanjuje tačnost sa 87.4% na 79.9% — pad od 8% nakon potpunog treninga.

**Zašto Razlika?** EMA je **Eksponencijalni Pomični Prosek** težina modela koji stabilizuje trening tokom dugih izvršavanja. U kratkim eksperimentima, oba modela još uvek istražuju i nisu još susreli nestabilnost koju EMA sprečava. Tokom 50,000+ epoha, EMA sprečava katastrofičnu divergenciju i pikove prekomerne prilagođenosti, čineći ga esencijalnim za finalnu performansu.

---

Hvala vam što ste pročitali ovaj tutorijal i vidimo se u sledećem.

