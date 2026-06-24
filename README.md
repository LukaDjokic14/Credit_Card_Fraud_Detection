# Primena neuronskih mreža za predvidjanje prevara u finansijama (credit card fraud)

1. Opis problema

Danas skoro svi koristimo kreditne kartice za plaćanja, od kupovine namirnica do plaćanja računa. Kako se svet seli na internet, tako nažalost raste i broj prevara. Prevaranti više ne biraju gde će napasti – od malih radnji do velikih banaka, svi su na meti. Zbog toga je postalo presudno da imamo sisteme koji mogu brzo i precizno da prepoznaju kada nešto nije u redu.

Cilj ovog projekta je da napravimo sistem koji će "uhvatiti" prevaru pre nego što se transakcija uopšte odobri. Trebalo bi da omogući da u trenutku kada korisnik prisloni karticu na terminal (ili odradi online transakciju), sistem za delić sekunde mora da odluči – da li je sve u redu ili neko pokušava da ti ukrade novac (izvrši bilo kakvu nedozvoljenu radnju). Ako sistem posumnja na prevaru, transakcija se odmah blokira, a ako je sve čisto, novac prolazi.

Najveći problem ovde je što je prevara jako malo u odnosu na normalne kupovine. ako imamo 1000 transakcija, a samo par njih su prevare. Ako bi sistem bio "lenj", mogao bi samo da kaže da je sve normalno i bio bi u pravu u 99% slučajeva, ali bi propustio one ključne prevare koje nam trebaju. Tu obični algoritmi često greše, pa smo se zato okrenuli veštačkim neuronskim mrežama. Neuronske mreže mogu da nauče i najsitnije, skrivene detalje koji razlikuju običnu kupovinu od one sumnjive.

U ovom radu sam se fokusirao na to kako da naučimo neuronsku mrežu da prepozna te prevare. Pored samog pravljenja modela, mnogo sam pažnje posvetio i pripremi podataka, jer ako podaci nisu dobro sređeni, ni najbolji algoritam ne može da radi kako treba. Na kraju, cilj mi je bio da napravim model koji zaista može da razlikuje prevaru od poštene transakcije, uzimajući u obzir i rizik da se greškom blokira kartica nekome ko samo želi da plati svoj račun.

2.Podaci

Izvor i struktura

Dataset korišćen u ovom projektu preuzet je sa platforme Kaggle (dataset preuzet sa: https://www.kaggle.com/datasets/kartik2112/fraud-detection
). Originalni skup sadrži preko 1.500.000 transakcija, a za potrebe ovog projekta izdvojen je reprezentativan podskup od 200.000 transakcija. Svaka transakcija sadrži 23 obeležja koja opisuju različite aspekte plaćanja, uključujući podatke o korisniku (npr. pol, zanimanje, lokacija), detalje o trgovcu i same karakteristike transakcije (iznos, vreme).

Analiza, čišćenje i preprocesiranje

Pre obučavanja modela, podaci su prošli kroz rigorozan proces pripreme:

Selekcija i inženjering obeležja: Iz originalnog skupa izdvojene su relevantne kolone, dok su redundantne informacije uklonjene kako bi se smanjila dimenzionalnost. Posebna pažnja posvećena je kategoričkim podacima (poput kategorije trgovca ili pola korisnika) – nad njima je primenjena one-hot enkoding metoda (get_dummies), čime su kategorije pretvorene u numeričke kolone pogodne za rad neuronske mreže.

Testiranje normalnosti: Sproveden je Shapiro-Wilk test nad svim obeležjima. Rezultati su pokazali da nijedna varijabla ne prati normalnu raspodelu, odnosno da postoji određen broj autlajera (npr. u koloni amount)

Skaliranje podataka: Zbog osetljivosti na autlajere, izbegnut je StandardScaler. Primenjen je RobustScaler, koji koristi medijanu i interkvartilni opseg, čime je postignuta stabilnost modela čak i pri velikim finansijskim oscilacijama.

Podela skupa: Podaci su podeljeni na trening i test skup u odnosu 80/20. Korišćena je stratifikovana podela, čime je osigurano da procenat prevara (0.82%) bude identičan u oba skupa, što je preduslov za realnu procenu performansi modela.

3. Arhitektura modela
![Arhitektura modela](arhitektura.png)

Za potrebe binarne klasifikacije transakcija, implementirana je Feedforward neuronska mreža (Multilayer Perceptron). Model je dizajniran tako da postigne balans između sposobnosti učenja kompleksnih obrazaca i otpornosti na pretreniranost (overfitting).

Arhitektura mreže se sastoji od:

Ulazni sloj: Prima 23 ulazna obeležja koja opisuju transakciju.

Skriveni slojevi: Mreža sadrži dva skrivena sloja sa 64, odnosno 32 neurona. Izbor ReLU aktivacione funkcije omogućava mreži uvođenje nelinearnosti, što je ključno za detekciju skrivenih anomalija u podacima.

Regularizacija (Dropout): U svakom sloju primenjen je Dropout mehanizam, koji nasumično "gasi" deo neurona tokom treninga. Ovo primorava mrežu da ne postane zavisna od pojedinačnih neurona, čime se značajno smanjuje rizik od pamćenja šuma iz podataka umesto učenja pravih šablona prevara.

Izlazni sloj: Jedan neuron koji daje sirovi izlazni signal (logit), koji se kasnije pomoću Sigmoid funkcije mapira u verovatnoću pripadnosti klasi prevare.

Za obučavanje modela korišćena je BCEWithLogitsLoss funkcija greške (pogodna za numeričku stabilnost kod binarnih problema) i Adam optimizator sa stopom učenja (learning rate) od 0.001.
