
**v14 notasyonu (aynen koruyorum):** p satış fiyatı, c birim maliyet, s salvage, d return toplama maliyeti, g shortage cezası, r return olasılığı, k resalability, G brüt talep, N net talep, Q sipariş, ve türetilmiş pG, pN, gN. Maliyet yapısı da v14'teki gibi:

$$
C(Q) = (c-s),\mathbb{E}[(Q-N)^+] + (p_N - c + g_N),\mathbb{E}[(N-Q)^+]
$$

**Eklenen tek sembol: φ (recapture kesri).** Tanım:

$$
\varphi = 1 - \frac{\mathbb{E}[N]}{\mathbb{E}[G]}, \qquad \varphi \in [0,, rk]
$$

Yani φ, brüt talebin recapture edilen (zamanında gelen resalable return'le karşılanan) kesri. Bir tek bunu ekliyorum; gerisi v14.

---

**Değişen formül:**

$$
p_G(\varphi) = (1-r),p + \varphi,(p-d) + (r-\varphi),(s-d)
$$

$$
p_N(\varphi) = \frac{p_G(\varphi)}{1-\varphi}, \qquad g_N(\varphi) = \frac{g}{1-\varphi}
$$

v14'te bu üçü rk sabitiyle yazılıydı; tek değişiklik  **rk → φ** .

---

**Kanıt — Bölüm A: pG(φ) nereden geliyor (asıl anlamadığın kısım bu).**

Bir birim brüt talebin başına ne geldiğini para cinsinden topluyoruz. Bir satılan birim için olasılık ağacı:

1. **İade edilmez** — olasılık (1−r). Müşteri p öder, biter. Katkı:  **p** .
2. **İade edilir** — olasılık r. İçinde iki dal:
   a. **Resalable değil** — olasılık r(1−k). Satıldı (+p), iade edildi (−p, birbirini götürür), toplama maliyeti (−d), salvage'a gider (+s). Katkı:  **(s−d)** .
   b. **Resalable** — olasılık rk. Satıldı (+p), iade (−p, götürür), toplama (−d). Sonra  **zamanlamaya göre ikiye ayrılır** :
   * **Zamanında gelir, yeniden satılır:** p geliri. Katkı  **(p−d)** .
   * **Geç gelir, satılamaz, salvage olur:** s geliri. Katkı  **(s−d)** .

İşte 2b'deki ayrım v14'te yoktu — v14 tüm resalable'ları (p−d) sayıyordu. Şimdi rk resalable'ın φ kadarı zamanında (p−d), kalan (rk−φ) kadarı geç → salvage (s−d). Toplarsak:

$$
p_G(\varphi) = \underbrace{(1-r)p} *{\text{iade yok}} + \underbrace{\varphi(p-d)}* {\text{zamanında resale}} + \underbrace{(rk-\varphi)(s-d)} *{\text{geç resalable → salvage}} + \underbrace{r(1-k)(s-d)}* {\text{resalable değil → salvage}}
$$

Son iki salvage terimini birleştir:

$$
(rk-\varphi)(s-d) + r(1-k)(s-d) = (rk - \varphi + r - rk)(s-d) = (r-\varphi)(s-d)
$$

Sonuç:

$$
\boxed{p_G(\varphi) = (1-r)p + \varphi(p-d) + (r-\varphi)(s-d)}
$$

Senin late-resalable salvage terimin açıkça **(rk−φ)(s−d)** — yani recapture edilemeyen resalable return'ler, resale yerine salvage'la değerleniyor. v14'te bu terim sıfırdı (çünkü φ=rk varsayılıyordu).

---

**Kanıt — Bölüm B: v14 özel hal olarak çıkıyor.** φ=rk koy (tüm resalable'lar zamanında = S1, full recapture):

$$
p_G(rk) = (1-r)p + rk(p-d) + (r-rk)(s-d) = (1-r)p + rk(p-d) + r(1-k)(s-d)
$$

Bu birebir v14'ün pG formülü. Yani **v14 = φ=rk özel hali.** Yeni formül v14'ü genelleştiriyor, çelişmiyor.

---

**Kanıt — Bölüm C: pN, gN ölçeği.** φ'nin tanımı gereği E[N] = (1−φ)E[G]. pG, *brüt talep birimi* başına gelir; ama newsvendor *net talep* N üzerinde çözülüyor. Toplam beklenen gelir iki şekilde de aynı olmalı:

$$
p_G \cdot \mathbb{E}[G] = p_N \cdot \mathbb{E}[N] = p_N \cdot (1-\varphi),\mathbb{E}[G] ;\Rightarrow; p_N = \frac{p_G}{1-\varphi}
$$

Shortage cezası g de aynı brüt→net köprüsüyle ölçekleniyor: gN = g/(1−φ). Bu, v14'ün yaptığı dönüşümün rk→φ genellemesi; yeni bir varsayım değil, aynı kurgu.

**Sınırlar:** φ ∈ [0, rk]. φ=rk → S1 (v14). φ=0 → S3 (hiç recapture; pG=(1−r)p+r(s−d), pN=pG, gN=g — tüm return'ler salvage). φ ∈ (0,rk) → S4 (E[N]/E[G]'den okunur).

---

**Şimdi senin asıl sorun: ayrı .py şart mı? — Hayır.**

İki şeyi karıştırma:

* **Analitik formül (yukarıdaki):** Bu, tek parametrelik bir değişiklik. `parameter_estimation.py`'deki `compute_cost_params`'a `phi` argümanı ekleyip `denom = 1 - phi` ve `pG = (1-r)*p + phi*(p-d) + (r-phi)*(s-d)` yazman yeterli (φ verilmezse `phi=r*k`'ya düşer, v14 davranışı korunur). **Yeni dosya gerekmez.** Tez formülü de bu, kanıtı da bu.
* **Doğrudan simülasyon (`cost_simulation.py`):** Bu ayrı dosya **sadece bir doğrulama aracı** — analitik formülün doğru Q'yu verdiğini olay-düzeyinde teyit etmek için. Geçen tur koştuğumuzda Q_real ≈ Q_analitik çıkmıştı; yani analitik formüle güveniyorsan bu dosyaya **mecbur değilsin.** İstersen robustness kontrolü olarak tutarsın (tezde "indirgemeyi doğrudan simülasyonla doğruladık" cümlesi için), istemezsen hiç kullanmazsın.

Yani tez için ihtiyacın olan: yukarıdaki formül + kanıt (metin), ve `compute_cost_params`'ta tek parametrelik düzenleme (kod). cost_simulation.py opsiyonel.

Tek karar sende: φ'yi her kural için simülasyondan mı okuyacaksın (E[N]/E[G]) yoksa S1/S3/S4'e analitik değerlerini mi (rk / 0 / aradaki) atayacaksın? İlki daha veri-temelli, ikincisi daha kapalı-form. Hangisini seçersen `compute_cost_params`'a giden φ'nin kaynağı değişir, formül aynı kalır. Hangisine meyilli olduğunu söyle, `compute_cost_params` düzenlemesinin tam diff'ini ona göre yazayım.
