import random
import string

# ---------- Kumpulan data kosakata Bahasa Sula beserta artinya ----------
kamus = [
    ("GAYA", "MAKAN"),
    ("WIN", "MINUM"),
    ("NIB", "DUDUK"),
    ("GEHE", "BERDIRI"),
    ("NONA", "TIDUR"),
    ("BOYA", "LAPAR"),
    ("FAFOI", "PAKAIAN"),
    ("HAMA", "MATA"),
    ("NANA", "KECIL"),
    ("AYA", "BESAR"),
    ("LIMA", "TANGAN"),
    ("NIHI", "GIGI"),
    ("BABA", "AYAH"),
    ("BIRA", "BERAS"),
    ("KENA", "IKAN"),
]

ALPHABET = string.ascii_uppercase

# ---------- Wadah penyimpanan variabel selama proses GA berjalan ----------
ga_state = {
    "sudah_jalan": False,
    "target": None,
    "seed": None,
    "populasi": [],          # daftar kromosom pada populasi awal
    "keterangan": [],
    "fitness_benar": [],     # banyaknya huruf yang cocok per individu
    "fitness_nilai": [],     # nilai fitness setelah dibagi panjang kromosom
    "total_fitness": 0.0,
    "rata_lama": 0.0,
    "probabilitas": [],
    "interval": [],          # pasangan (batas_bawah, batas_atas) tiap individu
    "r1": 0.0, "r2": 0.0,
    "idx_induk1": 0, "idx_induk2": 0,
    "induk1": "", "induk2": "",
    "titik_crossover": 0,
    "anak1_cross": "", "anak2_cross": "",
    "idx_anak_mutasi": 0,
    "posisi_mutasi": 0,
    "huruf_lama": "", "huruf_baru": "",
    "anak1_final": "", "anak2_final": "",
    "fitness_anak1_sebelum": 0.0, "fitness_anak1_sesudah": 0.0,
    "fitness_anak2": 0.0,
    "idx_diganti": [],
    "populasi_baru": [],
    "fitness_baru_nilai": [],
    "total_baru": 0.0,
    "rata_baru": 0.0,
    "individu_terbaik": "",
}


# ---------- Kumpulan fungsi untuk mengelola kamus ----------

def tampilkan_kamus():
    print("\n>> DAFTAR KOSAKATA BAHASA SULA <<")
    print(f"{'No':<4}{'Bahasa Sula':<15}{'Arti (Bahasa Indonesia)':<25}")
    print("-" * 44)
    for i, (kata, arti) in enumerate(kamus, 1):
        print(f"{i:<4}{kata:<15}{arti:<25}")


def cari_kata():
    kata_cari = input("Masukkan kata yang ingin dicari: ").strip().upper()
    for kata, arti in kamus:
        if kata_cari == kata.upper():
            print(f"\n'{kata}' (Bahasa Sula) berarti '{arti}' dalam Bahasa Indonesia")
            return
        if kata_cari == arti.upper():
            print(f"\n'{arti}' (Bahasa Indonesia) padanannya adalah '{kata}' (Bahasa Sula)")
            return
    print(f"\nMaaf, kata '{kata_cari}' tidak terdapat pada kamus.")


# ---------- Kumpulan fungsi pendukung proses Algoritma Genetika ----------

def huruf_lain(huruf_asal):
    """Membangkitkan satu huruf acak dari alfabet selain huruf_asal."""
    return random.choice([c for c in ALPHABET if c != huruf_asal])


def hitung_huruf_benar(individu, target):
    return sum(1 for a, b in zip(individu, target) if a == b)


def buat_populasi_awal(target):
    """
    Menyusun populasi awal dengan pola yang sudah ditentukan (bukan sepenuhnya
    acak):
    - Setiap individu I1 sampai IL dibuat berbeda satu huruf saja dari kata
      target, secara berurutan mulai posisi pertama hingga posisi terakhir.
    - Satu individu tambahan dibangkitkan sepenuhnya acak sehingga tidak ada
      satu huruf pun yang sama dengan target.
    """
    L = len(target)
    populasi = []
    keterangan = []

    urutan_posisi = ["ke-1", "ke-2", "ke-3", "ke-4", "ke-5",
                      "ke-6", "ke-7", "ke-8"]

    for pos in range(L):
        chars = list(target)
        chars[pos] = huruf_lain(chars[pos])
        populasi.append(''.join(chars))
        nama_posisi = urutan_posisi[pos] if pos < len(urutan_posisi) else f"ke-{pos+1}"
        keterangan.append(f"Berbeda pada huruf {nama_posisi}")

    acak = ''.join(huruf_lain(c) for c in target)
    populasi.append(acak)
    keterangan.append("Dibangkitkan acak, semua huruf tidak ada yang cocok")

    return populasi, keterangan


def cari_interval(nilai_r, interval):
    """Menentukan indeks individu berdasarkan letak nilai_r pada interval kumulatifnya."""
    for i, (bawah, atas) in enumerate(interval):
        if atas > bawah and bawah <= nilai_r < atas:
            return i
        if atas > bawah and abs(atas - 1.0) < 1e-9 and nilai_r == 1.0:
            return i
    # jika tidak ada interval yang cocok, pilih individu berfitness paling tinggi
    return max(range(len(interval)), key=lambda i: interval[i][1] - interval[i][0])


# ---------- Alur utama proses Algoritma Genetika ----------

def jalankan_ga():
    print("\n>> MENJALANKAN ALGORITMA GENETIKA <<")
    tampilkan_kamus()
    try:
        pilihan = int(input("\nPilih No kata (Bahasa Sula) sebagai target pencarian: "))
        target = kamus[pilihan - 1][0].upper()
    except (ValueError, IndexError):
        print("Pilihan tidak valid.")
        return

    seed_input = input("Masukkan random seed (kosongkan untuk pakai default = 7): ").strip()
    seed = int(seed_input) if seed_input else 7
    random.seed(seed)

    L = len(target)
    ga_state["target"] = target
    ga_state["seed"] = seed

    # Tahap 1: menyusun representasi individu dan populasi awal
    populasi, keterangan = buat_populasi_awal(target)
    ga_state["populasi"] = populasi
    ga_state["keterangan"] = keterangan

    print(f"\nTarget pencarian     : {target}")
    print(f"Jumlah gen/kromosom  : {L}")
    print(f"Seed acak yang dipakai : {seed} (supaya hasil bisa diulang persis sama)")

    print(f"\n[Tahap 1] Individu dan Populasi Awal")
    print(f"{'Individu':<10}{'Kromosom':<12}{'Keterangan':<35}")
    for i, (ind, ket) in enumerate(zip(populasi, keterangan), 1):
        print(f"I{i:<9}{ind:<12}{ket:<35}")

    # Tahap 2: menghitung nilai fitness tiap individu
    fitness_benar = [hitung_huruf_benar(ind, target) for ind in populasi]
    fitness_nilai = [round(b / L, 2) for b in fitness_benar]
    total_fitness = round(sum(fitness_nilai), 2)
    rata_lama = round(total_fitness / len(populasi), 2)

    ga_state["fitness_benar"] = fitness_benar
    ga_state["fitness_nilai"] = fitness_nilai
    ga_state["total_fitness"] = total_fitness
    ga_state["rata_lama"] = rata_lama

    print(f"\n[Tahap 2] Perhitungan Fitness")
    print("Fitness dihitung dari: jumlah huruf yang tepat dibagi panjang kata\n")
    print(f"{'Individu':<10}{'Kromosom':<12}{'Huruf Benar':<14}{'Fitness':<10}")
    for i, (ind, b, f) in enumerate(zip(populasi, fitness_benar, fitness_nilai), 1):
        print(f"I{i:<9}{ind:<12}{b:<14}{f:<10}")
    print(f"\nJumlah seluruh fitness populasi = {total_fitness}")
    print(f"Fitness rata-rata populasi awal = {total_fitness}/{len(populasi)} = {rata_lama}")

    # Tahap 3: seleksi induk memakai metode roulette wheel
    if total_fitness > 0:
        probabilitas = [round(f / total_fitness, 4) for f in fitness_nilai]
    else:
        probabilitas = [round(1 / len(populasi), 4) for _ in populasi]

    interval = []
    batas_bawah = 0.0
    for p in probabilitas:
        batas_atas = round(batas_bawah + p, 4)
        interval.append((batas_bawah, batas_atas))
        batas_bawah = batas_atas
    ga_state["probabilitas"] = probabilitas
    ga_state["interval"] = interval

    print(f"\n[Tahap 3] Seleksi Roulette Wheel")
    print("Probabilitas tiap individu = fitness individu dibagi total fitness\n")
    print(f"{'Individu':<10}{'Fitness':<10}{'Probabilitas':<14}{'Interval Kumulatif':<20}")
    for i, (f, p, (a, b)) in enumerate(zip(fitness_nilai, probabilitas, interval), 1):
        teks_interval = f"{a:.2f} - {b:.2f}" if b > a else "-"
        print(f"I{i:<9}{f:<10}{p:<14}{teks_interval:<20}")

    r1 = round(random.random(), 4)
    idx1 = cari_interval(r1, interval)
    r2 = round(random.random(), 4)
    idx2 = cari_interval(r2, interval)
    induk1 = populasi[idx1]
    induk2 = populasi[idx2]

    ga_state.update({"r1": r1, "r2": r2, "idx_induk1": idx1, "idx_induk2": idx2,
                      "induk1": induk1, "induk2": induk2})

    a1, b1 = interval[idx1]
    a2, b2 = interval[idx2]
    print(f"\nInduk dipilih memakai bilangan acak (seed = {seed}):")
    print(f"  r1 = {r1} -> masuk interval I{idx1+1} ({a1:.2f}-{b1:.2f}), jadi Induk 1 = I{idx1+1} ({induk1})")
    print(f"  r2 = {r2} -> masuk interval I{idx2+1} ({a2:.2f}-{b2:.2f}), jadi Induk 2 = I{idx2+1} ({induk2})")

    # Tahap 4: pindah silang (crossover) satu titik
    if L > 1:
        titik = random.randint(1, L - 1)
    else:
        titik = 0
    anak1 = induk1[:titik] + induk2[titik:]
    anak2 = induk2[:titik] + induk1[titik:]
    ga_state.update({"titik_crossover": titik, "anak1_cross": anak1, "anak2_cross": anak2})

    print(f"\n[Tahap 4] Pindah Silang (Crossover)")
    print("Teknik yang dipakai: Single-Point Crossover")
    print(f"Posisi titik potong (acak) = {titik}\n")
    print(f"Induk 1 ({induk1}) : {induk1[:titik]} | {induk1[titik:]}")
    print(f"Induk 2 ({induk2}) : {induk2[:titik]} | {induk2[titik:]}")
    print(f"\nAnak 1 = '{induk1[:titik]}' + '{induk2[titik:]}' = {anak1}")
    print(f"Anak 2 = '{induk2[:titik]}' + '{induk1[titik:]}' = {anak2}")
    if anak1 == induk1 and anak2 == induk2:
        print("\nCatatan: kebetulan segmen gen yang ditukar bernilai sama,")
        print("sehingga anak yang terbentuk identik dengan induknya. Hal ini wajar terjadi.")

    # Tahap 5: mutasi gen
    anak_list = [anak1, anak2]
    idx_anak_mutasi = random.randint(0, 1)
    posisi_mutasi = random.randint(0, L - 1)
    huruf_lama = anak_list[idx_anak_mutasi][posisi_mutasi]
    huruf_baru = huruf_lain(huruf_lama)

    fitness_sebelum = round(hitung_huruf_benar(anak_list[idx_anak_mutasi], target) / L, 2)

    anak_chars = list(anak_list[idx_anak_mutasi])
    anak_chars[posisi_mutasi] = huruf_baru
    anak_list[idx_anak_mutasi] = ''.join(anak_chars)

    fitness_sesudah = round(hitung_huruf_benar(anak_list[idx_anak_mutasi], target) / L, 2)
    anak1_final, anak2_final = anak_list
    fitness_anak_lain = round(hitung_huruf_benar(anak_list[1 - idx_anak_mutasi], target) / L, 2)

    ga_state.update({
        "idx_anak_mutasi": idx_anak_mutasi, "posisi_mutasi": posisi_mutasi,
        "huruf_lama": huruf_lama, "huruf_baru": huruf_baru,
        "anak1_final": anak1_final, "anak2_final": anak2_final,
        "fitness_anak1_sebelum": fitness_sebelum, "fitness_anak1_sesudah": fitness_sesudah,
        "fitness_anak2": fitness_anak_lain,
    })

    nama_anak = f"Anak {idx_anak_mutasi + 1}"
    nama_anak_lain = f"Anak {2 - idx_anak_mutasi}"
    print(f"\n[Tahap 5] Mutasi Gen")
    print(f"Anak yang terkena mutasi : {nama_anak} ({anak2 if idx_anak_mutasi == 1 else anak1})")
    print(f"Gen yang berubah         : gen ke-{posisi_mutasi + 1}")
    print(f"Huruf semula             : '{huruf_lama}'")
    print(f"Huruf pengganti (acak)   : '{huruf_baru}'")
    print(f"Kromosom setelah mutasi  : {anak_list[idx_anak_mutasi]}")
    print(f"\nFitness {nama_anak} sebelum dimutasi = {fitness_sebelum}")
    print(f"Fitness {nama_anak} sesudah dimutasi = {fitness_sesudah}")
    print(f"{nama_anak_lain} tidak ikut bermutasi, kromosomnya tetap {anak_list[1 - idx_anak_mutasi]} "
          f"dengan fitness {fitness_anak_lain}")

    # Tahap 6: pembentukan populasi generasi berikutnya
    pasangan_fitness = list(enumerate(fitness_nilai))
    pasangan_urut = sorted(pasangan_fitness, key=lambda x: x[1])
    idx_diganti = [pasangan_urut[0][0], pasangan_urut[1][0]]

    populasi_baru = populasi.copy()
    populasi_baru[idx_diganti[0]] = anak1_final
    populasi_baru[idx_diganti[1]] = anak2_final

    fitness_baru_nilai = [round(hitung_huruf_benar(ind, target) / L, 2) for ind in populasi_baru]
    total_baru = round(sum(fitness_baru_nilai), 2)
    rata_baru = round(total_baru / len(populasi_baru), 2)
    idx_terbaik = fitness_baru_nilai.index(max(fitness_baru_nilai))
    individu_terbaik = f"I{idx_terbaik+1} ({populasi_baru[idx_terbaik]})"

    ga_state.update({
        "idx_diganti": idx_diganti, "populasi_baru": populasi_baru,
        "fitness_baru_nilai": fitness_baru_nilai, "total_baru": total_baru,
        "rata_baru": rata_baru, "individu_terbaik": individu_terbaik,
        "sudah_jalan": True,
    })

    print(f"\n[Tahap 6] Populasi Generasi ke-1 (Baru)")
    print(f"Dua individu dengan fitness paling rendah pada populasi lama diganti")
    print(f"(I{idx_diganti[0]+1} dan I{idx_diganti[1]+1}) oleh Anak 1 dan Anak 2 hasil crossover+mutasi.\n")
    print(f"{'Individu':<10}{'Kromosom':<12}{'Fitness':<10}")
    for i, (ind, f) in enumerate(zip(populasi_baru, fitness_baru_nilai), 1):
        print(f"I{i:<9}{ind:<12}{f:<10}")
    print(f"\nJumlah fitness populasi baru = {total_baru}")
    print(f"Fitness rata-rata populasi baru = {total_baru}/{len(populasi_baru)} = {rata_baru} "
          f"({'meningkat' if rata_baru >= rata_lama else 'menurun'} dibanding rata-rata populasi awal {total_fitness}/{len(populasi)} = {rata_lama})")
    print(f"\nIndividu paling unggul pada Generasi ke-1: {individu_terbaik}")
    if populasi_baru[idx_terbaik] == target:
        print(f">> Selamat, kata target '{target}' berhasil ditemukan pada generasi ke-1!")
    else:
        print(f">> Kata target '{target}' masih belum ditemukan secara tepat pada generasi ini;")
        print(f">> proses seleksi-crossover-mutasi masih perlu diulang pada generasi selanjutnya")
        print(f">> sampai ditemukan individu dengan fitness = 1.00.")

    print("\nProses 1 generasi Algoritma Genetika telah selesai dijalankan.")
    print("Silakan pilih menu 4 sampai 9 untuk melihat kembali rincian tiap tahapnya.")


# ==================== MENU 4-9: TAMPILKAN ULANG TIAP TAHAP ====================

def cek_sudah_jalan():
    if not ga_state["sudah_jalan"]:
        print("\nJalankan dahulu Algoritma Genetika melalui menu 3 sebelum melihat tahap ini.")
        return False
    return True


def tampilkan_populasi():
    if not cek_sudah_jalan():
        return
    print(f"\n>> POPULASI AWAL (Target: {ga_state['target']}) <<")
    print(f"{'Individu':<10}{'Kromosom':<12}{'Keterangan':<35}")
    for i, (ind, ket) in enumerate(zip(ga_state["populasi"], ga_state["keterangan"]), 1):
        print(f"I{i:<9}{ind:<12}{ket:<35}")


def tampilkan_fitness():
    if not cek_sudah_jalan():
        return
    print(f"\n>> NILAI FITNESS (Target: {ga_state['target']}) <<")
    print(f"{'Individu':<10}{'Kromosom':<12}{'Huruf Benar':<14}{'Fitness':<10}")
    for i, (ind, b, f) in enumerate(zip(ga_state["populasi"], ga_state["fitness_benar"], ga_state["fitness_nilai"]), 1):
        print(f"I{i:<9}{ind:<12}{b:<14}{f:<10}")
    print(f"\nTotal seluruh fitness = {ga_state['total_fitness']}")
    print(f"Fitness rata-rata     = {ga_state['rata_lama']}")


def tampilkan_roulette():
    if not cek_sudah_jalan():
        return
    print(f"\n>> SELEKSI ROULETTE WHEEL (Target: {ga_state['target']}) <<")
    print(f"{'Individu':<10}{'Fitness':<10}{'Probabilitas':<14}{'Interval Kumulatif':<20}")
    for i, (f, p, (a, b)) in enumerate(zip(ga_state["fitness_nilai"], ga_state["probabilitas"], ga_state["interval"]), 1):
        teks_interval = f"{a:.2f} - {b:.2f}" if b > a else "-"
        print(f"I{i:<9}{f:<10}{p:<14}{teks_interval:<20}")
    print(f"\nr1 = {ga_state['r1']}  ==> Induk 1 = I{ga_state['idx_induk1']+1} ({ga_state['induk1']})")
    print(f"r2 = {ga_state['r2']}  ==> Induk 2 = I{ga_state['idx_induk2']+1} ({ga_state['induk2']})")


def tampilkan_crossover():
    if not cek_sudah_jalan():
        return
    titik = ga_state["titik_crossover"]
    induk1, induk2 = ga_state["induk1"], ga_state["induk2"]
    print(f"\n>> PINDAH SILANG / CROSSOVER (titik potong posisi {titik}) <<")
    print(f"Induk 1 ({induk1}) : {induk1[:titik]} | {induk1[titik:]}")
    print(f"Induk 2 ({induk2}) : {induk2[:titik]} | {induk2[titik:]}")
    print(f"\nAnak 1 = {ga_state['anak1_cross']}")
    print(f"Anak 2 = {ga_state['anak2_cross']}")


def tampilkan_mutasi():
    if not cek_sudah_jalan():
        return
    idx = ga_state["idx_anak_mutasi"]
    nama_anak = f"Anak {idx + 1}"
    nama_anak_lain = f"Anak {2 - idx}"
    print(f"\n>> MUTASI GEN <<")
    print(f"Anak yang termutasi     : {nama_anak}")
    print(f"Perubahan pada gen ke-{ga_state['posisi_mutasi']+1} : '{ga_state['huruf_lama']}' menjadi '{ga_state['huruf_baru']}'")
    hasil = ga_state["anak1_final"] if idx == 0 else ga_state["anak2_final"]
    lain = ga_state["anak2_final"] if idx == 0 else ga_state["anak1_final"]
    print(f"Kromosom hasil mutasi   : {hasil}")
    print(f"Fitness sebelum dan sesudah : {ga_state['fitness_anak1_sebelum']} -> {ga_state['fitness_anak1_sesudah']}")
    print(f"\n{nama_anak_lain} tetap seperti semula (tanpa mutasi) : {lain} (fitness = {ga_state['fitness_anak2']})")


def tampilkan_generasi_baru():
    if not cek_sudah_jalan():
        return
    idx_diganti = ga_state["idx_diganti"]
    print(f"\n>> POPULASI GENERASI KE-1 (BARU) <<")
    print(f"I{idx_diganti[0]+1} dan I{idx_diganti[1]+1} (fitness paling rendah) "
          f"digantikan oleh Anak 1 & Anak 2 hasil crossover+mutasi.\n")
    print(f"{'Individu':<10}{'Kromosom':<12}{'Fitness':<10}")
    for i, (ind, f) in enumerate(zip(ga_state["populasi_baru"], ga_state["fitness_baru_nilai"]), 1):
        print(f"I{i:<9}{ind:<12}{f:<10}")
    print(f"\nTotal fitness populasi baru = {ga_state['total_baru']}  (rata-rata = {ga_state['rata_baru']})")
    print(f"Individu paling unggul      = {ga_state['individu_terbaik']}")


# ==================== TAMPILAN MENU UTAMA ====================

def main():
    while True:
        print("\n=== Kamus Bahasa Daerah ===")
        print("1. Tampilkan Kamus")
        print("2. Cari Kata")
        print("3. Jalankan Algoritma Genetika")
        print("4. Tampilkan Populasi")
        print("5. Hasil Fitness")
        print("6. Seleksi Roulette")
        print("7. Cross Over")
        print("8. Mutasi")
        print("9. Generasi Baru")
        print("10. Keluar")
        pilihan = input("Pilih menu (1-10): ").strip()

        if pilihan == "1":
            tampilkan_kamus()
        elif pilihan == "2":
            cari_kata()
        elif pilihan == "3":
            jalankan_ga()
        elif pilihan == "4":
            tampilkan_populasi()
        elif pilihan == "5":
            tampilkan_fitness()
        elif pilihan == "6":
            tampilkan_roulette()
        elif pilihan == "7":
            tampilkan_crossover()
        elif pilihan == "8":
            tampilkan_mutasi()
        elif pilihan == "9":
            tampilkan_generasi_baru()
        elif pilihan == "10":
            print("\nSampai jumpa, program telah selesai dijalankan.")
            break
        else:
            print("\nPilihan tidak dikenali, silakan ulangi.")


if __name__ == "__main__":
    main()
