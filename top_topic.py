import pandas as pd
import numpy as np
import nltk
from spacy.lang.id import Indonesian
import string
import re #regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE     
# from fitsne import FItSNE as TSNE                                                                                                                                                                                                                                                                                                                                         
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
import time
from gensim.utils import simple_preprocess
from lda import perform_lda, create_lda_inputs, perform_tsne



def calculate_word_counts(formatted_topics, input_text):
    word_counts = {}
    topic_words = [word for topic in formatted_topics for word in topic['words']]   
    for doc in input_text:
        words = doc.split()
        for word in words:
            if word in topic_words:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)  
    return sorted_word_counts










texts = ["Saya memang lama dalam membuat keputusan, tapi sekalinya membuat keputusan saya setia\" Yenny Wahid.Pak Prabowo Subianto menerima kunjungan putri Presiden ke-4 RI Abdurrahman Wahid (Gus Dur) Yenny Wahid di kediaman Kertanegara.\n\n#prabowo #indonesia #PakvsBan #gerindra #yennywahid https://t.co/3D9a6gg40R",
"@sobat_anies Berandai-andai.. sobat. Prabowo berpasangan dengan Ganjar.. Apa mungkin terjadi... Advantage pilpres hanya dua pasang pasangan capres.. @KPU_ID  Anies -Imin AMIN dan Prabowo - Pranowo WOWO .. @pemilu",
"Ganjar paling solutif terbuka\nAnies mbulet\nMata najwa\nTka china \nhttps://t.co/5VeuXsRFUi",
"MUI tak masalah Ganjar tampil di tayangan Adzan Magrib\n#GanjarPranowo #GanjarCapres #GanjarUntukSemua #GanjarPresiden #GanjarPresiden2024 #IndonesiaMaju https://t.co/rJ2u3IwWFD",
"dalam kerja-kerja kampanye damai dan satu demi memenangkan Pak Prabowo di pemilu 2024 mendatang. ungkapnya”.",
"@ganjarpranowo Pak Ganjar, Indonesia maju!",
"TPN Ganjar fokus sosialisasi Program.\nGanjar Mahfud IdolaRakyat\nPilih Pemimpin T3rbaik https://t.co/g2mvvxdhEp",
"@Gus_Raharjo tandanya pak ganjar lawan yg hebat sih yaa wkwkw, malah nebar hoax buat jatohin, tp ga mempanny",
"@KataNadiaaa Kinerja pak ganjar sebelumnya di jateng sangat bagus dan terbukti berhasil membangun jateng lebih baik Ganjar paling solutif",
"Dukungan Kuat Berdatangan Berkat Komitmen Prabowo Kedepankan Persatuan untuk Rakyat - Tribun Jabar #Jokowinomics #MenataMasaDepan #BersamaPrabowo https://t.co/CrDJQT0que",
"new neighbors, have to move again\n\nPrabowo Gibran PDIP #KamiMuak Muntilan #FreePalestine Jokowi Pagiii Projo Dipendem Sidoarjo Minggu #PalestineGenocide Senin #ArusPerubahanAMIN #StrongGirlNamSoonEp4 shawn mendes Onic Pecco Miya https://t.co/CtMq0qzQbE",
"@Yom_N_Friends Wkwk bujer2 yg dlu jelek2in azan dan anti politisasi agama mendadak goblok meliat si ganjar ini",
"Ganjar Pranowo berhasil bangun 2.369 desa mandiri energi\n#GanjarCapres https://t.co/QUJtb6fg4j",
"@AndreasSolusi @ganjarpranowo @TheLadyJoker Diskusi tentang Pak Ganjar sering kali berputar-putar di satu topik yang sama.",
"Ditanya Soal Petugas Partai atau Petugas Rakyat, Ganjar Pranowo: Saya Kader, tapi Presiden Bukan #TempoNasional https://t.co/3mKs9TO6sk",
"@BungkusTukang bilang aja acara relawan ganjar. Gak usah bilang alumni Perguruan Tinggi cuman buat nutupin  berita Ganjar di UI",
"@NarasiNewsroom Pak ganjar sosok yang luar biasa penuh motivasi,sederhana,merakyat dan juga selalu mengutamakan sejarah ,all in pak ganjar🔥",
"@Paltiwest Berarti anda berkata bahwa pak @jokowi bohong? JD hanya untuk menutupi? Wow.. pak Jokowi sengaja menutupi aib pak prabowo guys.. fantastis😀😀🫢. Gimana nih min @Gerindra",
"@henrysubiakto Sangat pincang. Terbalik. Seharusnya Mahfud yang Capres. Sedang Ganjar jadi Cawapresnya.",
"@Naz_lira Tinggal sebut nama Prabowo dan Ganjar aja kok repot sih",
"Optimis Ganjar Pranowo Presiden 2024 https://t.co/vPRZyr8WzI",
"@tvOneNews Nggak setuju blas,,,,\nPrabowo akan jadi benelu,,,,\nJadi mantri aja nepotisme nya besar \nNggak ada prestasi juga,,,,,\nNggak lah sekali lagi",
"@Gus_Raharjo @ganjarpranowo Tetep pilihan ku pak Ganjar siapapun cawapres beliau",
"@Melihat_Indo @ganjarpranowo hanya Pak Ganjar pemimpin yang sangat merakyat\n#TuankuRakyat",
"@Prihati_utami kita dukung sampai kapan pun ya Pak Ganjar\n#TuankuRakyat",
"Ganjar pranowo mendorong terobosan terbaru\n\n#GanjarPemimpinMengayomi https://t.co/8SFEEAdJSc",
"@VIVAcoid Ganjar meraih elektabilitas tertinggi dengan 34,1 persen. Kemudian disusul dengan Prabowo yang memiliki 31,3 persen, dan Anies dengan 19,2 persen.\n\nGanjar Menang",
"@kiki_daliyo Ganjar Mahfud dipilih berdasarkan pengalaman mereka, tanpa drama apapun #GanjarMahfudTanpaDrama https://t.co/85ObpZr9my",
"Jual Prolq Produk Dr.Boyke Untuk Stamina Pria Dewasa\n⬇️Beli Disini⬇️\nhttps://t.co/gOfnwzBKjM\n\nMalam Jumat Aldi Taher Batik Air Miring Narnia Kapolsek Jenglot Harry Potter MC SOHEE Rempang tannie iPhone 15 Pak Ganjar #viral #viralindo #viralvideo #viral https://t.co/8WqdLhZUKn",
"@Fahrihamzah Ayoe Move on dari Prabowo... Seperti Image @Fahrihamzah Ayo Move On 2024#",
"Hadirrr siap kawal pak ganjar sampe jadi presiden💪🏼💪🏼💪🏼\n@nyinyirsedunia\nhttps://t.co/rOeT5pErwY",
"Warga NU masih tetap tegak lurus  sbg benteng NKRI, tdk ada kompromi  apalagi berkolaborasi dgn  gerombolan kadrun.\nGanjar No kadrun.👍",
"@prabowoisme Mantab, Pak Prabowo menaikkan keyakinan rakyat dengan selalu mengutamakan kerja berkualitas untuk kepentingan rakyat dan NKRI.\n\nPakbowoKamu GakSendirian\nSatukanLangkah SatuPaksubianto",
"Lanjut saja, Pak Prabowo menang\n@hariqosatria https://t.co/CvFJuNJv0U"
]

dictionary, doc_term_matrix = create_lda_inputs(texts)

# # # Perform LDA topic modeling
lda_model, formatted_topics = perform_lda(doc_term_matrix, total_topics=5, dictionary=dictionary, number_words=3)
word_counts = calculate_word_counts(formatted_topics,texts)

# Print the word counts
print("Word\t\tWord Count")
for word, count in word_counts:
    print(f"{word.ljust(10)}\t{count}")

