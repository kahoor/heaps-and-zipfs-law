import collections
import math
import os
import random
import pylab

import arabic_reshaper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bidi.algorithm import get_display
from sklearn.linear_model import LinearRegression


def generate_heap_table(text):
    
    """
    Create a list of dictionaries containing words, their frequencies and
    other Zipfian data.
    """

    text = _remove_punctuation(text)

    text = text.lower()

    # With no argument, split() separates the string
    # by 1 or more consecutive instances of whitespace.
    words = text.split()

    words_num, distinct_words_num = distinct_all(words)

    return words_num, distinct_words_num, 


def _remove_punctuation(text):
    
    """
    Removes the characters:
    !\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789
    from the text.
    """

    chars_to_remove = "!❊#$٪^&*)(ـ+\()*&^٪$#❊!؟<>::؛؛۱۲۳۴۵۶۷۸۹۰[]÷ًٌٍَُِّ؛:؟<>/|''''''»«!#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789"

    tr = str.maketrans("", "", chars_to_remove)

    return text.translate(tr)

def distinct_all(words):
    
    """
    Create a list of tuples containing the most
    frequent words and their frequencies
    in descending order.
    """

    
    # random.shuffle(words)
    words_num = list(map(int, np.linspace(1, len(words), 200)))
    distinct_words_num = [len(set(words[:wn])) for wn in words_num]

    return words_num, distinct_words_num

def M(k, T, b):
    return k*(T)**b

def linear_fit(words_num, distinct_words_num):
    words_num_log = [math.log10(x) for x in words_num]
    distinct_words_num_log = [math.log10(x) for x in distinct_words_num]
    
    plt.plot(words_num_log, distinct_words_num_log)
    
    
    # fitting linear estimator
    X = np.array(words_num_log).reshape(-1, 1)
    y = np.array(distinct_words_num_log).reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    coef = reg.coef_[0][0]
    intercept = reg.intercept_[0]

    return coef, intercept
    
def heap_plot(words_num, distinct_words_num):
    
    # coef, intercept = linear_fit(words_num, distinct_words_num)
    
    words_num_log = [math.log10(x) for x in words_num]
    distinct_words_num_log = [math.log10(x) for x in distinct_words_num]
    
    plt.plot(words_num_log, distinct_words_num_log)
    
    
    # fitting linear estimator
    X = np.array(words_num_log).reshape(-1, 1)
    y = np.array(distinct_words_num_log).reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    coef = reg.coef_[0][0]
    intercept = reg.intercept_[0]

    
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + coef * x_vals
    plt.plot(x_vals, y_vals, '--', label='estimate')
    plt.xlabel('log10T')
    plt.ylabel('log10M')
    plt.title("heaps law")
    plt.grid(True)
    plt.show()
    


def nolog_heap_plot(words_num, distinct_words_num):
    coef, intercept = linear_fit(words_num, distinct_words_num)
    
    k = 10**intercept
    b = coef
    print(f"Heaps' parameters: k={k:.2f}, b={b:.2f}")

    estimate = M(k, words_num[-1], b)

    plt.plot(words_num, distinct_words_num, label='observed')
    plt.plot(words_num, [M(k, T, b) for T in words_num], label='predicted')
    plt.legend(loc="upper left")
    plt.xlabel('number of words')
    plt.ylabel('number of distinct words')
    plt.grid(True)
    plt.title(f"heap plot\n Heap's parameters: k={k:.2f}, b={b:.2f}")
    plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(True)
    no_verbose = plt.xticks()
    plt.show()

###############################################################################################################################################################################################

def generate_zipf_table(text):
    
    """
    Create a list of dictionaries containing words, their frequencies and
    other Zipfian data.
    """

    text = _remove_punctuation(text)

    text = text.lower()

    word_frequencies = _word_frequencies(text)

    zipf_table = _create_zipf_table(word_frequencies)

    return zipf_table


def _remove_punctuation(text):
    
    """
    Removes the characters:
    !\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789
    from the text.
    """

    chars_to_remove = "!❊#$٪^&*)(ـ++ـ()*&^٪$#❊!؟<>::؛؛۱۲۳۴۵۶۷۸۹۰[]÷ًٌٍَُِّ؛:؟<>/|''''''»«!#$%&'()*+,-./:;<=>?@[\]^_`{|}~0123456789"

    tr = str.maketrans("", "", chars_to_remove)

    return text.translate(tr)


def _word_frequencies(text):
    
    """
    Create a list of tuples containing the most
    frequent words and their frequencies
    in descending order.
    """

    # With no argument, split() separates the string
    # by 1 or more consecutive instances of whitespace.
    words = text.split()

    # Create a collections.Counter instance from an
    # iterable, in this case our list of words.
    word_frequencies = collections.Counter(words)

    top_word_frequencies = word_frequencies.most_common()

    return top_word_frequencies


def _create_zipf_table(frequencies):
    
    """
    Takes the list created by _top_word_frequencies
    and inserts it into a list of dictionaries,
    along with the Zipfian data.
    """

    zipf_table = []

    top_frequency = frequencies[0][1]

    for index, item in enumerate(frequencies, start=1):
    
        relative_frequency = "1/{}".format(index)
        zipf_frequency = top_frequency * (1 / index)
        difference_actual = item[1] - zipf_frequency
        difference_percent = (item[1] / zipf_frequency) * 100

        zipf_table.append({"index": index,
                           "word": item[0],
                           "total": item[1],
                           "relative": relative_frequency,
                           "zipf": zipf_frequency,
                           "difference": difference_actual,
                           "difference_percent": difference_percent})

    return pd.DataFrame(zipf_table)


def print_zipf_table(df):

    """
    Prints the list created by generate_zipf_table
    in table format with column headings.
    """
    print(df)

def zipf_plot(df):
    counts = df.total
    words = df.word
    ranks = pylab.arange(1, len(counts)+1)
    indices = pylab.argsort(-counts)
    frequencies = counts[indices]
    pylab.plt.figure(figsize=(8,6))
    uper_bound=0
    while 10**uper_bound<frequencies[0]: uper_bound+=1
    uper_bound+=1
    pylab.plt.ylim(1,10**uper_bound)
    pylab.plt.xlim(1,10**uper_bound)
    pylab.loglog(ranks, frequencies, marker=".")
    pylab.plt.plot([1,frequencies[0]],[frequencies[0],1])
    pylab.title("Zipf plot")
    pylab.xlabel("Frequency rank of token")
    pylab.ylabel("Absolute frequency of token")
    pylab.grid(True)
    for n in list(pylab.logspace(-0.5, pylab.log10(len(counts)-2), 25).astype(int)):
        dummy = pylab.text(ranks[n], frequencies[n], " " + words[indices[n]], 
                     verticalalignment="bottom",
                     horizontalalignment="left")
    plt.show()

def frequency_plot(df):
    y_pos = np.arange(500)
    pylab.plt.figure(figsize=(10,8))
    s = 1
    expected_zipf = [df.sort_values(by='total', ascending=False)['total'][0]/(i+1)**s for i in y_pos]
    pylab.plt.bar(y_pos, df.sort_values(by='total', ascending=False)['total'][:500], align='center', alpha=0.5)
    pylab.plt.plot(y_pos, expected_zipf, linestyle='--',linewidth=2,alpha=0.5)
    pylab.plt.ylabel('Frequency')
    pylab.plt.title('Top 500 tokens in news')
    pylab.plt.show()

def top50_plot(df):
    y_pos = np.arange(50)
    pylab.plt.figure(figsize=(12,10))
    pylab.plt.bar(y_pos, df.sort_values(by='total', ascending=False)['total'][:50], align='center', alpha=0.5)
    pylab.plt.xticks(y_pos, df.sort_values(by='total', ascending=False)['word'][:50],rotation='vertical')
    pylab.plt.ylabel('Frequency')
    pylab.plt.xlabel('Top 50 tokens')
    pylab.plt.title('Top 50 tokens in news')
    pylab.plt.show()



###############################################################################################################################################################################################












def get_txt(name):
    # one file
    f = open(name, "r")
    data = f.read()
    f.close()
    return data

def get_csv(name, column):
    datadf = pd.read_csv(name)
    data = ' '.join(datadf[column].tolist())
    return data



def correct_persian(text):
    text = arabic_reshaper.reshape(text)
    text = get_display(text)
    return text


def make_all_one(dir = './persian-data',filename="allper.txt"):
    gf = open(filename, "a")
    for dirpath, dirnames, files in os.walk(dir, topdown=False):
        print(f'Found directory: {dirpath}')
        for file_name in files:
            path = f"{dirpath}/{file_name}"
            f = open(path, "r")
            text = correct_persian(text)
            gf.write(text)
            f.close()

    gf.close()

    


def main():


    while (True):
        print("""\nEnter q to exit.\n\nDataSets:\n    1. English news(https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)\n    2. Persian news(https://bigdata-ir.com/wp-content/uploads/2019/07/news-dataset.zip)""")
        i=input("Which one?(1 or 2):")
        if i=="q":break
        elif i=='1':
            data = get_csv("english_news/news.csv", "text")
            # words_num = [1, 90181, 180362, 270543, 360724, 450905, 541086, 631267, 721448, 811629, 901810, 991991, 1082172, 1172353, 1262534, 1352715, 1442896, 1533077, 1623258, 1713439, 1803619, 1893800, 1983981, 2074162, 2164343, 2254524, 2344705, 2434886, 2525067, 2615248, 2705429, 2795610, 2885791, 2975972, 3066153, 3156334, 3246515, 3336696, 3426877, 3517058, 3607238, 3697419, 3787600, 3877781, 3967962, 4058143, 4148324, 4238505, 4328686, 4418867, 4509048, 4599229, 4689410, 4779591, 4869772, 4959953, 5050134, 5140315, 5230496, 5320677, 5410857, 5501038, 5591219, 5681400, 5771581, 5861762, 5951943, 6042124, 6132305, 6222486, 6312667, 6402848, 6493029, 6583210, 6673391, 6763572, 6853753, 6943934, 7034115, 7124296, 7214476, 7304657, 7394838, 7485019, 7575200, 7665381, 7755562, 7845743, 7935924, 8026105, 8116286, 8206467, 8296648, 8386829, 8477010, 8567191, 8657372, 8747553, 8837734, 8927915, 9018095, 9108276, 9198457, 9288638, 9378819, 9469000, 9559181, 9649362, 9739543, 9829724, 9919905, 10010086, 10100267, 10190448, 10280629, 10370810, 10460991, 10551172, 10641353, 10731534, 10821714, 10911895, 11002076, 11092257, 11182438, 11272619, 11362800, 11452981, 11543162, 11633343, 11723524, 11813705, 11903886, 11994067, 12084248, 12174429, 12264610, 12354791, 12444972, 12535153, 12625333, 12715514, 12805695, 12895876, 12986057, 13076238, 13166419, 13256600, 13346781, 13436962, 13527143, 13617324, 13707505, 13797686, 13887867, 13978048, 14068229, 14158410, 14248591, 14338772, 14428952, 14519133, 14609314, 14699495, 14789676, 14879857, 14970038, 15060219, 15150400, 15240581, 15330762, 15420943, 15511124, 15601305, 15691486, 15781667, 15871848, 15962029, 16052210, 16142391, 16232571, 16322752, 16412933, 16503114, 16593295, 16683476, 16773657, 16863838, 16954019, 17044200, 17134381, 17224562, 17314743, 17404924, 17495105, 17585286, 17675467, 17765648, 17855829, 17946010]
            # distinct_words_num = [1, 13177, 19468, 24388, 28589, 32441, 35788, 38930, 41917, 44638, 47262, 49816, 52296, 54624, 56889, 59062, 61075, 63092, 65083, 67011, 68953, 70824, 72587, 74472, 76240, 77992, 79691, 81349, 82930, 84533, 86143, 87687, 89214, 90666, 92180, 93655, 95177, 96616, 98098, 99487, 100859, 102221, 103556, 104850, 106161, 107458, 108765, 109965, 111203, 112418, 113720, 114997, 116281, 117510, 118758, 119925, 121159, 122313, 123470, 124616, 125699, 126869, 128031, 129129, 130230, 131325, 132405, 133454, 134555, 135601, 136667, 137666, 138670, 139670, 140695, 141709, 142720, 143704, 144716, 145697, 146692, 147637, 148601, 149555, 150524, 151467, 152441, 153387, 154333, 155259, 156182, 157117, 157990, 158892, 159780, 160671, 161575, 162420, 163305, 164163, 165045, 165908, 166760, 167655, 168497, 169326, 170164, 170977, 171769, 172577, 173465, 174270, 175096, 175898, 176724, 177529, 178276, 179132, 179877, 180591, 181390, 182170, 182937, 183740, 184498, 185220, 185954, 186680, 187437, 188190, 188921, 189626, 190398, 191095, 191794, 192449, 193131, 193811, 194535, 195190, 195919, 196560, 197264, 197987, 198706, 199385, 200037, 200703, 201392, 202029, 202694, 203336, 203979, 204602, 205211, 205830, 206498, 207099, 207711, 208357, 208946, 209535, 210147, 210752, 211361, 211943, 212534, 213162, 213728, 214358, 214989, 215545, 216155, 216707, 217300, 217936, 218522, 219156, 219719, 220294, 220885, 221430, 222003, 222593, 223153, 223720, 224316, 224869, 225417, 225995, 226524, 227047, 227548, 228078, 228624, 229199, 229740, 230262, 230805, 231289]

        elif i=='2':
            data = get_txt("persian_news/news.txt")
            print(type(data))
            # words_num = [1, 8021, 16041, 24062, 32082, 40102, 48123, 56143, 64163, 72184, 80204, 88224, 96245, 104265, 112285, 120306, 128326, 136346, 144367, 152387, 160407, 168428, 176448, 184468, 192489, 200509, 208529, 216550, 224570, 232590, 240611, 248631, 256651, 264672, 272692, 280712, 288733, 296753, 304773, 312794, 320814, 328834, 336855, 344875, 352895, 360916, 368936, 376956, 384977, 392997, 401017, 409038, 417058, 425078, 433099, 441119, 449139, 457160, 465180, 473200, 481221, 489241, 497261, 505282, 513302, 521322, 529343, 537363, 545383, 553404, 561424, 569444, 577465, 585485, 593505, 601526, 609546, 617566, 625587, 633607, 641627, 649648, 657668, 665688, 673709, 681729, 689749, 697770, 705790, 713810, 721831, 729851, 737871, 745892, 753912, 761932, 769953, 777973, 785993, 794014, 802034, 810055, 818075, 826095, 834116, 842136, 850156, 858177, 866197, 874217, 882238, 890258, 898278, 906299, 914319, 922339, 930360, 938380, 946400, 954421, 962441, 970461, 978482, 986502, 994522, 1002543, 1010563, 1018583, 1026604, 1034624, 1042644, 1050665, 1058685, 1066705, 1074726, 1082746, 1090766, 1098787, 1106807, 1114827, 1122848, 1130868, 1138888, 1146909, 1154929, 1162949, 1170970, 1178990, 1187010, 1195031, 1203051, 1211071, 1219092, 1227112, 1235132, 1243153, 1251173, 1259193, 1267214, 1275234, 1283254, 1291275, 1299295, 1307315, 1315336, 1323356, 1331376, 1339397, 1347417, 1355437, 1363458, 1371478, 1379498, 1387519, 1395539, 1403559, 1411580, 1419600, 1427620, 1435641, 1443661, 1451681, 1459702, 1467722, 1475742, 1483763, 1491783, 1499803, 1507824, 1515844, 1523864, 1531885, 1539905, 1547925, 1555946, 1563966, 1571986, 1580007, 1588027, 1596048]
            # distinct_words_num = [1, 3039, 4793, 6127, 7336, 8358, 9311, 10188, 10962, 11735, 12467, 13197, 13826, 14452, 15041, 15622, 16191, 16742, 17262, 17727, 18268, 18769, 19255, 19718, 20185, 20623, 21089, 21538, 21949, 22419, 22853, 23253, 23666, 24053, 24447, 24839, 25202, 25597, 25980, 26355, 26714, 27050, 27403, 27772, 28075, 28398, 28739, 29097, 29421, 29749, 30049, 30410, 30725, 31067, 31383, 31681, 31992, 32306, 32610, 32908, 33161, 33479, 33748, 34049, 34339, 34628, 34887, 35138, 35420, 35714, 36018, 36294, 36588, 36864, 37121, 37390, 37647, 37921, 38179, 38428, 38664, 38907, 39152, 39398, 39677, 39932, 40194, 40424, 40715, 40969, 41249, 41489, 41730, 41962, 42214, 42484, 42735, 42965, 43192, 43405, 43636, 43858, 44091, 44307, 44561, 44781, 45032, 45280, 45475, 45689, 45924, 46144, 46381, 46614, 46830, 47038, 47226, 47442, 47675, 47876, 48100, 48314, 48532, 48750, 48973, 49155, 49353, 49578, 49770, 49962, 50174, 50373, 50591, 50810, 50998, 51198, 51382, 51576, 51785, 51973, 52185, 52385, 52568, 52755, 52953, 53159, 53344, 53550, 53734, 53947, 54162, 54395, 54577, 54763, 54932, 55135, 55326, 55500, 55691, 55877, 56062, 56239, 56417, 56601, 56778, 56951, 57119, 57295, 57463, 57650, 57827, 57982, 58136, 58312, 58467, 58637, 58818, 58975, 59163, 59343, 59538, 59714, 59881, 60060, 60248, 60410, 60570, 60754, 60920, 61113, 61274, 61452, 61620, 61799, 61957, 62119, 62302, 62486, 62671, 62847]
        
        else:
            print("Wrong!")
            break
        print("""\n    1. heaps law\n    2. zipfs las""")
        i=input("Which one?(1 or 2):")
        if i=='1':
            print("\n    1. loglog plot\n    2. frequency plot")

            words_num, distinct_words_num = generate_heap_table(data)
            
            j=input("Which one?(1 or 2):")
            if j=='1':
                heap_plot(words_num, distinct_words_num)
            elif j=='2':
                nolog_heap_plot(words_num, distinct_words_num)
            else:
                print("Wrong!")
                break
        elif i=='2':
            df = generate_zipf_table(data)
            print("\n    1. loglog plot\n    2. frequency plot\n    3. top 50 words")
            j=input("Which one?(1 or 2 or 3):")
            if j=='1':
                zipf_plot(df)
            elif j=='2':
                frequency_plot(df)
            elif j=='3':
                top50_plot(df)
            else:
                print("Wrong!")
                break
        else:
            print("Wrong!")
            break

    

main()
