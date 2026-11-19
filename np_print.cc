// Copyright 2026 Alpha Lam <arufa.hc@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.


#include "np_print.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "embedded_data.h"
#include "np_log.h"

// ---- BEGIN CIE TABLES (written by the companion profiling tools, not in this repository) ----
// The CIE 1931 2 degree standard observer and the relative spectral power of CIE illuminants D50 and D55, on
// 380-780 nm at 1 nm, as colour-science 0.4.7 gives them (D50 and D55 interpolated from the CIE's 5 nm tables).
// Three z_bar values at 650-652 nm are interpolation residue of the order of 1e-22, one of them negative, where the
// CIE's table is 0. Used for the colour of a print. export_cie_header.py of the companion profiling tools writes
// and checks this block from a CSV whose SHA-256 begins b69d080c7b908b1e; nothing in this repository does.
constexpr int kCieSamples = 401;
constexpr double kCieFirstNm = 380.0;   // 1 nm steps

// x_bar
constexpr double kCieXBar[kCieSamples] = {
    0.001368, 0.00150205, 0.001642328, 0.001802382, 0.001995757, 0.002236, 0.002535385, 0.002892603,
    0.003300829, 0.003753236, 0.004243, 0.004762389, 0.005330048, 0.005978712, 0.006741117, 0.00765,
    0.008751373, 0.01002888, 0.0114217, 0.01286901, 0.01431, 0.01570443, 0.01714744, 0.01878122,
    0.02074801, 0.02319, 0.02620736, 0.02978248, 0.03388092, 0.03846824, 0.04351, 0.0489956,
    0.0550226, 0.0617188, 0.069212, 0.07763, 0.08695811, 0.09717672, 0.1084063, 0.1207672,
    0.13438, 0.1493582, 0.1653957, 0.1819831, 0.198611, 0.21477, 0.2301868, 0.2448797,
    0.2587773, 0.2718079, 0.2839, 0.2949438, 0.3048965, 0.3137873, 0.3216454, 0.3285,
    0.3343513, 0.3392101, 0.3431213, 0.3461296, 0.34828, 0.3495999, 0.3501474, 0.350013,
    0.349287, 0.34806, 0.3463733, 0.3442624, 0.3418088, 0.3390941, 0.3362, 0.3331977,
    0.3300411, 0.3266357, 0.3228868, 0.3187, 0.3140251, 0.308884, 0.3032904, 0.2972579,
    0.2908, 0.2839701, 0.2767214, 0.2689178, 0.2604227, 0.2511, 0.2408475, 0.2298512,
    0.2184072, 0.2068115, 0.19536, 0.1842136, 0.1733273, 0.1626881, 0.1522833, 0.1421,
    0.1321786, 0.1225696, 0.1132752, 0.1042979, 0.09564, 0.08729955, 0.07930804, 0.07171776,
    0.06458099, 0.05795001, 0.05186211, 0.04628152, 0.04115088, 0.03641283, 0.03201, 0.0279172,
    0.0241444, 0.020687, 0.0175404, 0.0147, 0.01216179, 0.00991996, 0.00796724, 0.006296346,
    0.0049, 0.003777173, 0.00294532, 0.00242488, 0.002236293, 0.0024, 0.00292552, 0.00383656,
    0.00517484, 0.00698208, 0.0093, 0.01214949, 0.01553588, 0.01947752, 0.02399277, 0.0291,
    0.03481485, 0.04112016, 0.04798504, 0.05537861, 0.06327, 0.07163501, 0.08046224, 0.08973996,
    0.09945645, 0.1096, 0.1201674, 0.1311145, 0.1423679, 0.1538542, 0.1655, 0.1772571,
    0.18914, 0.2011694, 0.2133658, 0.2257499, 0.2383209, 0.2510668, 0.2639922, 0.2771017,
    0.2904, 0.3038912, 0.3175726, 0.3314384, 0.3454828, 0.3597, 0.3740839, 0.3886396,
    0.4033784, 0.4183115, 0.4334499, 0.4487953, 0.464336, 0.480064, 0.4959713, 0.5120501,
    0.5282959, 0.5446916, 0.5612094, 0.5778215, 0.5945, 0.6112209, 0.6279758, 0.6447602,
    0.6615697, 0.6784, 0.6952392, 0.7120586, 0.7288284, 0.7455188, 0.7621, 0.7785432,
    0.7948256, 0.8109264, 0.8268248, 0.8425, 0.8579325, 0.8730816, 0.8878944, 0.9023181,
    0.9163, 0.9297995, 0.9427984, 0.9552776, 0.9672179, 0.9786, 0.9893856, 0.9995488,
    1.0090892, 1.0180064, 1.0263, 1.0339827, 1.040986, 1.047188, 1.0524667, 1.0567,
    1.0597944, 1.0617992, 1.0628068, 1.0629096, 1.0622, 1.0607352, 1.0584436, 1.0552244,
    1.0509768, 1.0456, 1.0390369, 1.0313608, 1.0226662, 1.0130477, 1.0026, 0.9913675,
    0.9793314, 0.9664916, 0.9528479, 0.9384, 0.923194, 0.907244, 0.890502, 0.87292,
    0.8544499, 0.835084, 0.814946, 0.794186, 0.772954, 0.7514, 0.7295836, 0.7075888,
    0.6856022, 0.6638104, 0.6424, 0.6215149, 0.6011138, 0.5811052, 0.5613977, 0.5419,
    0.5225995, 0.5035464, 0.4847436, 0.4661939, 0.4479, 0.4298613, 0.412098, 0.394644,
    0.3775333, 0.3608, 0.3444563, 0.3285168, 0.3130192, 0.2980011, 0.2835, 0.2695448,
    0.2561184, 0.2431896, 0.2307272, 0.2187, 0.2070971, 0.1959232, 0.1851708, 0.1748323,
    0.1649, 0.1553667, 0.14623, 0.13749, 0.1291467, 0.1212, 0.1136397, 0.106465,
    0.09969044, 0.09333061, 0.0874, 0.08190096, 0.07680428, 0.07207712, 0.06768664, 0.0636,
    0.05980685, 0.05628216, 0.05297104, 0.04981861, 0.04677, 0.04378405, 0.04087536, 0.03807264,
    0.03540461, 0.0329, 0.03056419, 0.02838056, 0.02634484, 0.02445275, 0.0227, 0.02108429,
    0.01959988, 0.01823732, 0.01698717, 0.01584, 0.01479064, 0.01383132, 0.01294868, 0.0121292,
    0.01135916, 0.01062935, 0.009938846, 0.009288422, 0.008678854, 0.008110916, 0.007582388, 0.007088746,
    0.006627313, 0.006195408, 0.005790346, 0.005409826, 0.005052583, 0.004717512, 0.004403507, 0.004109457,
    0.003833913, 0.003575748, 0.003334342, 0.003109075, 0.002899327, 0.002704348, 0.00252302, 0.002354168,
    0.002196616, 0.00204919, 0.00191096, 0.001781438, 0.00166011, 0.001546459, 0.001439971, 0.001340042,
    0.001246275, 0.001158471, 0.00107643, 0.0009999493, 0.0009287358, 0.0008624332, 0.0008007503, 0.000743396,
    0.0006900786, 0.0006405156, 0.0005945021, 0.0005518646, 0.000512429, 0.0004760213, 0.0004424536, 0.0004115117,
    0.0003829814, 0.0003566491, 0.0003323011, 0.0003097586, 0.0002888871, 0.0002695394, 0.0002515682, 0.0002348261,
    0.000219171, 0.0002045258, 0.0001908405, 0.0001780654, 0.0001661505, 0.0001550236, 0.0001446219, 0.0001349098,
    0.000125852, 0.000117413, 0.0001095515, 0.0001022245, 9.539445e-05, 8.90239e-05, 8.307527e-05, 7.751269e-05,
    7.231304e-05, 6.745778e-05, 6.292844e-05, 5.870652e-05, 5.477028e-05, 5.109918e-05, 4.767654e-05, 4.448567e-05,
    4.150994e-05};

// y_bar
constexpr double kCieYBar[kCieSamples] = {
    3.9e-05, 4.28264e-05, 4.69146e-05, 5.15896e-05, 5.71764e-05, 6.4e-05, 7.234421e-05, 8.221224e-05,
    9.350816e-05, 0.0001061361, 0.00012, 0.000134984, 0.000151492, 0.000170208, 0.000191816, 0.000217,
    0.0002469067, 0.00028124, 0.00031852, 0.0003572667, 0.000396, 0.0004337147, 0.000473024, 0.000517876,
    0.0005722187, 0.00064, 0.00072456, 0.0008255, 0.00094116, 0.00106988, 0.00121, 0.001362091,
    0.001530752, 0.001720368, 0.001935323, 0.00218, 0.0024548, 0.002764, 0.0031178, 0.0035264,
    0.004, 0.00454624, 0.00515932, 0.00582928, 0.00654616, 0.0073, 0.008086507, 0.00890872,
    0.00976768, 0.01066443, 0.0116, 0.01257317, 0.01358272, 0.01462968, 0.01571509, 0.01684,
    0.01800736, 0.01921448, 0.02045392, 0.02171824, 0.023, 0.02429461, 0.02561024, 0.02695857,
    0.02835125, 0.0298, 0.03131083, 0.03288368, 0.03452112, 0.03622571, 0.038, 0.03984667,
    0.041768, 0.043766, 0.04584267, 0.048, 0.05024368, 0.05257304, 0.05498056, 0.05745872,
    0.06, 0.06260197, 0.06527752, 0.06804208, 0.07091109, 0.0739, 0.077016, 0.0802664,
    0.0836668, 0.0872328, 0.09098, 0.09491755, 0.09904584, 0.1033674, 0.1078846, 0.1126,
    0.117532, 0.1226744, 0.1279928, 0.1334528, 0.13902, 0.1446764, 0.1504693, 0.1564619,
    0.1627177, 0.1693, 0.1762431, 0.1835581, 0.1912735, 0.199418, 0.20802, 0.2171199,
    0.2267345, 0.2368571, 0.2474812, 0.2586, 0.2701849, 0.2822939, 0.2950505, 0.308578,
    0.323, 0.3384021, 0.3546858, 0.3716986, 0.3892875, 0.4073, 0.4256299, 0.4443096,
    0.4633944, 0.4829395, 0.503, 0.5235693, 0.544512, 0.56569, 0.5869653, 0.6082,
    0.6293456, 0.6503068, 0.6708752, 0.6908424, 0.71, 0.7281852, 0.7454636, 0.7619694,
    0.7778368, 0.7932, 0.8081104, 0.8224962, 0.8363068, 0.8494916, 0.862, 0.8738108,
    0.8849624, 0.8954936, 0.9054432, 0.9148501, 0.9237348, 0.9320924, 0.9399226, 0.9472252,
    0.954, 0.9602561, 0.9660074, 0.9712606, 0.9760225, 0.9803, 0.9840924, 0.9874182,
    0.9903128, 0.9928116, 0.9949501, 0.9967108, 0.9980983, 0.999112, 0.9997482, 1,
    0.9998567, 0.9993046, 0.9983255, 0.9968987, 0.995, 0.9926005, 0.9897426, 0.9864444,
    0.9827241, 0.9786, 0.9740837, 0.9691712, 0.9638568, 0.9581349, 0.952, 0.9454504,
    0.9384992, 0.9311628, 0.9234576, 0.9154, 0.9070064, 0.8982772, 0.8892048, 0.8797816,
    0.87, 0.8598613, 0.849392, 0.838622, 0.8275813, 0.8163, 0.8047947, 0.793082,
    0.781192, 0.7691547, 0.757, 0.7447541, 0.7324224, 0.7200036, 0.7074965, 0.6949,
    0.6822192, 0.6694716, 0.6566744, 0.6438448, 0.631, 0.6181555, 0.6053144, 0.5924756,
    0.5796379, 0.5668, 0.5539611, 0.5411372, 0.5283528, 0.5156323, 0.503, 0.4904688,
    0.4780304, 0.4656776, 0.4534032, 0.4412, 0.42908, 0.417036, 0.405032, 0.393032,
    0.381, 0.3689184, 0.3568272, 0.3447768, 0.3328176, 0.321, 0.3093381, 0.2978504,
    0.2865936, 0.2756245, 0.265, 0.2547632, 0.2448896, 0.2353344, 0.2260528, 0.217,
    0.2081616, 0.1995488, 0.1911552, 0.1829744, 0.175, 0.1672235, 0.1596464, 0.1522776,
    0.1451259, 0.1382, 0.1315003, 0.1250248, 0.1187792, 0.1127691, 0.107, 0.1014762,
    0.09618864, 0.09112296, 0.08626485, 0.0816, 0.07712064, 0.07282552, 0.06871008, 0.06476976,
    0.061, 0.05739621, 0.05395504, 0.05067376, 0.04754965, 0.04458, 0.04175872, 0.03908496,
    0.03656384, 0.03420048, 0.032, 0.02996261, 0.02807664, 0.02632936, 0.02470805, 0.0232,
    0.02180077, 0.02050112, 0.01928108, 0.01812069, 0.017, 0.01590379, 0.01483718, 0.01381068,
    0.01283478, 0.01192, 0.01106831, 0.01027339, 0.009533311, 0.008846157, 0.00821, 0.007623781,
    0.007085424, 0.006591476, 0.006138485, 0.005723, 0.005343059, 0.004995796, 0.004676404, 0.004380075,
    0.004102, 0.003838453, 0.003589099, 0.003354219, 0.003134093, 0.002929, 0.002738139, 0.002559876,
    0.002393244, 0.002237275, 0.002091, 0.001953587, 0.00182458, 0.00170358, 0.001590187, 0.001484,
    0.001384496, 0.001291268, 0.001204092, 0.001122744, 0.001047, 0.0009765896, 0.0009111088, 0.0008501332,
    0.0007932384, 0.00074, 0.0006900827, 0.00064331, 0.000599496, 0.0005584547, 0.00052, 0.0004839136,
    0.0004500528, 0.0004183452, 0.0003887184, 0.0003611, 0.0003353835, 0.0003114404, 0.0002891656, 0.0002684539,
    0.0002492, 0.0002313019, 0.0002146856, 0.0001992884, 0.0001850475, 0.0001719, 0.0001597781, 0.0001486044,
    0.0001383016, 0.0001287925, 0.00012, 0.0001118595, 0.0001043224, 9.73356e-05, 9.084587e-05, 8.48e-05,
    7.914667e-05, 7.3858e-05, 6.8916e-05, 6.430267e-05, 6e-05, 5.598187e-05, 5.22256e-05, 4.87184e-05,
    4.544747e-05, 4.24e-05, 3.956104e-05, 3.691512e-05, 3.444868e-05, 3.214816e-05, 3e-05, 2.799125e-05,
    2.611356e-05, 2.436024e-05, 2.272461e-05, 2.12e-05, 1.977855e-05, 1.845285e-05, 1.721687e-05, 1.606459e-05,
    1.499e-05};

// z_bar
constexpr double kCieZBar[kCieSamples] = {
    0.006450001, 0.007083216, 0.007745488, 0.008501152, 0.009414544, 0.01054999, 0.0119658, 0.01365587,
    0.01558805, 0.01773015, 0.02005001, 0.02251136, 0.02520288, 0.02827972, 0.03189704, 0.03621,
    0.04143771, 0.04750372, 0.05411988, 0.06099803, 0.06785001, 0.07448632, 0.08136156, 0.08915364,
    0.09854048, 0.1102, 0.1246133, 0.1417017, 0.1613035, 0.1832568, 0.2074, 0.2336921,
    0.2626114, 0.2947746, 0.3307985, 0.3713, 0.4162091, 0.4654642, 0.5196948, 0.5795303,
    0.6456, 0.7184838, 0.7967133, 0.8778459, 0.959439, 1.0390501, 1.1153673, 1.1884971,
    1.2581233, 1.3239296, 1.3856, 1.4426352, 1.4948035, 1.5421903, 1.5848807, 1.62296,
    1.6564048, 1.6852959, 1.7098745, 1.7303821, 1.74706, 1.7600446, 1.7696233, 1.7762637,
    1.7804334, 1.7826, 1.7829682, 1.7816998, 1.7791982, 1.7758671, 1.77211, 1.7682589,
    1.764039, 1.7589438, 1.7524663, 1.7441, 1.7335595, 1.7208581, 1.7059369, 1.6887372,
    1.6692, 1.6475287, 1.6234127, 1.5960223, 1.564528, 1.5281, 1.4861114, 1.4395215,
    1.3898799, 1.3387362, 1.28764, 1.2374223, 1.1878243, 1.1387611, 1.090148, 1.0419,
    0.9941976, 0.9473473, 0.9014531, 0.8566193, 0.8129501, 0.7705173, 0.7294448, 0.6899136,
    0.6521049, 0.6162, 0.5823286, 0.5504162, 0.5203376, 0.4919673, 0.46518, 0.4399246,
    0.4161836, 0.3938822, 0.3729459, 0.3533, 0.3348578, 0.3175521, 0.3013375, 0.2861686,
    0.272, 0.2588171, 0.2464838, 0.2347718, 0.2234533, 0.2123, 0.2011692, 0.1901196,
    0.1792254, 0.1685608, 0.1582, 0.1481383, 0.1383758, 0.1289942, 0.1200751, 0.1117,
    0.1039048, 0.09666748, 0.08998272, 0.08384531, 0.07824999, 0.07320899, 0.06867816, 0.06456784,
    0.06078835, 0.05725001, 0.05390435, 0.05074664, 0.04775276, 0.04489859, 0.04216, 0.03950728,
    0.03693564, 0.03445836, 0.03208872, 0.02984, 0.02771181, 0.02569444, 0.02378716, 0.02198925,
    0.0203, 0.01871805, 0.01724036, 0.01586364, 0.01458461, 0.0134, 0.01230723, 0.01130188,
    0.01037792, 0.009529306, 0.008749999, 0.0080352, 0.0073816, 0.0067854, 0.0062428, 0.005749999,
    0.0053036, 0.0048998, 0.0045342, 0.0042024, 0.0039, 0.0036232, 0.0033706, 0.0031414,
    0.0029348, 0.002749999, 0.0025852, 0.0024386, 0.0023094, 0.0021968, 0.0021, 0.002017733,
    0.0019482, 0.0018898, 0.001840933, 0.0018, 0.001766267, 0.0017378, 0.0017112, 0.001683067,
    0.001650001, 0.001610133, 0.0015644, 0.0015136, 0.001458533, 0.0014, 0.001336667, 0.00127,
    0.001205, 0.001146667, 0.0011, 0.0010688, 0.0010494, 0.0010356, 0.0010212, 0.001,
    0.00096864, 0.00092992, 0.00088688, 0.00084256, 0.0008, 0.00076096, 0.00072368, 0.00068592,
    0.00064544, 0.0006, 0.0005478667, 0.0004916, 0.0004354, 0.0003834667, 0.00034, 0.0003072533,
    0.00028316, 0.00026544, 0.0002518133, 0.00024, 0.0002295467, 0.00022064, 0.00021196, 0.0002021867,
    0.00019, 0.0001742133, 0.00015564, 0.00013596, 0.0001168533, 0.0001, 8.613333e-05, 7.46e-05,
    6.5e-05, 5.693333e-05, 4.999999e-05, 4.416e-05, 3.948e-05, 3.572e-05, 3.264e-05, 3e-05,
    2.765333e-05, 2.556e-05, 2.364e-05, 2.181333e-05, 2e-05, 1.813333e-05, 1.62e-05, 1.42e-05,
    1.213333e-05, 1e-05, 7.733333e-06, 5.4e-06, 3.2e-06, 1.333333e-06, 2.1175824e-22, -3.1763736e-22,
    5.2939559e-23, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0};

// D50
constexpr double kCieD50[kCieSamples] = {
    24.488, 25.0262, 25.5644, 26.1026, 26.6408, 27.179, 27.7174, 28.2558,
    28.7942, 29.3326, 29.871, 31.8146, 33.7582, 35.7018, 37.6454, 39.589,
    41.5328, 43.4766, 45.4204, 47.3642, 49.308, 50.0284, 50.7488, 51.4692,
    52.1896, 52.91, 53.6306, 54.3512, 55.0718, 55.7924, 56.513, 56.865,
    57.217, 57.569, 57.921, 58.273, 58.6252, 58.9774, 59.3296, 59.6818,
    60.034, 59.8124, 59.5908, 59.3692, 59.1476, 58.926, 58.7044, 58.4828,
    58.2612, 58.0396, 57.818, 59.5186, 61.2192, 62.9198, 64.6204, 66.321,
    68.0218, 69.7226, 71.4234, 73.1242, 74.825, 76.0672, 77.3094, 78.5516,
    79.7938, 81.036, 82.2782, 83.5204, 84.7626, 86.0048, 87.247, 87.5836,
    87.9202, 88.2568, 88.5934, 88.93, 89.2664, 89.6028, 89.9392, 90.2756,
    90.612, 90.6876, 90.7632, 90.8388, 90.9144, 90.99, 91.0656, 91.1412,
    91.2168, 91.2924, 91.368, 91.742, 92.116, 92.49, 92.864, 93.238,
    93.6122, 93.9864, 94.3606, 94.7348, 95.109, 94.7944, 94.4798, 94.1652,
    93.8506, 93.536, 93.2214, 92.9068, 92.5922, 92.2776, 91.963, 92.339,
    92.715, 93.091, 93.467, 93.843, 94.2192, 94.5954, 94.9716, 95.3478,
    95.724, 95.813, 95.902, 95.991, 96.08, 96.169, 96.2578, 96.3466,
    96.4354, 96.5242, 96.613, 96.6646, 96.7162, 96.7678, 96.8194, 96.871,
    96.9226, 96.9742, 97.0258, 97.0774, 97.129, 97.626, 98.123, 98.62,
    99.117, 99.614, 100.111, 100.608, 101.105, 101.602, 102.099, 101.965,
    101.83, 101.696, 101.561, 101.427, 101.293, 101.158, 101.024, 100.889,
    100.755, 100.911, 101.067, 101.224, 101.38, 101.536, 101.692, 101.848,
    102.005, 102.161, 102.317, 102.085, 101.854, 101.622, 101.391, 101.159,
    100.927, 100.695, 100.464, 100.232, 100, 99.7736, 99.5472, 99.3208,
    99.0944, 98.868, 98.6414, 98.4148, 98.1882, 97.9616, 97.735, 97.8534,
    97.9718, 98.0902, 98.2086, 98.327, 98.4452, 98.5634, 98.6816, 98.7998,
    98.918, 98.376, 97.834, 97.292, 96.75, 96.208, 95.6662, 95.1244,
    94.5826, 94.0408, 93.499, 93.9178, 94.3366, 94.7554, 95.1742, 95.593,
    96.012, 96.431, 96.85, 97.269, 97.688, 97.846, 98.004, 98.162,
    98.32, 98.478, 98.6362, 98.7944, 98.9526, 99.1108, 99.269, 99.2462,
    99.2234, 99.2006, 99.1778, 99.155, 99.1324, 99.1098, 99.0872, 99.0646,
    99.042, 98.71, 98.378, 98.046, 97.714, 97.382, 97.05, 96.718,
    96.386, 96.054, 95.722, 96.0356, 96.3492, 96.6628, 96.9764, 97.29,
    97.6034, 97.9168, 98.2302, 98.5436, 98.857, 98.538, 98.219, 97.9,
    97.581, 97.262, 96.943, 96.624, 96.305, 95.986, 95.667, 95.9194,
    96.1718, 96.4242, 96.6766, 96.929, 97.1812, 97.4334, 97.6856, 97.9378,
    98.19, 98.6714, 99.1528, 99.6342, 100.116, 100.597, 101.078, 101.559,
    102.041, 102.522, 103.003, 102.616, 102.229, 101.842, 101.455, 101.068,
    100.681, 100.294, 99.907, 99.52, 99.133, 97.9578, 96.7826, 95.6074,
    94.4322, 93.257, 92.0818, 90.9066, 89.7314, 88.5562, 87.381, 87.8032,
    88.2254, 88.6476, 89.0698, 89.492, 89.9144, 90.3368, 90.7592, 91.1816,
    91.604, 91.7324, 91.8608, 91.9892, 92.1176, 92.246, 92.3746, 92.5032,
    92.6318, 92.7604, 92.889, 91.2856, 89.6822, 88.0788, 86.4754, 84.872,
    83.2684, 81.6648, 80.0612, 78.4576, 76.854, 77.8198, 78.7856, 79.7514,
    80.7172, 81.683, 82.6486, 83.6142, 84.5798, 85.5454, 86.511, 87.118,
    87.725, 88.332, 88.939, 89.546, 90.1528, 90.7596, 91.3664, 91.9732,
    92.58, 91.145, 89.71, 88.275, 86.84, 85.405, 83.97, 82.535,
    81.1, 79.665, 78.23, 76.1762, 74.1224, 72.0686, 70.0148, 67.961,
    65.9072, 63.8534, 61.7996, 59.7458, 57.692, 60.215, 62.738, 65.261,
    67.784, 70.307, 72.8302, 75.3534, 77.8766, 80.3998, 82.923, 82.4582,
    81.9934, 81.5286, 81.0638, 80.599, 80.134, 79.669, 79.204, 78.739,
    78.274};

// D55
constexpr double kCieD55[kCieSamples] = {
    32.584, 33.1342, 33.6844, 34.2346, 34.7848, 35.335, 35.8854, 36.4358,
    36.9862, 37.5366, 38.087, 40.3732, 42.6594, 44.9456, 47.2318, 49.518,
    51.8042, 54.0904, 56.3766, 58.6628, 60.949, 61.7094, 62.4698, 63.2302,
    63.9906, 64.751, 65.5116, 66.2722, 67.0328, 67.7934, 68.554, 68.8562,
    69.1584, 69.4606, 69.7628, 70.065, 70.3674, 70.6698, 70.9722, 71.2746,
    71.577, 71.2108, 70.8446, 70.4784, 70.1122, 69.746, 69.3796, 69.0132,
    68.6468, 68.2804, 67.914, 69.6832, 71.4524, 73.2216, 74.9908, 76.76,
    78.529, 80.298, 82.067, 83.836, 85.605, 86.8438, 88.0826, 89.3214,
    90.5602, 91.799, 93.0378, 94.2766, 95.5154, 96.7542, 97.993, 98.24,
    98.487, 98.734, 98.981, 99.228, 99.475, 99.722, 99.969, 100.216,
    100.463, 100.408, 100.353, 100.298, 100.243, 100.188, 100.133, 100.078,
    100.023, 99.968, 99.913, 100.196, 100.478, 100.761, 101.043, 101.326,
    101.609, 101.891, 102.174, 102.456, 102.739, 102.273, 101.807, 101.341,
    100.875, 100.409, 99.9428, 99.4766, 99.0104, 98.5442, 98.078, 98.3382,
    98.5984, 98.8586, 99.1188, 99.379, 99.6392, 99.8994, 100.16, 100.42,
    100.68, 100.682, 100.683, 100.685, 100.686, 100.688, 100.689, 100.691,
    100.692, 100.694, 100.695, 100.624, 100.553, 100.483, 100.412, 100.341,
    100.27, 100.199, 100.129, 100.058, 99.987, 100.409, 100.831, 101.254,
    101.676, 102.098, 102.52, 102.943, 103.365, 103.788, 104.21, 103.999,
    103.788, 103.578, 103.367, 103.156, 102.945, 102.734, 102.524, 102.313,
    102.102, 102.189, 102.275, 102.362, 102.448, 102.535, 102.622, 102.708,
    102.795, 102.881, 102.968, 102.671, 102.374, 102.078, 101.781, 101.484,
    101.187, 100.89, 100.594, 100.297, 100, 99.7216, 99.4432, 99.1648,
    98.8864, 98.608, 98.3296, 98.0512, 97.7728, 97.4944, 97.216, 97.2692,
    97.3224, 97.3756, 97.4288, 97.482, 97.5354, 97.5888, 97.6422, 97.6956,
    97.749, 97.1172, 96.4854, 95.8536, 95.2218, 94.59, 93.9584, 93.3268,
    92.6952, 92.0636, 91.432, 91.7308, 92.0296, 92.3284, 92.6272, 92.926,
    93.2246, 93.5232, 93.8218, 94.1204, 94.419, 94.4912, 94.5634, 94.6356,
    94.7078, 94.78, 94.852, 94.924, 94.996, 95.068, 95.14, 95.048,
    94.956, 94.864, 94.772, 94.68, 94.588, 94.496, 94.404, 94.312,
    94.22, 93.8428, 93.4656, 93.0884, 92.7112, 92.334, 91.9568, 91.5796,
    91.2024, 90.8252, 90.448, 90.6362, 90.8244, 91.0126, 91.2008, 91.389,
    91.5772, 91.7654, 91.9536, 92.1418, 92.33, 91.9824, 91.6348, 91.2872,
    90.9396, 90.592, 90.2444, 89.8968, 89.5492, 89.2016, 88.854, 89.0004,
    89.1468, 89.2932, 89.4396, 89.586, 89.7322, 89.8784, 90.0246, 90.1708,
    90.317, 90.6802, 91.0434, 91.4066, 91.7698, 92.133, 92.4964, 92.8598,
    93.2232, 93.5866, 93.95, 93.5506, 93.1512, 92.7518, 92.3524, 91.953,
    91.5536, 91.1542, 90.7548, 90.3554, 89.956, 88.9282, 87.9004, 86.8726,
    85.8448, 84.817, 83.789, 82.761, 81.733, 80.705, 79.677, 79.9932,
    80.3094, 80.6256, 80.9418, 81.258, 81.5744, 81.8908, 82.2072, 82.5236,
    82.84, 83.0404, 83.2408, 83.4412, 83.6416, 83.842, 84.0424, 84.2428,
    84.4432, 84.6436, 84.844, 83.383, 81.922, 80.461, 79, 77.539,
    76.0782, 74.6174, 73.1566, 71.6958, 70.235, 71.1416, 72.0482, 72.9548,
    73.8614, 74.768, 75.6746, 76.5812, 77.4878, 78.3944, 79.301, 79.8702,
    80.4394, 81.0086, 81.5778, 82.147, 82.7162, 83.2854, 83.8546, 84.4238,
    84.993, 83.6818, 82.3706, 81.0594, 79.7482, 78.437, 77.1256, 75.8142,
    74.5028, 73.1914, 71.88, 69.9714, 68.0628, 66.1542, 64.2456, 62.337,
    60.4282, 58.5194, 56.6106, 54.7018, 52.793, 55.1064, 57.4198, 59.7332,
    62.0466, 64.36, 66.6734, 68.9868, 71.3002, 73.6136, 75.927, 75.516,
    75.105, 74.694, 74.283, 73.872, 73.4612, 73.0504, 72.6396, 72.2288,
    71.818};
// ---- END CIE TABLES ----

void density_encode_in_place(uint16_t* pixels, int width, int height, int channels, const float merged[9],
                             double base_level, const double m[9], const DensityEncoding& enc, DensityClip& clip) {
  const double inv_base = 1.0 / base_level;
  const double offset = enc.offset, scale = enc.scale;
  const long max_code = enc.max_code();
  uint64_t low[3] = {0, 0, 0}, high[3] = {0, 0, 0}, floor[3] = {0, 0, 0};
  #pragma omp parallel for schedule(static) reduction(+ : low[:3], high[:3], floor[:3])
  for (int j = 0; j < height; ++j) {
    uint16_t* row = pixels + (size_t)j * width * channels;
    for (int i = 0; i < width; ++i) {
      uint16_t* px = row + (size_t)i * channels;
      const double r = px[0] / 65535.0, g = px[1] / 65535.0, b = px[2] / 65535.0;
      double d[3];
      for (int c = 0; c < 3; ++c) {
        const double v = (double)merged[c * 3] * r + (double)merged[c * 3 + 1] * g + (double)merged[c * 3 + 2] * b;
        const double t = v * inv_base;
        if (!(t > kDensityMinT)) ++floor[c];
        d[c] = -std::log10(std::max(t, kDensityMinT));
      }
      for (int c = 0; c < 3; ++c) {
        const double dens = m[c * 3] * d[0] + m[c * 3 + 1] * d[1] + m[c * 3 + 2] * d[2];
        long code = std::lround(offset + scale * enc.k[c] * dens);
        if (code < 0) {
          code = 0;
          ++low[c];
        } else if (code > max_code) {
          code = max_code;
          ++high[c];
        }
        px[c] = (uint16_t)code;
      }
    }
  }
  for (int c = 0; c < 3; ++c) {
    clip.low[c] = low[c];
    clip.high[c] = high[c];
    clip.floor[c] = floor[c];
  }
  clip.pixels = (uint64_t)width * (uint64_t)height;
}

std::string density_metadata_json(const DensityRecord& rec, int indent) {
  using nlohmann::ordered_json;
  ordered_json j;
  j["schema"] = "negicc.density/1";
  j["density"] = density_measure_name(rec.measure);
  if (rec.matrix && !rec.matrix->target_curves.empty()) j["target_curves"] = rec.matrix->target_curves;
  j["encoding"] = {{"container", rec.container},
                   {"bits", rec.encoding.bits},
                   {"offset", rec.encoding.offset},
                   {"scale", rec.encoding.scale},
                   {"channel_factors", {rec.encoding.k[0], rec.encoding.k[1], rec.encoding.k[2]}},
                   {"standard", rec.measure == kDensityApd ? "ADX, SMPTE ST 2065-3 (the channel factors applied)"
                                                           : (rec.measure == kDensityRp180 ? "Cineon scaling" : "none")},
                   {"formula", "code = offset + scale * k * D, D the density above the film base, k the channel factor"}};
  j["profile_bundle"] = rec.bundle_key;
  j["film_base"] = {{"rgb", {rec.film_base_rgb[0], rec.film_base_rgb[1], rec.film_base_rgb[2]}},
                    {"source", rec.film_base_source}};
  if (rec.matrix) {
    const double* m = rec.matrix->m;
    j["density_matrix"] = {{m[0], m[1], m[2]}, {m[3], m[4], m[5]}, {m[6], m[7], m[8]}};
  }
  if (rec.bundle) {
    if (!rec.bundle->density_fit_json.empty()) {
      j["density_fit"] = ordered_json::parse(rec.bundle->density_fit_json, nullptr, false);
    }
    if (!rec.bundle->light_source_json.empty()) {
      j["light_source"] = ordered_json::parse(rec.bundle->light_source_json, nullptr, false);
      // The bundle's three LED spectra (led_spectra) are left out when they are the all-on spectrum cut at its band
      // splits: the spectrum and the splits, both recorded, give them back. Spectra measured one LED at a time stay.
      ordered_json& light = j["light_source"];
      if (light.is_object() && light.contains("led_spectra") && light["led_spectra"].contains("band_splits_nm")) {
        light.erase("led_spectra");
      }
    }
  }
  if (rec.clip) {
    const DensityClip& c = *rec.clip;
    j["clipped_pixels"] = {{"below_code_0", {c.low[0], c.low[1], c.low[2]}},
                           {"above_max_code", {c.high[0], c.high[1], c.high[2]}},
                           {"at_density_floor", {c.floor[0], c.floor[1], c.floor[2]}},
                           {"pixels", c.pixels}};
  }
  // The picture conversion multiplies the light's green and blue channels by g and b to balance the frame; in
  // density that is an offset per channel. Through the measure's matrix, and with their mean taken out so that the
  // file's overall level stays, the offsets are the colour balance alone. The exposure E is recorded and left to
  // the grade: it is relative to the picture conversion's film profile, which a density file does not use.
  double measure[3], mean = 0.0;
  if (density_balance_offset(rec, measure, &mean)) {
    j["suggested_balance"] = {
        {"from", "the picture conversion's parameter search on this frame; not applied to the densities"},
        {"exposure", rec.balance_e},
        {"gain_g", rec.balance_g},
        {"gain_b", rec.balance_b},
        {"bracket", rec.balance_target},
        {"density_offset", {measure[0], measure[1], measure[2]}},
        {"code_offset",
         {measure[0] * rec.encoding.scale * rec.encoding.k[0], measure[1] * rec.encoding.scale * rec.encoding.k[1],
          measure[2] * rec.encoding.scale * rec.encoding.k[2]}},
        {"density_offset_mean_removed", mean},
        {"use", "D + density_offset (code + code_offset) has the colour balance the picture conversion gives this "
                "frame; the offsets sum to zero, so the overall level is unchanged and the exposure is left to the "
                "grade. The gains also shift the level, by density_offset_mean_removed in every channel; the print "
                "CLF beside the file, which sets the exposure, applies density_offset + density_offset_mean_removed, "
                "except for a print material with its own print search, whose CLF applies the offsets of that "
                "search's gains (print.materials.<id>.balance)"}};
  }
  // How the frame prints on each print material (below), computed for this frame and not applied to it.
  if (!rec.print_json.empty()) j["print"] = ordered_json::parse(rec.print_json, nullptr, false);
  j["tool"] = rec.tool_version;
  return j.dump(indent);
}

bool density_balance_offset(const DensityRecord& rec, double offset[3], double* mean_out) {
  offset[0] = offset[1] = offset[2] = 0.0;
  if (mean_out) *mean_out = 0.0;
  if (!(rec.has_balance && rec.balance_e > 0.0 && rec.balance_g > 0.0 && rec.balance_b > 0.0)) return false;
  const double light[3] = {0.0, -std::log10(rec.balance_g), -std::log10(rec.balance_b)};
  for (int c = 0; c < 3; ++c) offset[c] = light[c];
  if (rec.matrix) {
    const double* m = rec.matrix->m;
    for (int c = 0; c < 3; ++c) offset[c] = m[c * 3] * light[0] + m[c * 3 + 1] * light[1] + m[c * 3 + 2] * light[2];
  }
  const double mean = (offset[0] + offset[1] + offset[2]) / 3.0;
  for (int c = 0; c < 3; ++c) offset[c] -= mean;
  if (mean_out) *mean_out = mean;
  return true;
}

namespace {

std::string date_time(long long timestamp, char sep) {
  time_t t = timestamp > 0 ? (time_t)timestamp : time(nullptr);
  struct tm tm_buf;
  localtime_r(&t, &tm_buf);
  char buf[80];
  snprintf(buf, sizeof buf, "%04d:%02d:%02d%c%02d:%02d:%02d", tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
           sep, tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
  return buf;
}

// ---- TIFF (little-endian, one strip) --------------------------------------------------------------------------------
struct TiffEntry {
  uint16_t tag, type;
  uint32_t count, value;
};

void put16(std::vector<uint8_t>& b, uint16_t v) {
  b.push_back((uint8_t)(v & 0xff));
  b.push_back((uint8_t)(v >> 8));
}
void put32(std::vector<uint8_t>& b, uint32_t v) {
  for (int k = 0; k < 4; ++k) b.push_back((uint8_t)((v >> (8 * k)) & 0xff));
}

// ---- DPX (big-endian) -----------------------------------------------------------------------------------------------
void be16(uint8_t* p, uint16_t v) {
  p[0] = (uint8_t)(v >> 8);
  p[1] = (uint8_t)(v & 0xff);
}
void be32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)(v >> 24);
  p[1] = (uint8_t)((v >> 16) & 0xff);
  p[2] = (uint8_t)((v >> 8) & 0xff);
  p[3] = (uint8_t)(v & 0xff);
}
void be_float(uint8_t* p, float f) {
  uint32_t v;
  memcpy(&v, &f, 4);
  be32(p, v);
}
void put_str(uint8_t* p, size_t field, const std::string& s) {
  memcpy(p, s.data(), std::min(field - 1, s.size()));
}

}  // namespace

int write_density_tiff(const std::string& path, const uint16_t* pixels, int width, int height, int channels,
                       const std::string& description, const DensityFileInfo& info) {
  const uint64_t strip_bytes = (uint64_t)width * (uint64_t)height * 6;
  if (strip_bytes > 0xffffff00ull) {
    log_error("The image is too large for a TIFF file (%dx%d)\n", width, height);
    return -1;
  }
  // The values that do not fit an entry, laid out after the IFD.
  const std::string date = date_time(info.timestamp, ' ');
  std::vector<TiffEntry> entries;
  std::vector<uint8_t> extra;
  const int n_entries = 18 + (info.make.empty() ? 0 : 1) + (info.model.empty() ? 0 : 1);
  const uint32_t ifd_offset = 8;
  const uint32_t extra_offset = ifd_offset + 2 + (uint32_t)n_entries * 12 + 4;
  auto add_extra = [&](const void* data, size_t n) {
    const uint32_t at = extra_offset + (uint32_t)extra.size();
    const uint8_t* p = (const uint8_t*)data;
    extra.insert(extra.end(), p, p + n);
    if (extra.size() % 2) extra.push_back(0);
    return at;
  };
  // An ASCII value with its terminating NUL; four bytes or fewer are stored in the entry itself (TIFF 6.0).
  auto ascii = [&](uint16_t tag, const std::string& s) {
    const uint32_t n = (uint32_t)s.size() + 1;
    if (n <= 4) {
      uint32_t v = 0;
      for (uint32_t k = 0; k < n; ++k) v |= (uint32_t)(uint8_t)s[k] << (8 * k);   // the bytes in file order
      entries.push_back({tag, 2, n, v});
      return;
    }
    entries.push_back({tag, 2, n, add_extra(s.c_str(), n)});
  };
  const uint16_t bps[3] = {16, 16, 16};
  const uint32_t res[2] = {300, 1};
  entries.push_back({254, 4, 1, 0});                       // NewSubfileType
  entries.push_back({256, 4, 1, (uint32_t)width});         // ImageWidth
  entries.push_back({257, 4, 1, (uint32_t)height});        // ImageLength
  {
    std::vector<uint8_t> b;
    for (uint16_t v : bps) put16(b, v);
    entries.push_back({258, 3, 3, add_extra(b.data(), b.size())});  // BitsPerSample
  }
  entries.push_back({259, 3, 1, 1});                       // Compression: none
  entries.push_back({262, 3, 1, 2});                       // PhotometricInterpretation: RGB
  ascii(270, description);                                 // ImageDescription
  if (!info.make.empty()) ascii(271, info.make);
  if (!info.model.empty()) ascii(272, info.model);
  const size_t strip_entry = entries.size();
  entries.push_back({273, 4, 1, 0});                       // StripOffsets (filled below)
  entries.push_back({274, 3, 1, 1});                       // Orientation: top left
  entries.push_back({277, 3, 1, 3});                       // SamplesPerPixel
  entries.push_back({278, 4, 1, (uint32_t)height});        // RowsPerStrip
  entries.push_back({279, 4, 1, (uint32_t)strip_bytes});   // StripByteCounts
  {
    std::vector<uint8_t> b;
    put32(b, res[0]);
    put32(b, res[1]);
    const uint32_t at = add_extra(b.data(), b.size());
    entries.push_back({282, 5, 1, at});                    // XResolution
    entries.push_back({283, 5, 1, at});                    // YResolution
  }
  entries.push_back({284, 3, 1, 1});                       // PlanarConfiguration: chunky
  entries.push_back({296, 3, 1, 2});                       // ResolutionUnit: inch
  ascii(305, info.software);                               // Software
  ascii(306, date);                                        // DateTime
  if ((int)entries.size() != n_entries) {
    log_error("Internal error: the TIFF directory has %d entries, expected %d\n", (int)entries.size(), n_entries);
    return -1;
  }
  const uint32_t data_offset = extra_offset + (uint32_t)extra.size();
  entries[strip_entry].value = data_offset;

  std::vector<uint8_t> head;
  head.push_back('I');
  head.push_back('I');
  put16(head, 42);
  put32(head, ifd_offset);
  put16(head, (uint16_t)entries.size());
  for (const TiffEntry& e : entries) {
    put16(head, e.tag);
    put16(head, e.type);
    put32(head, e.count);
    if (e.type == 3 && e.count == 1) {
      put16(head, (uint16_t)e.value);
      put16(head, 0);
    } else {
      put32(head, e.value);
    }
  }
  put32(head, 0);  // no further directory
  head.insert(head.end(), extra.begin(), extra.end());

  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) {
    log_error("Cannot open %s for writing\n", path.c_str());
    return -1;
  }
  bool ok = fwrite(head.data(), 1, head.size(), fp) == head.size();
  // The samples little-endian, as the header says ("II"), whatever the host's byte order; three per pixel.
  std::vector<uint8_t> row((size_t)width * 6);
  for (int j = 0; ok && j < height; ++j) {
    const uint16_t* src = pixels + (size_t)j * width * channels;
    for (int i = 0; i < width; ++i) {
      for (int c = 0; c < 3; ++c) {
        const uint16_t v = src[(size_t)i * channels + c];
        row[(size_t)i * 6 + 2 * c] = (uint8_t)(v & 0xff);
        row[(size_t)i * 6 + 2 * c + 1] = (uint8_t)(v >> 8);
      }
    }
    ok = fwrite(row.data(), 1, row.size(), fp) == row.size();
  }
  ok = (fclose(fp) == 0) && ok;
  if (!ok) log_error("Failed to write %s\n", path.c_str());
  return ok ? 0 : -1;
}

int write_density_dpx(const std::string& path, const uint16_t* pixels, int width, int height, int channels,
                      const std::string& description, const DensityEncoding& enc, DensityMeasure measure,
                      const DensityFileInfo& info) {
  if (enc.bits != 10) {
    log_error("Internal error: a DPX holds 10-bit density codes\n");
    return -1;
  }
  // The user-defined data: a 32-byte identification, then the metadata JSON; the image starts on a multiple of 8.
  const uint32_t kHeader = 2048, kUserId = 32, kMaxUser = 1024 * 1024;
  uint32_t user_size = kUserId + (uint32_t)description.size() + 1;
  user_size = (user_size + 7u) & ~7u;
  if (user_size > kMaxUser) {
    log_error("The density metadata (%u bytes) does not fit a DPX file's user data (1 MB)\n", user_size);
    return -1;
  }
  const uint64_t image_bytes = (uint64_t)width * (uint64_t)height * 4;
  const uint32_t image_offset = kHeader + user_size;
  if (image_bytes + image_offset > 0xffffff00ull) {
    log_error("The image is too large for a DPX file (%dx%d)\n", width, height);
    return -1;
  }
  std::vector<uint8_t> h(image_offset, 0);
  uint8_t* p = h.data();
  // File information (768 bytes).
  memcpy(p, "SDPX", 4);
  be32(p + 4, image_offset);
  put_str(p + 8, 8, "V2.0");
  be32(p + 16, (uint32_t)(image_offset + image_bytes));
  be32(p + 20, 1);                 // a new frame
  be32(p + 24, 1664);              // generic header: file, image and orientation information
  be32(p + 28, 384);               // industry header: film and television
  be32(p + 32, user_size);
  {
    const size_t slash = path.find_last_of("/\\");
    put_str(p + 36, 100, slash == std::string::npos ? path : path.substr(slash + 1));
  }
  put_str(p + 136, 24, date_time(info.timestamp, ':'));   // the capture's time, as the TIFF's DateTime
  put_str(p + 160, 100, info.software);
  be32(p + 660, 0xffffffffu);      // not encrypted
  // Image information (640 bytes).
  uint8_t* im = p + 768;
  be16(im, 0);                     // left to right, top to bottom
  be16(im + 2, 1);                 // one image element
  be32(im + 4, (uint32_t)width);
  be32(im + 8, (uint32_t)height);
  uint8_t* el = im + 12;
  be32(el, 0);                     // unsigned
  be32(el + 4, (uint32_t)enc.offset);                                      // reference low: the film base
  be_float(el + 8, 0.0f);
  be32(el + 12, (uint32_t)enc.max_code());                                 // reference high
  be_float(el + 16, (float)(enc.max_code() - enc.offset) / (float)enc.scale);
  el[20] = 50;                     // descriptor: R, G, B
  // The transfer characteristic and the colorimetric specification: 1, printing density, for RP 180 (the Cineon
  // convention) and for Academy Printing Density (the code tools know; the element's description says ADX); 0, user
  // defined, for Status M, which is not a printing density.
  const uint8_t kind = measure == kDensityStatusM ? 0 : 1;
  el[21] = kind;
  el[22] = kind;
  el[23] = 10;                     // bits
  be16(el + 24, 1);                // packing: filled to 32-bit words, method A
  be16(el + 26, 0);                // no run-length encoding
  be32(el + 28, image_offset);
  be32(el + 32, 0);
  be32(el + 36, 0);
  put_str(el + 40, 32, measure == kDensityStatusM ? "Status M density above base"
                           : measure == kDensityRp180 ? "RP 180 printing density"
                                                      : "ADX10, SMPTE ST 2065-3");
  for (int k = 1; k < 8; ++k) memset(im + 12 + k * 72, 0, 72);
  // Orientation information (256 bytes): undefined but for the original size and the input device.
  uint8_t* orient = p + 1408;
  be32(orient, 0);
  be32(orient + 4, 0);
  memset(orient + 8, 0xff, 8);     // centre: undefined
  be32(orient + 16, (uint32_t)width);
  be32(orient + 20, (uint32_t)height);
  put_str(orient + 148, 32, info.model.empty() ? info.make : info.model);
  memset(orient + 212, 0xff, 8);   // border validity
  be32(orient + 220, 1);           // pixel aspect ratio 1:1
  be32(orient + 224, 1);
  // Film information (256 bytes) and television information (128 bytes): undefined.
  uint8_t* film = p + 1664;
  memset(film + 48, 0xff, 20);     // frame position, sequence length, held count, frame rate, shutter angle
  uint8_t* tv = p + 1920;
  memset(tv, 0xff, 52);
  // User-defined data.
  put_str(p + kHeader, kUserId, "negicc.density/1");
  memcpy(p + kHeader + kUserId, description.data(), description.size());

  FILE* fp = fopen(path.c_str(), "wb");
  if (!fp) {
    log_error("Cannot open %s for writing\n", path.c_str());
    return -1;
  }
  bool ok = fwrite(h.data(), 1, h.size(), fp) == h.size();
  std::vector<uint8_t> row((size_t)width * 4);
  for (int j = 0; ok && j < height; ++j) {
    const uint16_t* src = pixels + (size_t)j * width * channels;
    for (int i = 0; i < width; ++i) {
      const uint32_t r = src[(size_t)i * channels] & 0x3ffu, g = src[(size_t)i * channels + 1] & 0x3ffu,
                     b = src[(size_t)i * channels + 2] & 0x3ffu;
      be32(&row[(size_t)i * 4], (r << 22) | (g << 12) | (b << 2));
    }
    ok = fwrite(row.data(), 4, (size_t)width, fp) == (size_t)width;
  }
  ok = (fclose(fp) == 0) && ok;
  if (!ok) log_error("Failed to write %s\n", path.c_str());
  return ok ? 0 : -1;
}


// ---- The print of a density file (np_print.h) ----------------------------------------------------------------------

namespace {

bool read_vector(const nlohmann::json& j, std::vector<double>& out) {
  if (!j.is_array()) return false;
  out.clear();
  for (const auto& v : j) {
    if (!v.is_number()) return false;
    out.push_back(v.get<double>());
    if (!std::isfinite(out.back())) return false;
  }
  return true;
}

// numpy.interp: linear between the points, the end values outside them.
double interp(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
  if (x <= xs.front()) return ys.front();
  if (x >= xs.back()) return ys.back();
  const size_t hi = std::upper_bound(xs.begin(), xs.end(), x) - xs.begin();
  const double x0 = xs[hi - 1], x1 = xs[hi];
  return ys[hi - 1] + (ys[hi] - ys[hi - 1]) * (x - x0) / (x1 - x0);
}

void amounts_to_xyz(const PrintMaterial& m, const double amount[3], double xyz[3]) {
  xyz[0] = xyz[1] = xyz[2] = 0.0;
  const size_t n = m.base.size();
  for (size_t w = 0; w < n; ++w) {
    const double t =
        std::pow(10.0, -(amount[0] * m.dye[0][w] + amount[1] * m.dye[1][w] + amount[2] * m.dye[2][w] + m.base[w]));
    xyz[0] += t * m.observer[0][w];
    xyz[1] += t * m.observer[1][w];
    xyz[2] += t * m.observer[2][w];
  }
}

}  // namespace

namespace {

// A column of spektrafilm data (numbers and nulls) on its own wavelengths w, resampled to 380-780 nm at 1 nm. Nulls
// before the first and after the last number take the nearest number (hold) or `missing`; a null between numbers is
// an error (density_matrices.py hold_gaps).
bool resample_column(const nlohmann::json& rows, int column, const std::vector<double>& w, bool hold, double missing,
                     bool as_power_of_ten, std::vector<double>& out) {
  if (!rows.is_array() || rows.size() != w.size()) return false;
  std::vector<double> v(w.size(), std::nan(""));
  for (size_t i = 0; i < w.size(); ++i) {
    const nlohmann::json& cell = column < 0 ? rows[i] : (rows[i].is_array() && (int)rows[i].size() > column
                                                             ? rows[i][column] : nlohmann::json());
    if (cell.is_number() && std::isfinite(cell.get<double>())) v[i] = cell.get<double>();
  }
  size_t first = 0, last = w.size();
  while (first < w.size() && std::isnan(v[first])) ++first;
  while (last > first && std::isnan(v[last - 1])) --last;
  if (first == w.size()) return false;
  if (hold) {
    for (size_t i = first; i < last; ++i) {
      if (std::isnan(v[i])) return false;
    }
    for (size_t i = 0; i < first; ++i) v[i] = v[first];
    for (size_t i = last; i < w.size(); ++i) v[i] = v[last - 1];
  }
  for (double& x : v) x = std::isnan(x) ? missing : as_power_of_ten ? std::pow(10.0, x) : x;
  out.resize(kCieSamples);
  for (int i = 0; i < kCieSamples; ++i) out[i] = interp(w, v, kCieFirstNm + i);
  return true;
}

void multiply3(const double a[9], const double b[9], double out[9]) {
  double r[9];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
  }
  std::copy(r, r + 9, out);
}

bool invert3(const double m[9], double out[9]) {
  const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                     m[2] * (m[3] * m[7] - m[4] * m[6]);
  if (!std::isfinite(det) || std::fabs(det) < 1e-12) return false;
  const double r[9] = {m[4] * m[8] - m[5] * m[7], m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
                       m[5] * m[6] - m[3] * m[8], m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[0] * m[5],
                       m[3] * m[7] - m[4] * m[6], m[1] * m[6] - m[0] * m[7], m[0] * m[4] - m[1] * m[3]};
  for (int k = 0; k < 9; ++k) out[k] = r[k] / det;
  return true;
}

// XYZ of the print for a log exposure of each layer.
void exposure_to_xyz(const PrintMaterial& m, const double log_e[3], double xyz[3]) {
  double amount[3];
  for (int c = 0; c < 3; ++c) amount[c] = interp(m.log_exposure, m.curves[c], log_e[c]);
  amounts_to_xyz(m, amount, xyz);
}

// Q: the 3x3 from a film's dye amounts to the density above the film base that response (rows R, G, B, 401 samples)
// reads, least squares with no offset on the film model's fixed sample colours.
bool film_to_print_density(const FilmModel& film, const std::vector<double> response[3], double q[9]) {
  const double kSpread = 0.15 * std::sqrt(1.5);   // each dye amount times exp(kSpread z), z = -1, 0, 1
  double sum[3] = {0, 0, 0}, base_density[3];
  for (int c = 0; c < 3; ++c) {
    double t = 0.0;
    for (int w = 0; w < kCieSamples; ++w) {
      sum[c] += response[c][w];
      t += std::pow(10.0, -film.base[w]) * response[c][w];
    }
    if (!(sum[c] > 0.0)) return false;
    base_density[c] = -std::log10(t / sum[c]);
  }
  double ata[9] = {0}, atd[9] = {0};            // sums of a a^T and of a d^T over the samples
  std::vector<double> transmittance(kCieSamples);
  for (size_t i = 0; i < film.log_exposure.size(); ++i) {
    if (film.log_exposure[i] < -1.5 || film.log_exposure[i] > 1.0) continue;
    for (int v = 0; v < 27; ++v) {
      const int z[3] = {v / 9 - 1, v / 3 % 3 - 1, v % 3 - 1};
      double a[3], d[3];
      for (int c = 0; c < 3; ++c) a[c] = std::max(0.0, film.neutral[c][i] * std::exp(kSpread * z[c]));
      for (int w = 0; w < kCieSamples; ++w) {
        transmittance[w] = std::pow(
            10.0, -(a[0] * film.dye[0][w] + a[1] * film.dye[1][w] + a[2] * film.dye[2][w] + film.base[w]));
      }
      for (int c = 0; c < 3; ++c) {
        double t = 0.0;
        for (int w = 0; w < kCieSamples; ++w) t += transmittance[w] * response[c][w];
        d[c] = -std::log10(t / sum[c]) - base_density[c];
      }
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          ata[r * 3 + c] += a[r] * a[c];
          atd[r * 3 + c] += a[r] * d[c];
        }
      }
    }
  }
  double inverse[9], solved[9];
  if (!invert3(ata, inverse)) return false;
  multiply3(inverse, atd, solved);              // solved[k][c]: the weight of amount k in density c
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) q[r * 3 + c] = solved[c * 3 + r];
  }
  return true;
}

}  // namespace

namespace {

// Whether a parsed JSON is a spektrafilm profile of a print material neg_process can print with: info.stage
// "printing", info.support paper or film (it picks the viewing light), log_exposure ascending with one row of
// density_curves per value, each layer's curve rising with exposure, and each layer's log_sensitivity peaking within 6
// (log10) of the most sensitive layer's. False with why in err. Only load_spektrafilm_print calls it, so one reader
// holds every rule.
bool check_spektrafilm_print(const nlohmann::json& j, std::string& err) {
  if (!j.is_object() || !j.contains("data") || !j["data"].is_object()) return err = "not a spektrafilm profile (no data)", false;
  if (!j.contains("info") || !j["info"].is_object()) return err = "a spektrafilm profile without info", false;
  const nlohmann::json& info = j["info"];
  // A print material: spektrafilm also has camera films (stage "filming"), which would compute without complaint.
  const std::string stage = info.contains("stage") && info["stage"].is_string() ? info["stage"].get<std::string>() : "";
  if (stage != "printing") {
    return err = stage.empty() ? "a spektrafilm profile that does not say it is a print material (info.stage)"
                               : "a spektrafilm profile of a " + stage + " stock, not of a print material", false;
  }
  // The viewing light comes from the support: D50 for a paper, D55 for a print film.
  const std::string support = info.contains("support") && info["support"].is_string() ? info["support"].get<std::string>() : "";
  if (support != "paper" && support != "film") {
    return err = "a spektrafilm print profile whose info.support is '" + support +
                 "', not paper or film: the viewing light is chosen by it", false;
  }
  // The exposure axis the curves are judged against: ascending, one value per row of density_curves.
  const nlohmann::json& d = j["data"];
  if (!d.contains("density_curves") || !d["density_curves"].is_array() || d["density_curves"].size() < 2) {
    return err = "a spektrafilm print profile without density_curves", false;
  }
  std::vector<double> log_exposure;
  if (!d.contains("log_exposure") || !read_vector(d["log_exposure"], log_exposure) ||
      log_exposure.size() != d["density_curves"].size()) {
    return err = "log_exposure of the spektrafilm print profile is missing or not one value per row of density_curves", false;
  }
  if (!std::is_sorted(log_exposure.begin(), log_exposure.end())) {
    return err = "log_exposure of the spektrafilm print profile is not in ascending order", false;
  }
  // Each layer's curve rises with exposure: by kMinCurveRise over the profile, and never more than kMaxCurveDip below
  // its highest value so far, which does not depend on how finely the curve is sampled. In spektrafilm's eight
  // published print profiles the largest step down is 0.00093 and the largest fall below the running maximum 0.0025,
  // where the print is at its minimum density.
  constexpr double kMinCurveRise = 0.5, kMaxCurveDip = 0.02;
  for (int c = 0; c < 3; ++c) {
    double first = 0.0, last = 0.0, highest = -INFINITY;
    for (size_t i = 0; i < d["density_curves"].size(); ++i) {
      const nlohmann::json& row = d["density_curves"][i];
      if (!row.is_array() || row.size() != 3 || !row[c].is_number() || !std::isfinite(row[c].get<double>())) {
        return err = "density_curves of the spektrafilm print profile has a missing value", false;
      }
      const double v = row[c].get<double>();
      if (i == 0) first = v;
      if (v < highest - kMaxCurveDip) {
        return err = "layer " + std::to_string(c) + "'s curve of the spektrafilm print profile falls with exposure", false;
      }
      highest = std::max(highest, v);
      last = v;
    }
    if (!(last - first > kMinCurveRise)) {
      return err = "layer " + std::to_string(c) + "'s curve of the spektrafilm print profile does not rise with exposure", false;
    }
  }
  // Each layer is sensitive somewhere: its peak log sensitivity within kMaxSensitivityGap (log10) of the most
  // sensitive layer's, so no layer of the print is blind to the printer's light. A row is read as the load reads it
  // (resample_column): the layer's column, whatever follows it.
  constexpr double kMaxSensitivityGap = 6.0;
  if (!d.contains("log_sensitivity") || !d["log_sensitivity"].is_array()) {
    return err = "a spektrafilm print profile without log_sensitivity", false;
  }
  double peak[3] = {-INFINITY, -INFINITY, -INFINITY};
  for (const nlohmann::json& row : d["log_sensitivity"]) {
    for (int c = 0; row.is_array() && c < (int)std::min<size_t>(3, row.size()); ++c) {
      if (row[c].is_number() && std::isfinite(row[c].get<double>())) peak[c] = std::max(peak[c], row[c].get<double>());
    }
  }
  const double best = std::max(peak[0], std::max(peak[1], peak[2]));
  for (int c = 0; c < 3; ++c) {
    if (!(peak[c] > best - kMaxSensitivityGap)) {
      return err = "layer " + std::to_string(c) + " of the spektrafilm print profile has no spectral sensitivity", false;
    }
  }
  return true;
}

// info.<key> when it is a string, else fallback: spektrafilm writes nulls in info (target_print is null in all eight
// published print profiles), and json::value throws on a key that is there with another type.
std::string info_string(const nlohmann::json& info, const char* key, const std::string& fallback) {
  return info.contains(key) && info[key].is_string() ? info[key].get<std::string>() : fallback;
}

}  // namespace

bool load_spektrafilm_print(const std::string& json_text, const std::string& id, const ProfileBundle& bundle,
                            PrintMaterial& out, std::string& err) {
  const nlohmann::json j = nlohmann::json::parse(json_text, nullptr, false);
  if (j.is_discarded()) return err = "not a spektrafilm profile (not JSON)", false;
  if (!check_spektrafilm_print(j, err)) return false;
  const nlohmann::json& d = j["data"];
  const nlohmann::json& info = j["info"];
  PrintMaterial m;
  m.id = info_string(info, "stock", id);
  m.name = info_string(info, "name", m.id);
  m.viewing_illuminant = info["support"] == "paper" ? "D50" : "D55";
  std::vector<double> w, sensitivity[3];
  bool ok = d.contains("wavelengths") && read_vector(d["wavelengths"], w) && w.size() >= 2 &&
            std::is_sorted(w.begin(), w.end()) && d.contains("log_sensitivity") && d.contains("channel_density") &&
            d.contains("base_density") && read_vector(d["log_exposure"], m.log_exposure) &&
            resample_column(d["base_density"], -1, w, true, 0.0, false, m.base);
  for (int c = 0; ok && c < 3; ++c) {
    // A layer's sensitivity where the profile has none is zero; its dye spectrum is held at the nearest value.
    ok = resample_column(d["log_sensitivity"], c, w, false, 0.0, true, sensitivity[c]) &&
         resample_column(d["channel_density"], c, w, true, 0.0, false, m.dye[c]);
    m.curves[c].resize(m.log_exposure.size());   // the check has read every value as a finite number
    for (size_t i = 0; i < m.log_exposure.size(); ++i) m.curves[c][i] = d["density_curves"][i][c].get<double>();
  }
  if (!ok) {
    err = "a field of the spektrafilm print profile is missing or malformed";
    return false;
  }
  // What the print's own density is read with: each layer's sensitivity under the printer lamp, a 3200 K blackbody.
  std::vector<double> response[3];
  for (int c = 0; c < 3; ++c) {
    response[c].resize(kCieSamples);
    for (int i = 0; i < kCieSamples; ++i) {
      const double metres = (kCieFirstNm + i) * 1e-9;
      const double lamp = 1.0 / (std::pow(metres, 5) * (std::exp(1.4388e-2 / (metres * 3200.0)) - 1.0));
      response[c][i] = sensitivity[c][i] * lamp;
    }
  }
  // The observer under the viewing light, Y of the light 1; the print's white is the light itself.
  const double* light = m.viewing_illuminant == "D50" ? kCieD50 : kCieD55;
  const double* cmf[3] = {kCieXBar, kCieYBar, kCieZBar};
  double light_y = 0.0;
  for (int i = 0; i < kCieSamples; ++i) light_y += light[i] * kCieYBar[i];
  for (int c = 0; c < 3; ++c) {
    m.observer[c].resize(kCieSamples);
    m.white[c] = 0.0;
    for (int i = 0; i < kCieSamples; ++i) {
      m.observer[c][i] = light[i] * cmf[c][i] / light_y;
      m.white[c] += m.observer[c][i];
    }
  }
  // XYZ under the viewing light to linear sRGB, adapted to D65 with CAT02.
  const double cat02[9] = {0.7328, 0.4296, -0.1624, -0.7036, 1.6975, 0.0061, 0.0030, 0.0136, 0.9834};
  const double xyz_to_srgb[9] = {3.2404542, -1.5371385, -0.4985314, -0.9692660, 1.8760108,
                                 0.0415560, 0.0556434,  -0.2040259, 1.0572252};
  const double d65[3] = {0.3127 / 0.3290, 1.0, (1.0 - 0.3127 - 0.3290) / 0.3290};
  double cat_inverse[9], scaled[9];
  invert3(cat02, cat_inverse);
  for (int r = 0; r < 3; ++r) {
    const double to = cat02[r * 3] * d65[0] + cat02[r * 3 + 1] * d65[1] + cat02[r * 3 + 2] * d65[2];
    const double from = cat02[r * 3] * m.white[0] + cat02[r * 3 + 1] * m.white[1] + cat02[r * 3 + 2] * m.white[2];
    for (int c = 0; c < 3; ++c) scaled[r * 3 + c] = cat02[r * 3 + c] * to / from;
  }
  multiply3(cat_inverse, scaled, scaled);
  multiply3(xyz_to_srgb, scaled, m.to_srgb);
  // aim: the exposure of each layer at which the print is an 18 % grey under its light. Newton's method on the three
  // exposures, with the Jacobian by differences.
  double e[3] = {0.0, 0.0, 0.0};
  for (int iteration = 0; iteration < 50; ++iteration) {
    double xyz[3], f[3], jac[9], inverse[9];
    exposure_to_xyz(m, e, xyz);
    double worst = 0.0;
    for (int c = 0; c < 3; ++c) {
      f[c] = xyz[c] - 0.18 * m.white[c];
      worst = std::max(worst, std::fabs(f[c]));
    }
    if (worst < 1e-13) break;
    for (int k = 0; k < 3; ++k) {
      double moved[3] = {e[0], e[1], e[2]}, xyz2[3];
      moved[k] += 1e-4;
      exposure_to_xyz(m, moved, xyz2);
      for (int c = 0; c < 3; ++c) jac[c * 3 + k] = (xyz2[c] - xyz[c]) / 1e-4;
    }
    if (!invert3(jac, inverse)) {
      err = "the print's curves do not reach an 18 % grey";
      return false;
    }
    for (int k = 0; k < 3; ++k) e[k] -= inverse[k * 3] * f[0] + inverse[k * 3 + 1] * f[1] + inverse[k * 3 + 2] * f[2];
  }
  double check[3];
  exposure_to_xyz(m, e, check);
  for (int c = 0; c < 3; ++c) {
    if (!(std::fabs(check[c] - 0.18 * m.white[c]) < 1e-6)) {
      err = "the print's curves do not reach an 18 % grey";
      return false;
    }
    m.aim[c] = e[c];
  }
  // The print matrix of each measure, for this bundle's film: Q inv(F), with any light_density_factors of the measure
  // undone on the light's channels (Q inv(F) + Q L diag(1 / alpha - 1) inv(L) inv(F) = Q L inv(M)) so that P M = Q L
  // for every measure and the bracket's uncalibrated APD slopes apply once.
  // py-ref: negicc_station/src/print_emulation.py (print_matrix)
  auto print_matrix = [](const double q[9], const DensityMatrix& dm, double p[9]) {
    double inv_f[9];
    if (!dm.present || !invert3(dm.film_to_measure, inv_f)) return false;
    multiply3(q, inv_f, p);
    const double* alpha = dm.light_density_factors;
    if (alpha[0] == 1.0 && alpha[1] == 1.0 && alpha[2] == 1.0) return true;
    double inv_l[9];
    if (!invert3(dm.light_to_film, inv_l)) return false;
    double ql[9], inv_lf[9], scaled[9], corr[9];
    multiply3(q, dm.light_to_film, ql);
    multiply3(inv_l, inv_f, inv_lf);
    for (int r = 0; r < 3; ++r) {
      const double d = 1.0 / alpha[r] - 1.0;
      for (int c = 0; c < 3; ++c) scaled[r * 3 + c] = d * inv_lf[r * 3 + c];
    }
    multiply3(ql, scaled, corr);
    for (int k = 0; k < 9; ++k) p[k] += corr[k];
    return true;
  };
  double q[9];
  if (bundle.film_model.present && film_to_print_density(bundle.film_model, response, q)) {
    for (int k = 0; k < kDensityMeasures; ++k) {
      PrintMaterial::ToPrint t;
      if (!print_matrix(q, bundle.density_matrix[k], t.m.data())) continue;
      t.film = bundle.film_model.name;
      m.to_print[k].push_back(t);
    }
  }
  out = std::move(m);
  return true;
}

std::string print_clf_nodes(const PrintMaterial& m, int curve_points, int cube) {
  char buf[256];
  std::string s;
  s.reserve((size_t)cube * cube * cube * 40 + (size_t)curve_points * 40 + 4096);
  // log E = aim - (D - grey)
  s += "  <Matrix inBitDepth=\"32f\" outBitDepth=\"32f\">\n    <Array dim=\"3 4\">\n";
  for (int i = 0; i < 3; ++i) {
    snprintf(buf, sizeof buf, "      %d %d %d %.10g\n", i == 0 ? -1 : 0, i == 1 ? -1 : 0, i == 2 ? -1 : 0,
             m.aim[i] + m.grey_density);
    s += buf;
  }
  s += "    </Array>\n  </Matrix>\n";
  const double lo = m.log_exposure.front(), hi = m.log_exposure.back();
  snprintf(buf, sizeof buf,
           "  <Range inBitDepth=\"32f\" outBitDepth=\"32f\">\n    <minInValue>%.10g</minInValue>\n"
           "    <maxInValue>%.10g</maxInValue>\n    <minOutValue>0</minOutValue>\n    <maxOutValue>1</maxOutValue>\n"
           "  </Range>\n",
           lo, hi);
  s += buf;
  snprintf(buf, sizeof buf, "  <LUT1D inBitDepth=\"32f\" outBitDepth=\"32f\" interpolation=\"linear\">\n"
                            "    <Array dim=\"%d 3\">\n", curve_points);
  s += buf;
  double top = 0.0;
  for (int c = 0; c < 3; ++c) top = std::max(top, *std::max_element(m.curves[c].begin(), m.curves[c].end()));
  top = std::ceil(top * 10.0) / 10.0;
  for (int i = 0; i < curve_points; ++i) {
    const double x = lo + (hi - lo) * i / (curve_points - 1);
    snprintf(buf, sizeof buf, "      %.6g %.6g %.6g\n", interp(m.log_exposure, m.curves[0], x),
             interp(m.log_exposure, m.curves[1], x), interp(m.log_exposure, m.curves[2], x));
    s += buf;
  }
  s += "    </Array>\n  </LUT1D>\n";
  snprintf(buf, sizeof buf, "  <Matrix inBitDepth=\"32f\" outBitDepth=\"32f\">\n    <Array dim=\"3 3\">\n"
                            "      %.10g 0 0\n      0 %.10g 0\n      0 0 %.10g\n    </Array>\n  </Matrix>\n",
           1.0 / top, 1.0 / top, 1.0 / top);
  s += buf;
  snprintf(buf, sizeof buf, "  <LUT3D inBitDepth=\"32f\" outBitDepth=\"32f\" interpolation=\"tetrahedral\">\n"
                            "    <Array dim=\"%d %d %d 3\">\n", cube, cube, cube);
  s += buf;
  // The cube: dye amounts 0..top on each axis, blue changing fastest, to linear sRGB, relative to the paper's white:
  // each of X, Y and Z scaled by the viewing light's over the paper's (the ICC's media-relative colorimetry), so the
  // paper renders at the display's white, as the print ICC's picture does (np_pipeline: the print's white).
  double paper[3], rel[3];
  print_paper_xyz(m, paper);
  for (int c = 0; c < 3; ++c) rel[c] = m.white[c] / paper[c];
  const size_t points = (size_t)cube * cube * cube;
  std::vector<double> rgb(points * 3);
  #pragma omp parallel for schedule(static)
  for (long long idx = 0; idx < (long long)points; ++idx) {
    const int b = (int)(idx % cube), g = (int)((idx / cube) % cube), r = (int)(idx / ((long long)cube * cube));
    const double amount[3] = {top * r / (cube - 1), top * g / (cube - 1), top * b / (cube - 1)};
    double xyz[3];
    amounts_to_xyz(m, amount, xyz);
    for (int c = 0; c < 3; ++c) xyz[c] *= rel[c];
    for (int c = 0; c < 3; ++c) {
      rgb[idx * 3 + c] = m.to_srgb[c * 3] * xyz[0] + m.to_srgb[c * 3 + 1] * xyz[1] + m.to_srgb[c * 3 + 2] * xyz[2];
    }
  }
  for (size_t idx = 0; idx < points; ++idx) {
    snprintf(buf, sizeof buf, "      %.6g %.6g %.6g\n", rgb[idx * 3], rgb[idx * 3 + 1], rgb[idx * 3 + 2]);
    s += buf;
  }
  s += "    </Array>\n  </LUT3D>\n";
  s += "  <Exponent inBitDepth=\"32f\" outBitDepth=\"32f\" style=\"monCurveRev\">\n"
       "    <ExponentParams exponent=\"2.4\" offset=\"0.055\" />\n  </Exponent>\n";
  return s;
}

bool frame_print(const DensityRecord& rec, const PrintMaterial& material, FramePrint& out, std::string& err) {
  if (!rec.bundle || !rec.matrix) return err = "no bundle", false;
  const ProfileBundle& bundle = *rec.bundle;
  // The film model the bundle's density matrices were fitted on names the material's matrix for this film.
  const nlohmann::json fit_record = nlohmann::json::parse(bundle.density_fit_json, nullptr, false);
  const std::string film = fit_record.is_object() ? fit_record.value("film_dye_spectra", std::string()) : std::string();
  const PrintMaterial::ToPrint* to_print = material.to_print_for((int)rec.measure, film);
  if (!to_print) return err = "no print matrix for this measure and this film", false;
  if (bundle.targets.empty()) return err = "the bundle has no brackets", false;
  FramePrint p;
  p.id = material.id;
  p.name = material.name;
  p.grey_density = material.grey_density;
  for (int k = 0; k < 9; ++k) p.to_print[k] = to_print->m[k];
  p.to_print_for = to_print->film;
  // The frame's bracket and exposure: the search's, else the middle bracket at its own exposure. A material with its
  // own print search (a density run with --print, on either light) takes that search's A, E and gains (below).
  p.searched = rec.has_balance && rec.balance_e > 0.0;
  p.bracket = p.searched ? rec.balance_target : bundle.targets[bundle.targets.size() / 2].name;
  p.exposure = p.searched ? rec.balance_e : 1.0;
  const auto own = rec.print_search.find(material.id);
  const bool by_print = own != rec.print_search.end() && bundle.grey_balance.present &&
                        bundle.density_matrix[kDensityApd].present;
  if (by_print) {
    p.searched = true;
    p.print_searched = true;
    p.bracket = own->second.bracket;
    p.exposure = own->second.e;
    p.a = own->second.a;
    p.gain_g = own->second.g;
    p.gain_b = own->second.b;
    p.converged = own->second.converged;
    p.steps = own->second.steps;
    for (int c = 0; c < 3; ++c) p.last_move[c] = own->second.last_move[c];
    p.above_knee = own->second.above_knee;
    for (int c = 0; c < 3; ++c) p.past_knee[c] = own->second.past_knee[c], p.past_cut[c] = own->second.past_cut[c];
    p.stopped_by = own->second.stopped_by;
    p.a0 = own->second.a0;
    p.wish_l = own->second.wish_l;
    for (int c = 0; c < 3; ++c) {
      p.above_white_pct[c] = own->second.above_white_pct[c];
      p.at_white_pct[c] = own->second.at_white_pct[c];
    }
    p.knobs = own->second.knobs;
    p.knobs_limited = own->second.knobs_limited;
    p.knob_base_a = own->second.knob_base_a;
    p.knob_base_g = own->second.knob_base_g;
    p.knob_base_b = own->second.knob_base_b;
    for (int c = 0; c < 3; ++c) p.knob_grey_lab[c] = own->second.knob_grey_lab[c];
    for (int c = 0; c < 3; ++c) p.knob_given[c] = own->second.knob_given[c];
  }
  const DensityBracket* bracket = nullptr;
  for (const DensityBracket& b : bundle.density_brackets) {
    if (b.name == p.bracket) bracket = &b;
  }
  if (!bracket) {
    return err = "the bundle has no density_matrices.brackets entry for " + p.bracket +
                 ": rebuild it with the profile-building tools", false;
  }
  // The exposure E multiplies the light's channels before the bracket's profile reads them: the film that the
  // profile reads as its mid grey is denser by log10(E) in every light channel, through the measure's matrix.
  // py-ref: negicc_station/src/print_emulation.py (frame_grey)
  // A bundle with the film's grey balance curve: the mid grey and the slopes are the curve's, as in its
  // print ICCs.
  const double* m = rec.matrix->m;
  const double shift = std::log10(p.exposure);
  double light[3] = {bracket->grey_light[0], bracket->grey_light[1], bracket->grey_light[2]};
  for (int c = 0; c < 3; ++c) p.slopes[c] = bracket->slopes[c];
  if (by_print) {
    // The print at its own search's mid grey A, as the picture conversion renders it: the curve's grey and tangent at
    // A, and the brackets' tilt at A (print_tilt_at: linear in A between the two brackets around it).
    curve_grey_at(bundle, p.a, light);
    curve_slopes_at(bundle, p.a, p.slopes);
    double t[3];
    print_tilt_at(bundle, p.a, t);
    for (int c = 0; c < 3; ++c) p.slopes[c] *= 1.0 + t[c];
    p.from_curve = true;
  } else if (bundle.grey_balance.present && bundle.density_matrix[kDensityApd].present) {
    curve_grey(bundle, *bracket, light);
    curve_slopes(bundle, *bracket, p.slopes);
    // The bracket's tilt relative to the curve, as its print ICC's curves take it: (G_c - A_mid)(1 + t_c) has the
    // slope (1 + t_c) times the curve's at the mid grey.
    for (int c = 0; c < 3; ++c) p.slopes[c] *= 1.0 + bracket->tilt[c];
    p.from_curve = true;
  } else if (!bracket->has_slopes) {
    return err = "the bundle has neither grey_balance_curves nor slopes for " + p.bracket +
                 ": rebuild it with the profile-building tools", false;
  }
  for (int c = 0; c < 3; ++c) {
    p.grey[c] = 0.0;
    for (int k = 0; k < 3; ++k) p.grey[c] += m[c * 3 + k] * (light[k] + shift);
  }
  out = p;
  return true;
}

// The density offsets bal the print CLF applies for the frame's colour balance, level included (the zero-sum offsets
// plus the mean taken out of them): from the material's own print search's gains when it has one, else from the
// picture conversion's search. False when there is no balance (bal is then zero).
static bool print_balance(const DensityRecord& rec, const FramePrint& print, double bal[3]) {
  DensityRecord own = rec;   // the material's own print search: its gains in place of the picture conversion's
  if (print.print_searched) {
    own.has_balance = true;
    own.balance_e = print.exposure;
    own.balance_g = print.gain_g;
    own.balance_b = print.gain_b;
  }
  double mean = 0.0;
  const bool balanced = density_balance_offset(own, bal, &mean);
  for (int c = 0; c < 3; ++c) bal[c] += mean;
  return balanced;
}

std::string frame_print_clf(const DensityRecord& rec, const FramePrint& print, const std::string& nodes) {
  // x: the file's code / max code.  D_j = (x_j * max - offset) / (scale * k_j);  with the balance bal and the frame's
  // mid grey,  pd_i = grey_density + s_i * sum_j P_ij (D_j + bal_j - grey_j).
  //
  // bal is the whole of what the search's gains g and b do to the density, level included (the zero-sum offsets and
  // the mean taken out of them). The conversion renders as the chart's mid grey the film that is denser than it by
  // log10 E, log10 E g and log10 E b in the light's channels; with grey carrying log10 E and bal the rest, that film
  // comes out at grey_density in all three channels.
  const DensityEncoding& e = rec.encoding;
  const double max_code = (double)e.max_code();
  double bal[3];
  const bool balanced = print_balance(rec, print, bal);
  double a[9], b[3];
  for (int i = 0; i < 3; ++i) {
    double shift = 0.0;
    for (int j = 0; j < 3; ++j) {
      const double p = print.slopes[i] * print.to_print[i * 3 + j];
      a[i * 3 + j] = p * max_code / (e.scale * e.k[j]);
      shift += p * (bal[j] - print.grey[j] - e.offset / (e.scale * e.k[j]));
    }
    b[i] = print.grey_density + shift;
  }
  // Names come from a bundle and a print material file: escaped, and the text built without a fixed buffer.
  auto xml = [](const std::string& text) {
    std::string out;
    for (char ch : text) {
      switch (ch) {
        case '&': out += "&amp;"; break;
        case '<': out += "&lt;"; break;
        case '>': out += "&gt;"; break;
        case '"': out += "&quot;"; break;
        default: out += ch;
      }
    }
    return out;
  };
  auto num = [](const char* format, double a, double b, double c) {
    char text[96];
    snprintf(text, sizeof text, format, a, b, c);
    return std::string(text);
  };
  char buf[256];
  const std::string print_name = xml(print.name);
  std::string s = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  s += "<ProcessList id=\"negicc.print." + xml(print.id) + "\" name=\"" + print_name + "\" compCLFversion=\"3.0\">\n";
  snprintf(buf, sizeof buf, "%s density in %d bits, code = %d + %d k D", density_measure_name(rec.measure), e.bits,
           e.offset, e.scale);
  s += "  <Description>This frame (" + xml(rec.bundle_key) + ", " + buf + ") printed on " + print_name +
       ", as sRGB. The first Matrix takes the file's code values to the print's printing density with the frame's "
       "colour balance, density offsets " + num("%+.4f %+.4f %+.4f", bal[0], bal[1], bal[2]) +
       (balanced ? "" : " (no balance was found: none applied)") + ", and slopes " +
       num("%.4f %.4f %.4f", print.slopes[0], print.slopes[1], print.slopes[2]) +
       " about the frame's mid grey, density " + num("%.4f %.4f %.4f", print.grey[0], print.grey[1], print.grey[2]) +
       ". " +
       (print.print_searched
            ? "The mid grey is the film's grey balance curve's grey at the print's own mid grey A " +
                  num("%.4f", print.a, 0.0, 0.0) +
                  ", which this material's own print search found on this frame, with that search's exposure and "
                  "gains (the nearest bracket: " + xml(print.bracket) + "); balanced, it prints as the print's mid grey. "
                  "The slopes are that curve's contrast there times one plus the brackets' tilt at A (grey_relative.b, "
                  "linear in A between the brackets): they give the film's grey scale equal contrast in the three "
                  "channels"
            : print.from_curve
                  ? "The mid grey is the film's grey balance curve's grey at the green density of the chart's mid grey on "
                    "the bracket " + xml(print.bracket) +
                        (print.searched ? " and at the exposure the picture conversion's search gave this frame"
                                        : " (no search: the middle bracket)") +
                        "; balanced, it prints as the print's mid grey. The slopes are that curve's contrast there "
                        "times one plus the bracket's tilt (grey_relative.b): they give the film's grey scale equal "
                        "contrast in the three channels"
                  : "The mid grey is the chart's on the bracket " + xml(print.bracket) +
                        (print.searched ? " and at the exposure the picture conversion's search gave this frame"
                                        : " (no search: the middle bracket)") +
                        "; balanced, it prints as the print's mid grey. The slopes are the bracket's in the film's "
                        "profile: they give the film's grey scale equal contrast in the three channels") +
       ". The density file holds none of this. Add a constant to the three offsets of the first Matrix to print "
       "lighter (+) or darker (-), 0.30 a stop.</Description>\n";
  s += "  <Description>The 1D and 3D tables are derived by neg_process from the print profile of this material in "
       "spektrafilm (Andrea Volpato, https://github.com/andreavolpato/spektrafilm): its curve per layer resampled, "
       "and its dye spectra computed into colours. CC BY-SA 4.0, "
       "https://github.com/andreavolpato/spektrafilm/blob/main/SPEKTRAFILM_LICENSE.txt. The matrices are computed "
       "by neg_process and are not spektrafilm's data. A model, not compared with a real print.</Description>\n";
  s += "  <InputDescriptor>density code values</InputDescriptor>\n  <OutputDescriptor>sRGB</OutputDescriptor>\n";
  s += "  <Matrix inBitDepth=\"32f\" outBitDepth=\"32f\">\n    <Array dim=\"3 4\">\n";
  for (int i = 0; i < 3; ++i) {
    snprintf(buf, sizeof buf, "      %.10g %.10g %.10g %.10g\n", a[i * 3], a[i * 3 + 1], a[i * 3 + 2], b[i]);
    s += buf;
  }
  s += "    </Array>\n  </Matrix>\n";
  s += nodes;
  s += "</ProcessList>\n";
  return s;
}

bool is_print_file(const std::string& name) {
  return name.find_first_of("/\\") != std::string::npos ||
         (name.size() > 5 && name.substr(name.size() - 5) == ".json");
}

std::vector<PrintMaterial> builtin_print_materials(const ProfileBundle* bundle) {
  static const ProfileBundle no_bundle;
  const std::string folder = "3rd_party/spektrafilm/profiles/";
  std::vector<PrintMaterial> out;
  for (const EmbeddedFile& file : list_embedded_files()) {
    const std::string name = file.name;
    if (name.rfind(folder, 0) != 0 || name.size() < folder.size() + 6 || name.substr(name.size() - 5) != ".json") {
      continue;
    }
    PrintMaterial material;
    std::string err;
    if (!load_spektrafilm_print(std::string((const char*)file.data, file.size),
                                name.substr(folder.size(), name.size() - folder.size() - 5),
                                bundle ? *bundle : no_bundle, material, err)) {
      log_warn("Print profile %s cannot be read: %s\n", name.c_str(), err.c_str());
      continue;
    }
    out.push_back(std::move(material));
  }
  return out;
}

bool read_print_file(const std::string& path, std::string& text, std::string& stem, std::string& err) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return err = "cannot be read", false;
  std::stringstream buffer;
  buffer << file.rdbuf();
  if (file.bad()) return err = "cannot be read", false;
  text = buffer.str();
  const size_t slash = path.find_last_of("/\\");
  stem = slash == std::string::npos ? path : path.substr(slash + 1);
  if (stem.size() > 5 && stem.substr(stem.size() - 5) == ".json") stem.resize(stem.size() - 5);
  return true;
}

DensityPrints density_prints(const DensityRecord& rec, const std::string& output,
                             const std::vector<std::string>& print_ids) {
  DensityPrints out;
  if (print_ids.empty()) return out;
  // The materials asked for: built-in ones by id (or all of them), and spektrafilm print profiles by path, whose
  // matrices are computed here for the bundle's film.
  const bool all = std::find(print_ids.begin(), print_ids.end(), "all") != print_ids.end();
  std::vector<PrintMaterial> materials;
  for (PrintMaterial& material : builtin_print_materials(rec.bundle)) {
    if (all || std::find(print_ids.begin(), print_ids.end(), material.id) != print_ids.end()) {
      materials.push_back(std::move(material));
    }
  }
  for (const std::string& id : print_ids) {
    if (!is_print_file(id)) continue;
    std::string text, stem, err;
    PrintMaterial material;
    if (!read_print_file(id, text, stem, err) || !rec.bundle ||
        !load_spektrafilm_print(text, stem, *rec.bundle, material, err)) {
      out.error = "no print CLF for " + id + ": " + (err.empty() ? "no profile bundle" : err);
      return out;
    }
    materials.push_back(std::move(material));
  }
  if (!materials.empty() && (!rec.bundle || !rec.bundle->film_model.present)) {
    out.error = "no print CLF: the profile bundle has no film_model, which a print's matrix is computed from: rebuild "
                "it with the profile-building tools";
    return out;
  }
  for (PrintMaterial& material : materials) {
    FramePrint print;
    std::string err;
    // Two materials of one id would share a CLF name: the first is kept.
    const std::string path = output + "." + material.id + ".clf";
    if (std::find(out.paths.begin(), out.paths.end(), path) != out.paths.end()) {
      log_warn("--print names the print material %s twice: its CLF is written once, for the first (built-in "
               "materials before files)\n", material.id.c_str());
      continue;
    }
    if (!frame_print(rec, material, print, err)) {
      out.error = "no print CLF for " + material.id + ": " + err;
      return out;
    }
    out.paths.push_back(output + "." + material.id + ".clf");
    out.materials.push_back(std::move(material));
    out.frames.push_back(print);
  }
  if (out.frames.empty()) return out;
  using nlohmann::ordered_json;
  ordered_json j;
  const std::vector<FramePrint>* prints = &out.frames;
  {
    // How the frame prints on each print material: computed for this frame, applied by the CLF beside the file.
    ordered_json materials = ordered_json::object();
    for (const FramePrint& p : *prints) {
      const double* t = p.to_print;
      ordered_json one = {
          {"print", p.name},
          {"to_print_density", {{t[0], t[1], t[2]}, {t[3], t[4], t[5]}, {t[6], t[7], t[8]}}},
          {"to_print_density_for", p.to_print_for},
          {"bracket", p.bracket},
          {"bracket_from", p.print_searched ? "the print's own search on this frame, on this material (the bracket nearest "
                                              "its mid grey A, named only: the tilt is interpolated at A)"
                           : p.searched ? "the picture conversion's search on this frame" : "the middle bracket (no search)"},
          {"exposure", p.exposure},
          {"frame_grey_density", {p.grey[0], p.grey[1], p.grey[2]}},
          {"slopes", {p.slopes[0], p.slopes[1], p.slopes[2]}},
          {"grey_and_slopes_from", p.print_searched ? "the film's grey balance curve at the print's own mid grey A, "
                                                      "the slopes times one plus the brackets' tilt at A"
                                   : p.from_curve ? "the film's grey balance curve at the bracket's mid grey, the slopes "
                                                    "times one plus the bracket's tilt"
                                                  : "the bracket's entry in the profile set"}};
      if (p.print_searched) {
        one["mid_grey_a"] = p.a;
        one["gains"] = {p.gain_g, p.gain_b};
        double bal[3];
        print_balance(rec, p, bal);
        one["balance"] = {bal[0], bal[1], bal[2]};   // the formula's balance for this material, as its CLF applies it
        one["search"] = {{"a0", p.a0}, {"wish_L", p.wish_l},
                         {"converged", p.converged}, {"stopped_by", p.stopped_by}, {"steps", p.steps},
                         {"last_move", {p.last_move[0], p.last_move[1], p.last_move[2]}},
                         {"above_knee_pct", 100.0 * p.above_knee},
                         {"past_knee_pct", {100.0 * p.past_knee[0], 100.0 * p.past_knee[1], 100.0 * p.past_knee[2]}},
                         {"past_cut_pct", {100.0 * p.past_cut[0], 100.0 * p.past_cut[1], 100.0 * p.past_cut[2]}},
                         {"above_white_pct", {p.above_white_pct[0], p.above_white_pct[1], p.above_white_pct[2]}},
                         {"at_white_pct", {p.at_white_pct[0], p.at_white_pct[1], p.at_white_pct[2]}}};
        if (p.knobs) {
          one["knobs"] = {{"what", "--brightness, --tint and --warmth on this print: mid_grey_a and gains are the "
                                   "moved ones (A, g and b solved on the print's profile, E held), from the search's"},
                          {"brightness", p.knob_given[0]}, {"tint", p.knob_given[1]}, {"warmth", p.knob_given[2]},
                          {"search", {{"a", p.knob_base_a}, {"g", p.knob_base_g}, {"b", p.knob_base_b}}},
                          {"grey_lab", {p.knob_grey_lab[0], p.knob_grey_lab[1], p.knob_grey_lab[2]}},
                          {"limited", p.knobs_limited}};
        }
      }
      materials[p.id] = one;
    }
    j["print"] = {
        {"what", "how this frame prints on each print material: the frame's mid grey lands on the print's mid grey, "
                 "and the slopes give the film's grey scale equal contrast in the three channels. With the print's own "
                 "search on a material, under either light, the mid grey A, the exposure and the gains are that "
                 "search's, and the mid grey and the slopes are the film's grey balance curve's at A "
                 "(density_matrices.grey_balance_curves). Otherwise the bracket and the exposure are the picture "
                 "conversion's search's; the mid grey and the slopes are the film's grey balance curve's at the "
                 "bracket's mid grey, or the bracket's in the profile set (density_matrices.brackets) "
                 "(grey_and_slopes_from). The curve's slopes are multiplied by one plus the brackets' tilt "
                 "(grey_relative.b) at the mid grey"},
        {"formula", "print density = 0.70 + slopes * (to_print_density @ (D + balance - frame_grey_density)), "
                    "balance = the material's balance where it has its own print search (materials.<id>.balance, "
                    "from its gains), else suggested_balance.density_offset + density_offset_mean_removed"},
        {"applied", "no: the densities are as read. The CLF beside the file for each print material "
                    "(<output>.<print>.clf) applies the balance, the exposure and the slopes; apply them once, there "
                    "or by hand, not both"},
        {"materials", materials}};
  }
  out.metadata_json = j["print"].dump();
  return out;
}

void remove_print_clfs(const DensityPrints& prints) {
  for (const std::string& path : prints.paths) remove(path.c_str());
}

int write_print_clfs(const DensityRecord& rec, const DensityPrints& prints) {
  for (size_t k = 0; k < prints.frames.size(); ++k) {
    const FramePrint& print = prints.frames[k];
    const std::string clf = frame_print_clf(rec, print, print_clf_nodes(prints.materials[k]));
    FILE* f = fopen(prints.paths[k].c_str(), "wb");
    const bool ok = f && fwrite(clf.data(), 1, clf.size(), f) == clf.size();
    if ((f && fclose(f) != 0) || !ok) {
      log_error("Cannot write '%s'\n", prints.paths[k].c_str());
      remove_print_clfs(prints);
      return 1;
    }
    log_info("Wrote '%s': this file as a print on %s, sRGB (bracket %s, exposure %.3f, slopes %.3f %.3f %.3f)\n",
             prints.paths[k].c_str(), print.name.c_str(), print.bracket.c_str(), print.exposure, print.slopes[0],
             print.slopes[1], print.slopes[2]);
  }
  return 0;
}

// ---- The print ICC --------------------------------------------------------------------------------------------------

namespace {

// The value of a table on start + step i at x, linear, the end values outside it (numpy.interp on a uniform grid).
double on_grid(const std::vector<double>& ys, double start, double step, double x) {
  const double t = (x - start) / step;
  if (t <= 0.0) return ys.front();
  if (t >= (double)(ys.size() - 1)) return ys.back();
  const size_t i = (size_t)t;
  return ys[i] + (t - (double)i) * (ys[i + 1] - ys[i]);
}

std::vector<double> grey_grid(const GreyBalanceCurves& g) {
  std::vector<double> x(g.apd[0].size());
  for (size_t i = 0; i < x.size(); ++i) x[i] = g.start + g.step * (double)i;
  return x;
}

// The print matrix for APD of a material, for the film of the bundle's density matrices; null when it has none.
const double* apd_print_matrix(const ProfileBundle& bundle, const PrintMaterial& m) {
  const nlohmann::json fit = nlohmann::json::parse(bundle.density_fit_json, nullptr, false);
  const std::string film = fit.is_object() ? fit.value("film_dye_spectra", std::string()) : std::string();
  const PrintMaterial::ToPrint* t = m.to_print_for(kDensityApd, film);
  return t ? t->m.data() : nullptr;
}

void put_u16(std::vector<uint8_t>& b, double v01) {
  const long q = std::lround(std::min(std::max(v01, 0.0), 1.0) * 65535.0);
  b.push_back((uint8_t)(q >> 8));
  b.push_back((uint8_t)(q & 255));
}
void put_u32(std::vector<uint8_t>& b, uint32_t v) {
  for (int s = 24; s >= 0; s -= 8) b.push_back((uint8_t)(v >> s));
}
void put_s15(std::vector<uint8_t>& b, double v) { put_u32(b, (uint32_t)(int32_t)std::lround(v * 65536.0)); }
uint32_t get_u32(const std::vector<uint8_t>& b, size_t o) {
  return (uint32_t)b[o] << 24 | (uint32_t)b[o + 1] << 16 | (uint32_t)b[o + 2] << 8 | b[o + 3];
}
// The offset of an ICC's A2B0 lut16 tag, or 0.
size_t a2b0_lut16(const std::vector<uint8_t>& icc) {
  if (icc.size() < 132) return 0;
  const uint32_t n = get_u32(icc, 128);
  for (uint32_t i = 0; i < n && 144 + 12 * i <= icc.size(); ++i) {
    if (std::memcmp(icc.data() + 132 + 12 * i, "A2B0", 4) != 0) continue;
    const size_t off = get_u32(icc, 136 + 12 * i);
    if (off + 52 > icc.size() || std::memcmp(icc.data() + off, "mft2", 4) != 0 || icc[off + 8] != 3) return 0;
    return off;
  }
  return 0;
}

constexpr double kD50[3] = {0.9642, 1.0, 0.8249};
constexpr double kPrintKnee = kKnee;  // the conversion's soft knee (profile_bundle.h), in double

}  // namespace

double bracket_mid_apd(const ProfileBundle& bundle, const DensityBracket& bracket) {
  const double* m = bundle.density_matrix[kDensityApd].m;
  return m[3] * bracket.grey_light[0] + m[4] * bracket.grey_light[1] + m[5] * bracket.grey_light[2];
}

void curve_grey_at(const ProfileBundle& bundle, double a, double light[3]) {
  const GreyBalanceCurves& g = bundle.grey_balance;
  const std::vector<double> x = grey_grid(g);
  for (int c = 0; c < 3; ++c) light[c] = interp(g.apd[c], x, a);
}

void curve_grey(const ProfileBundle& bundle, const DensityBracket& bracket, double light[3]) {
  curve_grey_at(bundle, bracket_mid_apd(bundle, bracket), light);
}

void print_tilt_at(const std::vector<double>& a_mid, const std::vector<std::array<double, 3>>& tilt, double a,
                   double t[3]) {
  for (int c = 0; c < 3; ++c) t[c] = 0.0;
  if (a_mid.empty() || a_mid.size() != tilt.size()) return;
  std::vector<size_t> o(a_mid.size());
  for (size_t i = 0; i < o.size(); ++i) o[i] = i;
  std::sort(o.begin(), o.end(), [&](size_t x, size_t y) { return a_mid[x] < a_mid[y]; });
  if (o.size() == 1) {   // one bracket: its tilt
    for (int c = 0; c < 3; ++c) t[c] = tilt[o[0]][c];
    return;
  }
  size_t k = 0;   // the segment o[k], o[k + 1]
  while (k + 2 < o.size() && a > a_mid[o[k + 1]]) ++k;
  const size_t i = o[k], j = o[k + 1];
  // Two brackets at the same A_mid: the first one's tilt (no division by zero).
  const double f = a_mid[j] > a_mid[i] ? std::clamp((a - a_mid[i]) / (a_mid[j] - a_mid[i]), 0.0, 1.0) : 0.0;
  for (int c = 0; c < 3; ++c) t[c] = tilt[i][c] + f * (tilt[j][c] - tilt[i][c]);
}

void print_tilt_at(const ProfileBundle& bundle, double a, double t[3]) {
  // Over the brackets of bundle.targets, as the picture's print (bracket_print_profiles' continuum).
  std::vector<double> am;
  std::vector<std::array<double, 3>> tl;
  for (const BundleTarget& target : bundle.targets)
    for (const DensityBracket& b : bundle.density_brackets) {
      if (b.name != target.name) continue;
      am.push_back(bracket_mid_apd(bundle, b));
      tl.push_back({b.tilt[0], b.tilt[1], b.tilt[2]});
      break;
    }
  print_tilt_at(am, tl, a, t);
}

void curve_slopes(const ProfileBundle& bundle, const DensityBracket& bracket, double slopes[3]) {
  curve_slopes_at(bundle, bracket_mid_apd(bundle, bracket), slopes);
}

void curve_slopes_at(const ProfileBundle& bundle, double a, double slopes[3]) {
  const GreyBalanceCurves& g = bundle.grey_balance;
  double n[3], inv[3], k[3] = {0, 0, 0};
  curve_grey_at(bundle, a, n);
  for (int c = 0; c < 3; ++c) {
    const double d = (on_grid(g.apd[c], g.start, g.step, n[c] + g.step) -
                      on_grid(g.apd[c], g.start, g.step, n[c] - g.step)) / (2.0 * g.step);
    inv[c] = 1.0 / d;
  }
  const double* m = bundle.density_matrix[kDensityApd].m;
  for (int c = 0; c < 3; ++c) k[c] = m[c * 3] * inv[0] + m[c * 3 + 1] * inv[1] + m[c * 3 + 2] * inv[2];
  slopes[0] = k[1] / k[0];
  slopes[1] = 1.0;
  slopes[2] = k[1] / k[2];
}

PrintCurves paper_curve(const PrintMaterial& m, const double p_apd[9]) {
  PrintCurves out;
  for (int c = 0; c < 3; ++c) {
    const double eta = p_apd[c * 3] + p_apd[c * 3 + 1] + p_apd[c * 3 + 2];
    out[c].resize(kPaperAlphaPoints);
    double run = INFINITY;
    for (int i = 0; i < kPaperAlphaPoints; ++i) {
      const double alpha = kPaperAlphaStart + kPaperAlphaStep * i;
      run = std::min(run, std::max(0.0, interp(m.log_exposure, m.curves[c], m.aim[c] - eta * alpha)));
      out[c][i] = run;
    }
    const double top = out[c][0];
    for (int i = 0; i < kPaperAlphaPoints; ++i)
      out[c][i] = 0.75 * (out[c][i] / top) + 0.25 * (1.0 - (double)i / (kPaperAlphaPoints - 1));
  }
  return out;
}

PrintCurves bracket_print_curves(const GreyBalanceCurves& g, const PrintCurves& paper, double a_mid, double n0,
                                 const double tilt[3], const double gains[3]) {
  const double d = 1.0 - kPrintKnee;
  PrintCurves out;
  for (int c = 0; c < 3; ++c) {
    out[c].resize(kPrintIccPoints);
    double run = -INFINITY;
    for (int i = 0; i < kPrintIccPoints; ++i) {
      const double v = (double)i / (kPrintIccPoints - 1);
      const double u = v <= kPrintKnee ? v
                                       : kPrintKnee + d * std::atanh(std::min(std::max((v - kPrintKnee) / d, 0.0), 0.998));
      const double dl = -std::log10(std::max(u * (gains ? gains[c] : 1.0), 1e-7) / n0);
      const double alpha = (on_grid(g.apd[c], g.start, g.step, dl) - a_mid) * (1.0 + tilt[c]);
      run = std::max(run, on_grid(paper[c], kPaperAlphaStart, kPaperAlphaStep, alpha));
      out[c][i] = run + 1e-9 * i;
    }
  }
  return out;
}

std::vector<double> print_table(const PrintMaterial& m, const double p_apd[9], const double m_apd[9],
                                const PrintCurves& paper) {
  // Q L = P M and w = M^-1 (1, 1, 1).
  double ql[9], inv[9], w[3];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      ql[r * 3 + c] = p_apd[r * 3] * m_apd[c] + p_apd[r * 3 + 1] * m_apd[3 + c] + p_apd[r * 3 + 2] * m_apd[6 + c];
  {
    const double* a = m_apd;
    const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                       a[2] * (a[3] * a[7] - a[4] * a[6]);
    inv[0] = (a[4] * a[8] - a[5] * a[7]) / det;  inv[1] = (a[2] * a[7] - a[1] * a[8]) / det;
    inv[2] = (a[1] * a[5] - a[2] * a[4]) / det;  inv[3] = (a[5] * a[6] - a[3] * a[8]) / det;
    inv[4] = (a[0] * a[8] - a[2] * a[6]) / det;  inv[5] = (a[2] * a[3] - a[0] * a[5]) / det;
    inv[6] = (a[3] * a[7] - a[4] * a[6]) / det;  inv[7] = (a[1] * a[6] - a[0] * a[7]) / det;
    inv[8] = (a[0] * a[4] - a[1] * a[3]) / det;
  }
  for (int r = 0; r < 3; ++r) w[r] = inv[r * 3] + inv[r * 3 + 1] + inv[r * 3 + 2];
  // The PCS: the identity for a paper (D50), Bradford from the light's white to D50 for a print film.
  double pcs[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  if (m.viewing_illuminant != "D50") {
    static const double B[9] = {0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296};
    double src[3] = {m.white[0] / m.white[1], 1.0, m.white[2] / m.white[1]}, gain[3];
    for (int r = 0; r < 3; ++r) {
      const double to = B[r * 3] * kD50[0] + B[r * 3 + 1] * kD50[1] + B[r * 3 + 2] * kD50[2];
      const double from = B[r * 3] * src[0] + B[r * 3 + 1] * src[1] + B[r * 3 + 2] * src[2];
      gain[r] = to / from;
    }
    // inv(B) diag(gain) B, with inv(B) computed exactly rather than taken from the rounded table above.
    double bi[9];
    {
      const double* a = B;
      const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                         a[2] * (a[3] * a[7] - a[4] * a[6]);
      bi[0] = (a[4] * a[8] - a[5] * a[7]) / det;  bi[1] = (a[2] * a[7] - a[1] * a[8]) / det;
      bi[2] = (a[1] * a[5] - a[2] * a[4]) / det;  bi[3] = (a[5] * a[6] - a[3] * a[8]) / det;
      bi[4] = (a[0] * a[8] - a[2] * a[6]) / det;  bi[5] = (a[2] * a[3] - a[0] * a[5]) / det;
      bi[6] = (a[3] * a[7] - a[4] * a[6]) / det;  bi[7] = (a[1] * a[6] - a[0] * a[7]) / det;
      bi[8] = (a[0] * a[4] - a[1] * a[3]) / det;
    }
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) {
        double v = 0.0;
        for (int k = 0; k < 3; ++k) v += bi[r * 3 + k] * gain[k] * B[k * 3 + c];
        pcs[r * 3 + c] = v;
      }
  }
  // The nodes' alpha: Paper_c inverted (it is decreasing, so read reversed).
  std::vector<double> alpha_grid(kPaperAlphaPoints), node_alpha[3];
  for (int i = 0; i < kPaperAlphaPoints; ++i) alpha_grid[kPaperAlphaPoints - 1 - i] = kPaperAlphaStart + kPaperAlphaStep * i;
  for (int c = 0; c < 3; ++c) {
    const std::vector<double> rev(paper[c].rbegin(), paper[c].rend());
    for (int i = 0; i < kPrintIccCube; ++i)
      node_alpha[c].push_back(interp(rev, alpha_grid, (double)i / (kPrintIccCube - 1)));
  }
  std::vector<double> out((size_t)kPrintIccCube * kPrintIccCube * kPrintIccCube * 3);
  size_t o = 0;
  for (int i = 0; i < kPrintIccCube; ++i)
    for (int j = 0; j < kPrintIccCube; ++j)
      for (int k = 0; k < kPrintIccCube; ++k) {
        const double wa[3] = {node_alpha[0][i] * w[0], node_alpha[1][j] * w[1], node_alpha[2][k] * w[2]};
        double amount[3], xyz[3];
        for (int c = 0; c < 3; ++c) {
          const double pd = m.grey_density + ql[c * 3] * wa[0] + ql[c * 3 + 1] * wa[1] + ql[c * 3 + 2] * wa[2];
          amount[c] = interp(m.log_exposure, m.curves[c], m.aim[c] - (pd - m.grey_density));
        }
        amounts_to_xyz(m, amount, xyz);
        for (int c = 0; c < 3; ++c) out[o++] = pcs[c * 3] * xyz[0] + pcs[c * 3 + 1] * xyz[1] + pcs[c * 3 + 2] * xyz[2];
      }
  return out;
}

std::array<int, 6> bundle_date(const ProfileBundle& bundle) {
  // The bundle's source file name (export.source_file) ends in _YYYYMMDD_HHMMSS (profile_<Key>_<date>_<time>.json),
  // the date the bundle's own print ICCs carry; 2026-01-01 00:00:00 when it does not.
  std::string stem = bundle.source_file;
  const size_t slash = stem.find_last_of("/\\");
  if (slash != std::string::npos) stem = stem.substr(slash + 1);
  if (stem.size() > 5 && stem.substr(stem.size() - 5) == ".json") stem.resize(stem.size() - 5);
  std::array<int, 6> d = {2026, 1, 1, 0, 0, 0};
  if (stem.size() >= 16 && stem[stem.size() - 16] == '_' && stem[stem.size() - 7] == '_') {
    const std::string t = stem.substr(stem.size() - 15, 8) + stem.substr(stem.size() - 6);
    if (std::all_of(t.begin(), t.end(), [](char c) { return c >= '0' && c <= '9'; })) {
      d = {std::stoi(t.substr(0, 4)), std::stoi(t.substr(4, 2)), std::stoi(t.substr(6, 2)), std::stoi(t.substr(8, 2)),
           std::stoi(t.substr(10, 2)), std::stoi(t.substr(12, 2))};
    }
  }
  return d;
}

// A textDescriptionType tag (ICC v2) holding description as its ASCII text, with empty Unicode and ScriptCode parts.
static std::vector<uint8_t> icc_desc_tag(const std::string& description) {
  // The ASCII part holds 7-bit text (ICC.1:2001-04 6.5.17): each character outside it (a UTF-8 material name's) is
  // written as one '?', as the station's write_icc writes it (Python's encode("ascii", "replace")).
  std::string ascii;
  for (const char ch : description) {
    const unsigned char u = (unsigned char)ch;
    if (u < 0x80) ascii += ch;
    else if (u >= 0xc0) ascii += '?';   // a character's first byte; its continuation bytes (0x80 to 0xbf) are dropped
  }
  std::vector<uint8_t> desc = {'d', 'e', 's', 'c', 0, 0, 0, 0};
  put_u32(desc, (uint32_t)ascii.size() + 1);
  desc.insert(desc.end(), ascii.begin(), ascii.end());
  desc.insert(desc.end(), 1 + 8 + 3 + 67, 0);
  return desc;
}

// A profile from its 128-byte header and its tags in order (each tag's data on a 4-byte boundary); the header's
// size field is set.
static std::vector<uint8_t> icc_assemble(std::vector<uint8_t> icc,
                                         const std::vector<std::pair<std::string, std::vector<uint8_t>>>& tags) {
  std::vector<uint8_t> table_bytes, body;
  const size_t start = 128 + 4 + 12 * tags.size();
  for (const auto& t : tags) {
    table_bytes.insert(table_bytes.end(), t.first.begin(), t.first.begin() + 4);
    put_u32(table_bytes, (uint32_t)(start + body.size()));
    put_u32(table_bytes, (uint32_t)t.second.size());
    body.insert(body.end(), t.second.begin(), t.second.end());
    while (body.size() % 4) body.push_back(0);
  }
  icc.resize(128);
  put_u32(icc, (uint32_t)tags.size());
  icc.insert(icc.end(), table_bytes.begin(), table_bytes.end());
  icc.insert(icc.end(), body.begin(), body.end());
  const uint32_t size = (uint32_t)icc.size();
  for (int k = 0; k < 4; ++k) icc[k] = (uint8_t)(size >> (24 - 8 * k));
  return icc;
}

// Replaces the profile's desc tag by one holding description; false when it has none or its tag table is not whole.
static bool set_icc_description(std::vector<uint8_t>& icc, const std::string& description) {
  if (icc.size() < 132) return false;
  const uint32_t n = get_u32(icc, 128);
  if (132 + 12 * (size_t)n > icc.size()) return false;
  std::vector<std::pair<std::string, std::vector<uint8_t>>> tags;
  bool found = false;
  for (uint32_t i = 0; i < n; ++i) {
    const std::string sig((const char*)icc.data() + 132 + 12 * i, 4);
    const size_t off = get_u32(icc, 136 + 12 * i), size = get_u32(icc, 140 + 12 * i);
    // Inside the file and after the tag table, without overflowing.
    if (off < 132 + 12 * (size_t)n || off > icc.size() || size > icc.size() - off) return false;
    if (sig == "desc") {
      tags.push_back({sig, icc_desc_tag(description)});
      found = true;
    } else {
      tags.push_back({sig, std::vector<uint8_t>(icc.begin() + off, icc.begin() + off + size)});
    }
  }
  if (!found) return false;
  icc = icc_assemble(std::vector<uint8_t>(icc.begin(), icc.begin() + 128), tags);
  return true;
}

std::vector<uint8_t> write_print_icc(const PrintCurves& curves, const std::vector<double>& table,
                                     const std::string& description, const std::string& copyright,
                                     const std::array<int, 6>& date) {
  const int n = (int)curves[0].size();
  std::vector<uint8_t> a2b0 = {'m', 'f', 't', '2', 0, 0, 0, 0, 3, 3, (uint8_t)kPrintIccCube, 0};
  for (int i = 0; i < 9; ++i) put_s15(a2b0, i % 4 == 0 ? 1.0 : 0.0);
  a2b0.push_back((uint8_t)(n >> 8));
  a2b0.push_back((uint8_t)(n & 255));
  a2b0.push_back(0);
  a2b0.push_back(2);
  for (int c = 0; c < 3; ++c)
    for (double v : curves[c]) put_u16(a2b0, v);
  for (double v : table) put_u16(a2b0, v / kPcsMax);
  for (int c = 0; c < 3; ++c) {
    put_u16(a2b0, 0.0);
    put_u16(a2b0, 1.0);
  }
  std::vector<uint8_t> cprt = {'t', 'e', 'x', 't', 0, 0, 0, 0};
  cprt.insert(cprt.end(), copyright.begin(), copyright.end());
  cprt.push_back(0);
  std::vector<uint8_t> wtpt = {'X', 'Y', 'Z', ' ', 0, 0, 0, 0};
  for (double v : kD50) put_s15(wtpt, v);
  std::vector<uint8_t> icc;
  put_u32(icc, 0);   // the size: icc_assemble sets it
  put_u32(icc, 0);
  put_u32(icc, 0x02400000);
  for (const char* s : {"scnr", "RGB ", "XYZ "}) icc.insert(icc.end(), s, s + 4);
  for (int v : date) {
    icc.push_back((uint8_t)(v >> 8));
    icc.push_back((uint8_t)(v & 255));
  }
  icc.insert(icc.end(), {'a', 'c', 's', 'p'});
  icc.insert(icc.end(), 24, 0);
  put_u32(icc, 0);
  for (double v : kD50) put_s15(icc, v);
  icc.insert(icc.end(), 48, 0);
  return icc_assemble(std::move(icc), {{"desc", icc_desc_tag(description)}, {"cprt", std::move(cprt)},
                                       {"wtpt", std::move(wtpt)}, {"A2B0", std::move(a2b0)}});
}

bool set_icc_wtpt(std::vector<uint8_t>& icc, const double xyz[3]) {
  if (icc.size() < 132) return false;
  const uint32_t n = get_u32(icc, 128);
  for (uint32_t i = 0; i < n && 144 + 12 * i <= icc.size(); ++i) {
    if (std::memcmp(icc.data() + 132 + 12 * i, "wtpt", 4) != 0) continue;
    const size_t off = get_u32(icc, 136 + 12 * i);
    if (off + 20 > icc.size() || std::memcmp(icc.data() + off, "XYZ ", 4) != 0) return false;
    for (int c = 0; c < 3; ++c) {
      const int32_t v = (int32_t)std::lround(xyz[c] * 65536.0);
      const uint32_t u = (uint32_t)v;
      icc[off + 8 + 4 * c] = (uint8_t)(u >> 24);
      icc[off + 9 + 4 * c] = (uint8_t)(u >> 16);
      icc[off + 10 + 4 * c] = (uint8_t)(u >> 8);
      icc[off + 11 + 4 * c] = (uint8_t)u;
    }
    return true;
  }
  return false;
}

void print_paper_xyz(const PrintMaterial& m, double xyz[3]) {
  double least[3];
  for (int c = 0; c < 3; ++c) least[c] = *std::min_element(m.curves[c].begin(), m.curves[c].end());
  amounts_to_xyz(m, least, xyz);
}

bool scale_print_icc_table(std::vector<uint8_t>& icc, const double k[3]) {
  const size_t off = a2b0_lut16(icc);
  if (!off) return false;
  const size_t n_in = (size_t)icc[off + 48] << 8 | icc[off + 49];
  const size_t grid = icc[off + 10];
  const size_t at = off + 52 + 6 * n_in;
  const size_t count = grid * grid * grid * 3;
  if (at + 2 * count > icc.size()) return false;
  for (size_t i = 0; i < count; ++i) {
    const size_t p = at + 2 * i;
    const long code = (long)icc[p] << 8 | icc[p + 1];
    const long q = std::min(65535L, std::lround(code * k[i % 3]));
    icc[p] = (uint8_t)(q >> 8);
    icc[p + 1] = (uint8_t)(q & 0xff);
  }
  return true;
}

bool set_print_icc_curves(std::vector<uint8_t>& icc, const PrintCurves& curves) {
  const size_t off = a2b0_lut16(icc);
  if (!off) return false;
  const size_t n = (size_t)icc[off + 48] << 8 | icc[off + 49];
  if (n != curves[0].size() || off + 52 + 6 * n > icc.size()) return false;
  std::vector<uint8_t> codes;
  for (int c = 0; c < 3; ++c)
    for (double v : curves[c]) put_u16(codes, v);
  std::copy(codes.begin(), codes.end(), icc.begin() + (long)(off + 52));
  return true;
}

void PrintContinuum::tilt_at(double a, double t[3]) const { print_tilt_at(a_mid, tilt, a, t); }

PrintCurves PrintContinuum::curves_at(double a, const double gains[3]) const {
  static const double zero[3] = {0.0, 0.0, 0.0};
  if (a_mid.empty()) return bracket_print_curves(g, paper, a, n0, zero, gains);
  double t[3];
  tilt_at(a, t);
  return bracket_print_curves(g, paper, a, n0, t, gains);
}

void PrintContinuum::trc_at(double a, const double gains[3], std::vector<float> trc[3]) const {
  const PrintCurves cv = curves_at(a, gains);
  for (int c = 0; c < 3; ++c) {
    trc[c].resize(cv[c].size());
    for (size_t i = 0; i < cv[c].size(); ++i)
      trc[c][i] = (float)std::lround(std::min(std::max(cv[c][i], 0.0), 1.0) * 65535.0) / 65535.0f;
  }
}

void PrintContinuum::set_curves_at(double a, ParsedFilmProfile& prof, const double gains[3]) const {
  trc_at(a, gains, prof.in_trc);
  for (int c = 0; c < 3; ++c) prof.data.in_trc_size[c] = (int)prof.in_trc[c].size();
  prof.rebind_pointers();
}

double PrintContinuum::a_lo() const { return *std::min_element(g.apd[1].begin(), g.apd[1].end()); }
double PrintContinuum::a_hi() const { return *std::max_element(g.apd[1].begin(), g.apd[1].end()); }

bool bracket_print_profiles(const ProfileBundle& bundle, const std::string& print, BracketPrints& out,
                            std::string& err) {
  if (!bundle.grey_balance.present) {
    return err = "the profile bundle has no grey_balance_curves (a print ICC is computed from them): rebuild it with "
                 "the profile-building tools", false;
  }
  if (!bundle.density_matrix[kDensityApd].present || !bundle.film_model.present) {
    return err = "the profile bundle has no APD matrix or film_model: rebuild it with the profile-building tools", false;
  }
  // The material: a built-in one by id, or a spektrafilm print profile by path.
  PrintMaterial material;
  bool found = false;
  std::string file_text;
  if (is_print_file(print)) {
    std::string stem;
    if (!read_print_file(print, file_text, stem, err)) return false;
    if (!load_spektrafilm_print(file_text, stem, bundle, material, err)) return false;
    found = true;
  } else {
    const EmbeddedFile* f = find_embedded_file("3rd_party/spektrafilm/profiles/" + print + ".json");
    if (f && !load_spektrafilm_print(std::string((const char*)f->data, f->size), print, bundle, material, err)) {
      return false;
    }
    found = f != nullptr;
  }
  if (!found) return err = "no such print material", false;
  const double* p = apd_print_matrix(bundle, material);
  if (!p) return err = "no print matrix for the bundle's film", false;
  const PrintCurves paper = paper_curve(material, p);
  const double n0 = bundle.normalization_target / 65535.0;
  auto bracket_of = [&](const std::string& name) -> const DensityBracket* {
    for (const DensityBracket& b : bundle.density_brackets)
      if (b.name == name) return &b;
    return nullptr;
  };
  // The table: the bundle's print ICC's when it is the file the bundle records, computed from the spektrafilm profile
  // built in here, and its own curves are those computed here for its bracket (the same film, print material and math,
  // to one 16-bit step); else computed here, with a warning saying why.
  const std::string description = bundle.key + " printed on " + material.name + " (spektrafilm)";
  std::vector<uint8_t> base;
  out = BracketPrints{};
  out.id = material.id;
  out.name = material.name;
  if (is_print_file(print)) {
    out.file = print;
    out.file_sha256 = sha256_hex(file_text);
  }
  for (const BundlePrint& bp : bundle.prints) {
    if (is_print_file(print) || bp.id != print) continue;
    // Read now, when the print asks for it (BundlePrint).
    std::vector<uint8_t> icc;
    if (!bp.read_icc || !bp.read_icc(icc, err)) return false;
    const DensityBracket* ref = bracket_of(bp.reference_bracket);
    const size_t off = a2b0_lut16(icc);
    if (!ref || !off) {
      log_warn("The bundle's print ICC for %s %s: its table is computed here\n", print.c_str(),
               !ref ? ("names a reference bracket the bundle has no density bracket for (" + bp.reference_bracket +
                       ")").c_str()
                    : "has no lut16 A2B0 table");
      break;
    }
    const EmbeddedFile* sf = find_embedded_file("3rd_party/spektrafilm/profiles/" + print + ".json");
    const std::string icc_sha = sha256_hex(std::string(icc.begin(), icc.end()));
    const std::string sf_sha = sf ? sha256_hex(std::string((const char*)sf->data, sf->size)) : "";
    if (bp.icc_in_file ? bp.icc_sha256 != icc_sha : !bp.icc_sha256.empty() && bp.icc_sha256 != icc_sha) {
      log_warn("The bundle's print ICC for %s is not the file the bundle records (its SHA-256 %s): its table is "
               "computed here\n", print.c_str(), bp.icc_sha256.empty() ? "is not recorded" : "differs");
      break;
    }
    if (bp.spektrafilm_sha256.empty() || bp.spektrafilm_sha256 != sf_sha) {
      log_warn("The bundle's print ICC for %s was computed from %s spektrafilm profile than the one built in: its "
               "table is computed here\n", print.c_str(), bp.spektrafilm_sha256.empty() ? "an unrecorded" : "another");
      break;
    }
    const PrintCurves mine = bracket_print_curves(bundle.grey_balance, paper, bracket_mid_apd(bundle, *ref), n0, ref->tilt);
    const size_t n = (size_t)icc[off + 48] << 8 | icc[off + 49];
    // The three input curves must lie inside the file: a truncated ICC is not read past its end.
    if (off + 52 + 6 * n > icc.size()) {
      log_warn("The bundle's print ICC for %s is cut short (%zu bytes): its table is computed here\n", print.c_str(),
               icc.size());
      break;
    }
    if (n != mine[0].size()) {
      log_warn("The bundle's print ICC for %s has %zu input curve entries, %zu computed now: its table is computed "
               "here\n", print.c_str(), n, mine[0].size());
      break;
    }
    long worst = 0;
    for (int c = 0; c < 3 && worst <= 1; ++c)
      for (size_t i = 0; i < n; ++i) {
        const size_t at = off + 52 + 2 * (c * n + i);
        const long code = (long)icc[at] << 8 | icc[at + 1];
        worst = std::max(worst, std::labs(code - std::lround(std::min(std::max(mine[c][i], 0.0), 1.0) * 65535.0)));
      }
    if (worst <= 1) {
      // Described as the table computed here is: the material, with no bracket (each bracket's curves and then the
      // print's A replace the file's).
      if (!set_icc_description(icc, description)) {
        log_warn("The bundle's print ICC for %s has no description tag to replace: its table is computed here\n",
                 print.c_str());
        break;
      }
      base = std::move(icc);
      out.from_bundle = true;
    } else {
      log_warn("The bundle's print ICC for %s does not match its film and print material as computed now (curves "
               "differ by %ld): its table is computed here\n", print.c_str(), worst);
    }
    break;
  }
  if (base.empty()) {
    const DensityBracket* mid = bracket_of(bundle.targets.empty() ? "" : bundle.targets[bundle.targets.size() / 2].name);
    if (!mid) return err = "the profile bundle has no density_matrices.brackets entry for its middle bracket", false;
    base = write_print_icc(bracket_print_curves(bundle.grey_balance, paper, bracket_mid_apd(bundle, *mid), n0, mid->tilt),
                           print_table(material, p, bundle.density_matrix[kDensityApd].m, paper),
                           description, kPrintIccCopyright, bundle_date(bundle));
  }
  for (const BundleTarget& t : bundle.targets) {
    const DensityBracket* b = bracket_of(t.name);
    if (!b) return err = "the profile bundle has no density_matrices.brackets entry for " + t.name, false;
    std::vector<uint8_t> icc = base;
    if (!set_print_icc_curves(icc, bracket_print_curves(bundle.grey_balance, paper, bracket_mid_apd(bundle, *b), n0, b->tilt))) {
      return err = "the print ICC has no lut16 A2B0 to write curves into", false;
    }
    out.icc.push_back(std::move(icc));
    out.continuum.a_mid.push_back(bracket_mid_apd(bundle, *b));
    out.continuum.tilt.push_back({b->tilt[0], b->tilt[1], b->tilt[2]});
  }
  out.continuum.g = bundle.grey_balance;
  out.continuum.paper = paper;
  out.continuum.n0 = n0;
  return true;
}

// FIPS 180-4 SHA-256.
std::string sha256_hex(const std::string& bytes) {
  static const uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
      0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
      0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
      0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
      0xc67178f2};
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::string m = bytes;
  const uint64_t bits = (uint64_t)bytes.size() * 8;
  m.push_back((char)0x80);
  while (m.size() % 64 != 56) m.push_back(0);
  for (int i = 7; i >= 0; --i) m.push_back((char)(bits >> (8 * i)));
  auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  for (size_t block = 0; block < m.size(); block += 64) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t)(uint8_t)m[block + 4 * i] << 24 | (uint32_t)(uint8_t)m[block + 4 * i + 1] << 16 |
             (uint32_t)(uint8_t)m[block + 4 * i + 2] << 8 | (uint32_t)(uint8_t)m[block + 4 * i + 3];
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  char out[65];
  for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
  return std::string(out, 64);
}
