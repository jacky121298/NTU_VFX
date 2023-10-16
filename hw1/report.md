# VFX HW1 -- High Dynamic Range Imaging (Group 20)

#### 學號：R10922081 ｜ 系級：資工碩一 ｜ 姓名：鄒宗霖

---

## Image Alignment

Implement Ward's Median Threshold Bitmap (MTB) alignment technique.

### 1. Convert RGB image into gray image & constrcut MTB and exclusion map

> 先將彩色影像轉到灰階影像，並建立 Median Threshold Bitmap (MTB) 以及 exclusion map (與 median 太接近的那些值我們不去採納，下圖中白色的部分是有被真正採納的部分)。

|                             MTB                              |                        exclusion map                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![mtb](/Users/jacky/Desktop/Programming/M1/VFX/homework/Digital_Visual_Effects_2022Spring/hw1/images/mtb.jpg) | ![exclusion](/Users/jacky/Desktop/Programming/M1/VFX/homework/Digital_Visual_Effects_2022Spring/hw1/images/exclusion.jpg) |

### 2. Use multiscale technique

> 建立 $log(max\_offset)$ 層金字塔，從金字塔最上層出發，每一層以第一張影像（shutter speed 最小）為參考影像，位移九個方向並計算 error，傳遞到下一層時位移量乘二。

|                           level 4                            |                           level 5                            |                           level 6                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![level_4](/Users/jacky/Desktop/Programming/M1/VFX/homework/Digital_Visual_Effects_2022Spring/hw1/images/level_4.jpg) | ![level_5](/Users/jacky/Desktop/Programming/M1/VFX/homework/Digital_Visual_Effects_2022Spring/hw1/images/level_5.jpg) | ![level_6](/Users/jacky/Desktop/Programming/M1/VFX/homework/Digital_Visual_Effects_2022Spring/hw1/images/level_6.jpg) |

### 3. Compute error & choose best shift

>計算位移過後的影像與參考影像的差異，並只考慮 exclusion map 中白色的部分。

$error = (mtb_1)\ XOR\ (shifted\_mtb_2)\ AND\ (exclusion\ map)$

## Construct HDR image

Implement Debevec's method.

### 1. Compute response curve

> 利用課堂上所學的 Debevec's 公式建立 A, b 矩陣，並利用 pseudo inverse 求出 $g$, $E_i$。

<img src="/Users/jacky/Desktop/Programming/M1/VFX/homework/Digital_Visual_Effects_2022Spring/hw1/images/crc.jpg" alt="crc" style="zoom:72%;" />

### 2. Compute radiance map

> 利用所有不同曝光值的影像以及剛剛求出的 camera response curve 計算 HDR radiance map。

<img src="/Users/jacky/Desktop/Programming/M1/VFX/homework/Digital_Visual_Effects_2022Spring/hw1/images/hdr.jpg" alt="hdr" style="zoom:72%;" />

## Tone mapping

Implement gamma tone mapping and durand's tone mapping.

### 1. Gamma tone mapping

> 將 HDR 影像利用公式： $image = image ^ {1 \over \gamma}$ 得到 tone mapping 後的結果。

<img src="/Users/jacky/Desktop/Programming/M1/VFX/homework/Digital_Visual_Effects_2022Spring/hw1/images/gamma.jpg" alt="gamma" style="zoom: 10%;" />

### 2. Durand's tone mapping

> 先將 HDR 影像拆成 intensity 以及 color，再利用 bilateral filter 將 intensity 拆成 base 以及 detail，然後在 log domain 下壓縮 base 的對比，最後將壓縮後的 intensity 以及 color 結合得到 tone mapping 後的結果。

#### 2-1. Seperate intensity & color

|                          intensity                           |                            color                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="/Users/jacky/Desktop/Programming/M1/VFX/homework/Digital_Visual_Effects_2022Spring/hw1/images/lum.jpg" alt="lum" style="zoom:10%;" /> | <img src="/Users/jacky/Desktop/Programming/M1/VFX/homework/Digital_Visual_Effects_2022Spring/hw1/images/clr.jpg" alt="clr" style="zoom:10%;" /> |

#### 2-2. Seperate base & detail

|                             base                             |                            detail                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="/Users/jacky/Desktop/Programming/M1/VFX/homework/Digital_Visual_Effects_2022Spring/hw1/images/lum_base.jpg" alt="lum_base" style="zoom:10%;" /> | <img src="/Users/jacky/Desktop/Programming/M1/VFX/homework/Digital_Visual_Effects_2022Spring/hw1/images/lum_detail.jpg" alt="lum_detail" style="zoom:10%;" /> |

#### 2-3. Reconstruct intensity

<img src="/Users/jacky/Desktop/Programming/M1/VFX/homework/Digital_Visual_Effects_2022Spring/hw1/images/lum2.jpg" alt="lum2" style="zoom:10%;" />

#### 2-4. Durand's tone mapping result

<img src="/Users/jacky/Desktop/Programming/M1/VFX/homework/Digital_Visual_Effects_2022Spring/hw1/images/durand.jpg" alt="durand" style="zoom:10%;" />