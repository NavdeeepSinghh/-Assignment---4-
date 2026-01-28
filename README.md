# 📊 Learning Probability Density Functions using GAN

## 🧾 Assignment Overview
This project focuses on learning an **unknown probability density function (PDF)** of a transformed random variable using a **Generative Adversarial Network (GAN)**.  
The probability distribution is learned **only from data samples**, without assuming any analytical or parametric form.

The dataset used consists of **NO₂ concentration values**, and the goal is to model the PDF of a **nonlinearly transformed variable** using adversarial learning.

---

## 🔢 Transformation Parameters ((a_r, b_r))

The transformation applied to the NO₂ concentration data \((x)\) is:

\[
z = x + a_r \sin(b_r x)
\]

The parameters are derived from the university roll number.

- 🎓 **Roll Number:** `102303787`

\[
a_r = 0.5 \times (r \bmod 7)
\]

\[
b_r = 0.3 \times ((r \bmod 5) + 1)
\]

✅ The values of **\(a_r\)** and **\(b_r\)** are **computed programmatically in the code**, ensuring reproducibility and strict adherence to the assignment guidelines.


## 🧠 GAN Architecture Description

A **vanilla GAN** is used to implicitly learn the probability density of the transformed variable \(z\).

### 🔹 Generator
- **Input:** Gaussian noise \(\epsilon \sim \mathcal{N}(0,1)\)
- **Architecture:**
  - Fully connected layers: `1 → 128 → 128 → 1`
  - ReLU activation functions
- **Objective:** Generate samples that resemble the real transformed data distribution

### 🔹 Discriminator
- **Input:** Real or generated samples of \(z\)
- **Architecture:**
  - Fully connected layers: `1 → 128 → 128 → 1`
  - LeakyReLU activation functions
  - Sigmoid output layer
- **Objective:** Distinguish real samples from generated samples

⚠️ No analytical or parametric probability distribution (Gaussian, exponential, etc.) is assumed at any stage.

---

## 📈 Learned PDF using GAN

After training the GAN, a large number of samples are generated from the generator using Gaussian noise.  
Since GANs do not provide an explicit probability density function, the PDF is estimated using **Kernel Density Estimation (KDE)**.

The comparison includes:
- **True PDF:** KDE estimated from real transformed data  
- **Learned PDF:** KDE estimated from GAN-generated samples  

### 📊 Final PDF Comparison Plot
*(See `GAN_Estimated_pmf.png` in this repository)*

---

## 🔍 Observations

### 🧩 Mode Coverage
The GAN successfully captures the dominant structure of the transformed data distribution.  
The major modes introduced by the sinusoidal transformation are represented, with **no evidence of mode collapse**.

### ⚖️ Training Stability
Training remains stable throughout the process. Generator and discriminator losses remain bounded and do not diverge, indicating a balanced adversarial learning process.

### 🎯 Quality of Generated Distribution
The learned distribution closely approximates the empirical distribution of the transformed data.  
Minor deviations are observed near the tails and peak amplitudes, which are expected due to adversarial training dynamics and KDE smoothing.

---

## ✨ Conclusion
This project demonstrates the effectiveness of **Generative Adversarial Networks** for **implicit density learning** when explicit probability models are unavailable.  
GANs, combined with KDE, provide a practical approach for approximating complex and unknown probability distributions using data alone.

---

## 🛠️ Technologies Used
- Python  
- PyTorch  
- NumPy  
- Pandas  
- Matplotlib  
- SciPy (KDE)  
- Google Colab  

---

## 📂 Repository Contents
- `Navdeep_singh_assignment_4.ipynb` — Complete implementation notebook  
- `GAN_Estimated_pmf.png` — Final PDF comparison plot  
- `README.md` — Project documentation  
