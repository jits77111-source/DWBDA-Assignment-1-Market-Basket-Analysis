# DWBDA Assignment 1: Market Basket Analysis using Apriori Algorithm

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jits77111-source/DWBDA-Assignment-1-Market-Basket-Analysis/blob/main/Assignment_1_Market_Basket_Analysis.ipynb)

**Author:** Jit Sarkar  
**Course:** Data Warehousing, Big Data & Analytics  
**Implementation:** Apriori Algorithm for Frequent Itemset Mining & Association Rule Discovery  

---

## 📌 Problem Overview
Market Basket Analysis (MBA) is an unsupervised data mining technique utilized by retailers and e-commerce platforms to discover affinities and co-occurrence relationships among products purchased together. 

This project implements the **Apriori Algorithm** using `mlxtend` on a dataset of customer electronics and workstation accessory orders.

---

## 📂 Repository Contents
```
DWBDA-Assignment-1-Market-Basket-Analysis/
├── Assignment_1_Market_Basket_Analysis.ipynb   # Interactive Jupyter Notebook (Colab Ready)
├── assignment_1_mba.py                         # Standalone executable Python script
├── computer_accessories_transactions.csv      # Retail transaction dataset
├── mba_rules_summary.csv                      # Output table of discovered association rules
├── mba_visualization.png                      # Scatter and bar chart plots
├── .gitignore                                 # Git ignore file
└── README.md                                  # Documentation
```

---

## 📊 Dataset Description
The dataset (`computer_accessories_transactions.csv`) includes transaction baskets from student, professional, and gaming customer segments with items:
- `Mechanical Keyboard`
- `Wireless Mouse` / `Gaming Mouse`
- `Mousepad XL`
- `Laptop Stand`
- `USB-C Hub`
- `HDMI Cable`
- `Webcam 1080p`
- `External SSD`
- `Headset`
- `Desk Lamp`

---

## 🚀 How to Run

### Method 1: 1-Click Run in Google Colab (Recommended)
Click the badge above or visit:  
👉 [Open in Google Colab](https://colab.research.google.com/github/jits77111-source/DWBDA-Assignment-1-Market-Basket-Analysis/blob/main/Assignment_1_Market_Basket_Analysis.ipynb)

### Method 2: Run Locally
```bash
# 1. Clone repository
git clone https://github.com/jits77111-source/DWBDA-Assignment-1-Market-Basket-Analysis.git
cd DWBDA-Assignment-1-Market-Basket-Analysis

# 2. Install dependencies
pip install pandas mlxtend matplotlib

# 3. Execute script
python assignment_1_mba.py
```

---

## 📈 Key Discovered Association Rules
| Antecedents | Consequents | Support | Confidence | Lift |
| :--- | :--- | :---: | :---: | :---: |
| `{Mechanical Keyboard, Gaming Mouse}` | `{Mousepad XL}` | 0.233 | 0.875 | **2.02** |
| `{Laptop Stand, USB-C Hub}` | `{HDMI Cable}` | 0.200 | 0.857 | **1.98** |
| `{Mechanical Keyboard}` | `{Mousepad XL}` | 0.300 | 0.750 | **1.73** |
