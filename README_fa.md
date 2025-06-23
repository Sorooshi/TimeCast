# بسته پیش‌بینی سری‌های زمانی

**🌍 زبان‌ها:** [English](README.md) | [Русский](README_ru.md) | [فارسی](README_fa.md)

بسته‌ای جامع بر اساس PyTorch برای پیش‌بینی سری‌های زمانی که چندین مدل پیشرفته یادگیری عمیق را با تنظیم خودکار فراپارامترها، مدیریت آزمایش‌ها و ردیابی قوی نتایج پیاده‌سازی می‌کند. **ریاضی تأیید شده** با فرمول‌بندی رسمی LaTeX و مطابقت کامل ابعاد.

## 🚀 ویژگی‌های کلیدی

- **چندین مدل پیشرفته**: LSTM، TCN، Transformer، HybridTCNLSTM، MLP
- **تنظیم خودکار فراپارامترها**: با استفاده از Optuna برای جستجوی بهینه پارامترها
- **مدیریت آزمایش‌ها**: ردیابی سازمان‌یافته آزمایش‌ها با توضیحات سفارشی
- **4 حالت آموزش**: گردکار جامع برای کاربردهای مختلف
- **پردازش قوی داده**: پیش‌پردازش تمیز و کارآمد بدون ویژگی‌های زمانی مصنوعی
- **پیش‌پردازش داده‌های تاجران**: پایپ‌لاین کامل برای تبدیل تراکنش به سری زمانی
- **تأیید ریاضی**: سازگاری فرمول‌بندی LaTeX تأیید شده
- **گزارش‌گیری جامع**: ثبت فایل‌های دقیق برای اشکال‌زدایی و تحلیل
- **پشتیبانی چند پلتفرمی**: ایجاد مجموعه‌ای قوی در سیستم‌عامل‌های مختلف
- **تصویرسازی غنی**: منحنی‌های آموزش و نمودارهای ارزیابی
- **معماری ماژولار**: ساختار کد تمیز و قابل نگهداری

## 📐 مبانی ریاضی

این بسته فرمول‌بندی پیش‌بینی سری زمانی شرح‌داده‌شده در مقاله تحقیقاتی ما را پیاده‌سازی می‌کند:

### فرمول‌بندی مسئله
با توجه به داده‌های تراکنش سطح تاجر، ما مصرف کل را با استفاده از دنباله‌های تاریخی پیش‌بینی می‌کنیم:

**نقشه‌برداری LaTeX → پیاده‌سازی:**
- دنباله تاریخی: $\mathcal{H}_t \in \mathbb{R}^{(k+1) \times N}$ ↔ `(sequence_length, n_features)`
- مصرف تاجر: $X_t \in \mathbb{R}^N$ ↔ `merchant_features[t]`
- پیش‌بینی هدف: $y_t = \sum_{m=1}^N x_{m,t}$ ↔ `np.sum(data[t])`

**✅ سازگاری ابعاد تأیید شده:**
```
LaTeX: 𝒽_t ∈ ℝ^{(k+1)×N}  ↔  پیاده‌سازی: (batch_size, sequence_length, n_features)
```

## 📊 مدل‌های پیاده‌سازی‌شده

| مدل | شرح | مورد استفاده | مرجع مقاله |
|-----|-----|--------------|-------------|
| **LSTM** | شبکه حافظه کوتاه-مدت طولانی | یادگیری الگوی ترتیبی | Hochreiter & Schmidhuber (1997) |
| **TCN** | شبکه کانولوشنال زمانی | استخراج ویژگی سلسله‌مراتبی | Bai et al. (2018) |
| **Transformer** | مدل مبتنی بر خودتوجهی | وابستگی‌های زمانی پیچیده | Vaswani et al. (2017) |
| **HybridTCNLSTM** | ترکیب TCN + LSTM | بهترین از هر دو معماری | پیاده‌سازی سفارشی |
| **MLP** | پرسپترون چندلایه | مقایسه خط پایه | Zhang et al. (1998) |

## 🛠️ نصب

1. **کلون کردن مخزن:**
```bash
git clone https://github.com/Sorooshi/Time_Series_Forecasting.git
cd Time_Series_Forecasting
```

2. **ایجاد و فعال‌سازی محیط مجازی:**
```bash
python -m venv venv
source venv/bin/activate  # در ویندوز: venv\Scripts\activate
```

3. **نصب وابستگی‌ها:**
```bash
pip install -r requirements.txt
```

## 📖 استفاده

### شروع سریع با داده‌های تاجران

برای پیش‌پردازش داده‌های تراکنش تاجران (نقطه شروع توصیه‌شده):

```bash
# مرحله 1: اجرای مثال پیش‌پردازش
python example.py

# مرحله 2: آموزش مدل‌ها روی داده‌های پیش‌پردازش‌شده با همه آرگومان‌ها
python main.py --model Transformer \
               --data_name merchant_processed \
               --data_path data/merchant_processed.csv \
               --mode apply_not_tuned \
               --experiment_description "merchant_baseline" \
               --n_trials 100 \
               --epochs 100 \
               --patience 25 \
               --sequence_length 5
```

### رابط خط فرمان

بسته یک CLI جامع با 4 حالت متمایز ارائه می‌دهد:

```bash
python main.py --model <MODEL_NAME> \
               --data_name <DATASET_NAME> \
               --mode <MODE> \
               --experiment_description <DESCRIPTION> \
               [گزینه‌های اضافی]
```

### 🎯 حالت‌های آموزش

| حالت | شرح | چه زمانی استفاده کنید |
|------|-----|---------------------|
| `tune` | تنظیم فراپارامتر + آموزش با بهترین پارامترها | اولین بار با داده/مدل جدید |
| `apply` | آموزش با پارامترهای تنظیم‌شده قبلی (یا پیش‌فرض) | استفاده از پارامترهای تنظیم‌شده موجود |
| `apply_not_tuned` | آموزش فقط با پارامترهای پیش‌فرض | مقایسه خط پایه یا تست سریع |
| `report` | نمایش نتایج ذخیره‌شده از اجراهای قبلی | تحلیل و مقایسه |

### 📋 آرگومان‌ها

#### آرگومان‌های ضروری
- `--model`: نام مدل (LSTM, TCN, Transformer, HybridTCNLSTM, MLP)
- `--data_name`: نام مجموعه داده (بدون پسوند .csv)

#### آرگومان‌های اختیاری
- `--data_path`: مسیر کامل فایل داده (پیش‌فرض: data/{data_name}.csv)
- `--mode`: حالت آموزش (پیش‌فرض: apply)
- `--experiment_description`: توضیح آزمایش سفارشی (پیش‌فرض: seq_len_{sequence_length})
- `--n_trials`: آزمایش‌های تنظیم فراپارامتر (پیش‌فرض: 100)
- `--epochs`: دوره‌های آموزش (پیش‌فرض: 100)
- `--patience`: صبر توقف زودهنگام (پیش‌فرض: 25)
- `--sequence_length`: طول دنباله ورودی (پیش‌فرض: 10)

## 🏪 پیش‌پردازش داده‌های تاجران

### پایپ‌لاین پیش‌پردازش (`example.py`)

پایپ‌لاین کامل برای تبدیل داده‌های تراکنش خام تاجران به فرمت سری زمانی:

```bash
python example.py
```

**مراحل پایپ‌لاین:**
1. **بارگیری داده‌های تراکنش**: بارگیری داده‌های سطح تراکنش خام
2. **تجمیع تاجران**: گروه‌بندی بر اساس دوره‌های زمانی و تاجران
3. **ویژگی‌های زمینه‌ای**: اضافه کردن ویژگی‌های مبتنی بر زمان (فصلی، تعطیلات و غیره)
4. **سازگاری LaTeX**: اطمینان از مطابقت ابعاد
5. **تأیید**: تست با TimeSeriesPreprocessor

**فرمت ورودی:**
```csv
timestamp,merchant_id,customer_id,amount,day_of_week,hour,is_weekend,is_holiday,transaction_speed,customer_loyalty_score
2023-01-01 03:41:00,1,23,16.02,6,3,True,False,8.87,79.8
2023-01-01 06:28:00,4,25,99.56,6,6,True,False,5.9,48.8
...
```

**فرمت خروجی:**
```csv
date,merchant_1,merchant_2,merchant_3,merchant_4,merchant_5,hour,day_of_week,is_weekend,month,day_of_month,sin_month,cos_month,sin_hour,cos_hour,is_holiday
2023-01-01,454.17,207.98,216.56,460.11,644.78,0,5,1.0,1,1,0.0,1.0,0.0,1.0,1.0
2023-01-02,423.89,189.45,234.12,501.23,678.91,0,0,0.0,1,2,0.0,1.0,0.0,1.0,0.0
...
```

## 🧪 تست و تأیید

### مجموعه تست جامع

فراخوانی کامل تست‌های اعتبارسنجی:

```bash
cd Test/
python test_script.py
```

**نتایج تست:** ✅ 100% موفقیت (11/11 تست)

### تست‌های تأیید ریاضی

- ✅ **تأیید ابعاد**: LaTeX ↔ پیاده‌سازی
- ✅ **محاسبه هدف**: `np.sum()` درست
- ✅ **فرمت داده**: تطابق بعد کامل
- ✅ **یکپارچگی پایپ‌لاین**: پردازش کامل

## 💡 گردکارهای مثال

### 1. گردکار کامل داده‌های تاجران

```bash
# مرحله 1: پیش‌پردازش داده‌های تاجران
python example.py

# مرحله 2: تنظیم فراپارامتر
python main.py --model Transformer \
               --data_name merchant_processed \
               --mode tune \
               --experiment_description "merchant_baseline" \
               --n_trials 50 \
               --epochs 100 \
               --sequence_length 5

# مرحله 3: اعمال با پارامترهای تنظیم‌شده
python main.py --model Transformer \
               --data_name merchant_processed \
               --mode apply \
               --experiment_description "merchant_tuned" \
               --epochs 100 \
               --sequence_length 5

# مرحله 4: مقایسه با پارامترهای پیش‌فرض
python main.py --model Transformer \
               --data_name merchant_processed \
               --mode apply_not_tuned \
               --experiment_description "merchant_default" \
               --epochs 100 \
               --sequence_length 5

# مرحله 5: مشاهده همه نتایج
python main.py --model Transformer \
               --data_name merchant_processed \
               --mode report \
               --experiment_description "merchant_baseline"
```

### 2. گردکار تست سریع

```bash
# تست سریع با پارامترهای پیش‌فرض
python main.py --model LSTM \
               --data_name my_data \
               --mode apply_not_tuned \
               --experiment_description "quick_test" \
               --epochs 20 \
               --sequence_length 5
```

## 📁 ساختار پروژه

```
Time_Series_Forecasting/
├── data/                    # فایل‌های مجموعه داده
├── models/                  # پیاده‌سازی مدل‌ها
├── utils/                   # ابزارهای کمکی
├── History/                 # تاریخچه آموزش
├── Hyperparameters/         # پارامترهای تنظیم‌شده
├── Results/                 # خلاصه نتایج
├── Plots/                   # تصویرسازی‌ها
├── Logs/                    # فایل‌های ثبت
├── Test/                    # مجموعه تست‌ها
├── main.py                  # نقطه ورود اصلی
├── example.py               # پیش‌پردازش تاجران
└── requirements.txt         # وابستگی‌ها
```

## 🤝 مشارکت

1. فورک پروژه
2. شاخه ویژگی ایجاد کنید (`git checkout -b feature/AmazingFeature`)
3. تغییرات خود را کامیت کنید (`git commit -m 'Add some AmazingFeature'`)
4. به شاخه پوش کنید (`git push origin feature/AmazingFeature`)
5. Pull Request باز کنید

## 📝 مجوز

این پروژه تحت مجوز MIT منتشر شده است. برای جزئیات بیشتر فایل [LICENSE](LICENSE) را ببینید.

## 📧 تماسs

سروش شلیله - [sorooshshalileh@example.com](mailto:sr.shalileh@gmail.com)

لینک پروژه: [https://github.com/Sorooshi/Time_Series_Forecasting](https://github.com/Sorooshi/Time_Series_Forecasting)

## 🎖️ تشکر

- تیم [PyTorch](https://pytorch.org/) برای فریمورک عالی
- توسعه‌دهندگان [Optuna](https://optuna.org/) برای کتابخانه تنظیم فراپارامتر
- جامعه متن‌باز برای ابزارها و کتابخانه‌های بی‌نظیر

---

**📊 آماده برای تحقیق و تولید** | **🔬 تأیید ریاضی** | **🌍 پشتیبانی چندزبانه** 